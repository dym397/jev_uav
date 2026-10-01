"""14-value kinematic JevHarness for UAV tactical conflict resolution.

Strictly consumes only ``state.numeric`` (the 14-dimensional paper observation)
and its own internal episode memory of ``state.numeric``. Physical metadata on
``Observation`` (``own_position``, ``goal``, ``due_time``, ``contacts``, etc.)
is never accessed.
"""

from __future__ import annotations

from math import atan2, cos, degrees, exp, hypot, isfinite, log, pi, radians, sin
from time import perf_counter
from typing import Callable

import numpy as np

from config import ACTION_SPACE, Action, EnvConfig, action_control
from environment.collision import point_to_segment_distance
from environment.observation import wrap_angle


HARNESS_POLICY_QUESTION = (
    "Choose the next action that maximizes the probability of completing the "
    "stated task successfully before its deadline. Use the visible state, "
    "action descriptions, remaining time, and recorded history."
)

THREAT_SCORE_QUESTION = (
    "Assess the tactical collision risk over the next 3 seconds from the "
    "visible sector distances and closing rates."
)

THREAT_SCORE_CRITERIA = [
    "Level 0 (clear): no closing intruder within 55 m in forward or lateral sectors.",
    "Level 1 (caution): intruder detected within 55 m; separation remains above 22 m.",
    "Level 2 (warning): closing intruder within 35 m; evasive steering or speed adjustment required.",
    "Level 3 (critical): immediate conflict within 16 m; maximum separation maneuver required.",
]

SAFETY_BOOLEAN_QUESTION = (
    "Will the UAV maintain at least 10 m separation from all detected sector "
    "threats during the next 3 seconds while progressing toward the goal?"
)

SAFETY_BOOLEAN_CRITERIA = {
    "true": "Minimum predicted separation stays above the 10 m collision boundary.",
    "false": "Separation drops near or below the 10 m collision boundary.",
}


class UAVJevHarness:
    """Deterministic 14-value kinematic tracker, projector, and Jev request builder."""

    def __init__(self, config: EnvConfig | None = None, *, episode_horizon: int = 30):
        self.config = config or EnvConfig()
        self.episode_horizon = episode_horizon
        self.reset_memory()

    def reset_memory(self) -> None:
        self.step_index = 0
        self.own_pos = (0.0, 0.0)
        self.prev_heading: float | None = None
        self.prev_speed: float | None = None
        self.prev_goal_dist_ratio: float | None = None
        self.prev_action: int | None = None
        self.prev_sectors: list[float] | None = None
        self.leg_length = 100.0
        self.due_time = 100.0 / 6.0
        self.init_vec: tuple[float, float] | None = None

    def _is_new_episode(self, heading: float, speed: float, goal_dist_ratio: float) -> bool:
        if (
            self.prev_heading is None
            or self.prev_speed is None
            or self.prev_goal_dist_ratio is None
            or self.prev_action is None
        ):
            return True
        yaw_rate, acc = action_control(self.prev_action)
        dt = self.config.dt
        expected_speed = min(max(self.prev_speed + acc * dt, self.config.speed_min), self.config.speed_max)
        expected_heading = wrap_angle(self.prev_heading + yaw_rate * dt)
        if abs(speed - expected_speed) > 1e-2 or abs(wrap_angle(heading - expected_heading)) > 1e-2:
            return True
        if goal_dist_ratio > self.prev_goal_dist_ratio + 0.15:
            return True
        return False

    def analyze(self, state) -> dict:
        """Extract unwrapped state, track sector range rates, and project all 9 actions."""
        values = np.asarray(state.numeric, dtype=np.float64)
        if values.shape != (14,):
            raise ValueError("UAVJevHarness requires exactly 14 numeric observation values")
        heading, speed, goal_dist_ratio, goal_bearing_ccw, eta_ratio = map(float, values[:5])
        sectors = [float(x) for x in values[5:]]

        dt = self.config.dt
        if self._is_new_episode(heading, speed, goal_dist_ratio):
            self.reset_memory()
            self.leg_length = 100.0
            if goal_dist_ratio > 1e-6 and eta_ratio > 1e-6:
                self.due_time = (self.leg_length * goal_dist_ratio / max(speed, self.config.speed_min)) / eta_ratio
            goal_world_angle = heading + goal_bearing_ccw
            self.init_vec = (goal_dist_ratio * cos(goal_world_angle), goal_dist_ratio * sin(goal_world_angle))
            v_ref_x, v_ref_y = speed * cos(heading), speed * sin(heading)
        else:
            v_ref_x = self.prev_speed * cos(self.prev_heading)
            v_ref_y = self.prev_speed * sin(self.prev_heading)
            self.own_pos = (
                self.own_pos[0] + v_ref_x * dt,
                self.own_pos[1] + v_ref_y * dt,
            )
            self.step_index += 1
            if self.step_index == 1 and self.init_vec is not None:
                goal_world_angle = heading + goal_bearing_ccw
                cur_vec = (goal_dist_ratio * cos(goal_world_angle), goal_dist_ratio * sin(goal_world_angle))
                diff = hypot(self.init_vec[0] - cur_vec[0], self.init_vec[1] - cur_vec[1])
                step_dist = hypot(self.own_pos[0], self.own_pos[1])
                if diff > 1e-4:
                    self.leg_length = step_dist / diff

        goal_world_angle = heading + goal_bearing_ccw
        goal_dist_m = self.leg_length * goal_dist_ratio
        goal_pos = (
            self.own_pos[0] + goal_dist_m * cos(goal_world_angle),
            self.own_pos[1] + goal_dist_m * sin(goal_world_angle),
        )
        signed_goal_bearing = wrap_angle(goal_bearing_ccw)
        now_s = self.step_index * dt
        rem_due_s = self.due_time - now_s

        threat_particles = []
        sector_summaries = []
        for idx, ratio in enumerate(sectors):
            if ratio >= 1.0:
                continue
            dist_m = ratio * self.config.detection_radius
            rel_angle = wrap_angle(idx * (2 * pi / 9))
            world_angle = wrap_angle(heading + rel_angle)
            ux, uy = cos(world_angle), sin(world_angle)

            default_closing = max(speed * cos(rel_angle) + 3.0, 2.5)
            closing_mps = default_closing
            if self.prev_sectors is not None and self.prev_heading is not None and self.prev_speed is not None:
                prev_own_x = self.own_pos[0] - v_ref_x * dt
                prev_own_y = self.own_pos[1] - v_ref_y * dt
                matched_rates = []
                for j in (idx, (idx - 1) % 9, (idx + 1) % 9):
                    if self.prev_sectors[j] >= 1.0:
                        continue
                    prev_d = self.prev_sectors[j] * self.config.detection_radius
                    prev_w_ang = wrap_angle(self.prev_heading + j * (2 * pi / 9))
                    qx = prev_own_x + prev_d * cos(prev_w_ang)
                    qy = prev_own_y + prev_d * sin(prev_w_ang)
                    d_from_cur_own = hypot(qx - self.own_pos[0], qy - self.own_pos[1])
                    if abs(dist_m - d_from_cur_own) <= 8.0:
                        matched_rates.append((abs(dist_m - d_from_cur_own), (prev_d - dist_m) / dt))
                if matched_rates:
                    matched_rates.sort(key=lambda item: item[0])
                    closing_mps = matched_rates[0][1]

            sector_summaries.append({
                "sector": idx,
                "angle_ccw_deg": idx * 40,
                "signed_angle_deg": round(degrees(rel_angle), 1),
                "distance_m": round(dist_m, 2),
                "closing_mps": round(closing_mps, 2),
            })

            is_forward_cone = idx in (0, 1, 8)
            is_lateral = idx in (2, 7)
            if (is_forward_cone and dist_m < 55.0 and (closing_mps > 0.2 or (dist_m < 28.0 and closing_mps > -1.0))) or (
                is_lateral and dist_m < 32.0 and closing_mps > 0.8
            ):
                v_own_par = v_ref_x * ux + v_ref_y * uy
                v_intr_par = max(min(v_own_par - closing_mps, 5.5), -5.5)
                v_perp_mag = (max(4.2 ** 2 - v_intr_par ** 2, 0.0)) ** 0.5
                sign_to_center = -1.0 if sin(rel_angle) >= 0 else 1.0
                tx, ty = sign_to_center * (-uy), sign_to_center * ux

                vel_hypotheses = [
                    (v_intr_par * ux + v_perp_mag * tx, v_intr_par * uy + v_perp_mag * ty, 1.0),
                    (v_intr_par * ux + 0.4 * v_perp_mag * tx, v_intr_par * uy + 0.4 * v_perp_mag * ty, 0.95),
                    (v_intr_par * ux, v_intr_par * uy, 0.9),
                ]
                for ang_off in (-9.0, 0.0, 9.0):
                    ang = world_angle + radians(ang_off)
                    px = self.own_pos[0] + dist_m * cos(ang)
                    py = self.own_pos[1] + dist_m * sin(ang)
                    for vx, vy, w in vel_hypotheses:
                        threat_particles.append((idx, dist_m, px, py, vx, vy, w))

        self.prev_sectors = sectors
        self.prev_heading = heading
        self.prev_speed = speed
        self.prev_goal_dist_ratio = goal_dist_ratio

        p1 = (
            self.own_pos[0] + speed * dt * cos(heading),
            self.own_pos[1] + speed * dt * sin(heading),
        )
        reached_on_p1 = point_to_segment_distance(goal_pos, self.own_pos, p1) <= self.config.goal_radius
        horizon_limit = min(self.config.max_steps, self.episode_horizon)
        max_rem_steps = max(1, horizon_limit - self.step_index - 1)

        candidates = []
        for a in range(9):
            yaw_rate, acc = action_control(a)
            v1 = min(max(speed + acc * dt, self.config.speed_min), self.config.speed_max)
            h1 = wrap_angle(heading + yaw_rate * dt)

            cur_p = p1
            cur_h = h1
            cur_v = v1
            min_clr = 100.0
            step1_clr = 100.0
            min_goal_miss = point_to_segment_distance(goal_pos, self.own_pos, p1)
            arrival_step = 1 if reached_on_p1 else None

            for tau in range(1, max_rem_steps + 1):
                if tau > 1:
                    g_ang = atan2(goal_pos[1] - cur_p[1], goal_pos[0] - cur_p[0])
                    diff = wrap_angle(g_ang - cur_h)
                    turn = max(min(diff, pi / 30), -pi / 30)
                    cur_h = wrap_angle(cur_h + turn)
                    rem_d = hypot(goal_pos[0] - cur_p[0], goal_pos[1] - cur_p[1])
                    if abs(diff) > 0.22 and rem_d < 35.0:
                        cur_v = max(cur_v - 3.0 * dt, 6.0)
                    elif cur_v < 9.0:
                        cur_v = min(cur_v + 3.0 * dt, self.config.speed_max)

                next_p = (cur_p[0] + cur_v * dt * cos(cur_h), cur_p[1] + cur_v * dt * sin(cur_h))

                if tau <= 3 and threat_particles:
                    for idx, d_m, px, py, vx, vy, weight in threat_particles:
                        ix0 = px + vx * tau * dt
                        iy0 = py + vy * tau * dt
                        ix1 = px + vx * (tau + 1) * dt
                        iy1 = py + vy * (tau + 1) * dt
                        for frac in (0.0, 0.5, 1.0):
                            ux = cur_p[0] + frac * (next_p[0] - cur_p[0])
                            uy = cur_p[1] + frac * (next_p[1] - cur_p[1])
                            ix = ix0 + frac * (ix1 - ix0)
                            iy = iy0 + frac * (iy1 - iy0)
                            c = hypot(ux - ix, uy - iy)
                            if tau == 1 and weight == 1.0:
                                step1_clr = min(step1_clr, c)
                            eff_c = c + 0.9 * (tau - 1) + (1.5 if weight < 1.0 else 0.0)
                            min_clr = min(min_clr, eff_c)

                seg_goal_d = point_to_segment_distance(goal_pos, cur_p, next_p)
                min_goal_miss = min(min_goal_miss, seg_goal_d)
                if arrival_step is None and seg_goal_d <= self.config.goal_radius:
                    arrival_step = tau + 1
                    break
                cur_p = next_p

            reaches_goal = arrival_step is not None
            est_total_time = (self.step_index + (arrival_step or max_rem_steps)) * dt
            eta_err = abs(est_total_time - self.due_time)
            on_time = reaches_goal and (eta_err <= self.config.on_time_window)

            safe = min_clr >= 12.0 and step1_clr >= 11.2
            if min_clr < 10.8 or step1_clr < 10.5:
                safety_score = -2000.0 + 60.0 * min(min_clr, step1_clr)
            elif min_clr < 15.5:
                safety_score = -15.0 * (15.5 - min_clr) ** 2
            elif min_clr < 22.0:
                safety_score = -1.5 * (22.0 - min_clr)
            else:
                safety_score = 0.0

            goal_bearing_p1 = atan2(goal_pos[1] - p1[1], goal_pos[0] - p1[0])
            heading_err = abs(wrap_angle(goal_bearing_p1 - h1))

            goal_score = 0.0
            if reaches_goal:
                goal_score += 200.0
                if on_time:
                    goal_score += 60.0 - 2.5 * eta_err
                else:
                    goal_score -= 8.0 * eta_err
            else:
                goal_score -= 25.0 * min_goal_miss

            if v1 < 2.5:
                speed_score = -220.0
            elif v1 < 5.5:
                speed_score = -8.0 if not threat_particles else -2.0
            else:
                speed_score = -0.8 * abs(v1 - 9.0)

            clr_bonus = 1.0 * min(min_clr, 26.0) + 0.6 * min(step1_clr, 26.0) if threat_particles else 0.0
            utility = safety_score + goal_score - 18.0 * heading_err + speed_score + clr_bonus

            candidates.append({
                "action": a,
                "name": ACTION_SPACE[a],
                "v1": round(v1, 2),
                "h1_deg": round(degrees(h1), 1),
                "heading_err_deg": round(degrees(heading_err), 1),
                "min_clr_m": round(min_clr, 1),
                "step1_clr_m": round(step1_clr, 1),
                "reaches_goal": reaches_goal,
                "on_time": on_time,
                "eta_err_s": round(eta_err, 1),
                "safe": safe,
                "utility": round(utility, 3),
            })

        candidates.sort(key=lambda c: c["utility"], reverse=True)
        return {
            "raw_text": state.state_to_text(),
            "step": self.step_index,
            "heading_deg": round(degrees(heading), 1),
            "speed_mps": round(speed, 2),
            "goal_dist_m": round(goal_dist_m, 1),
            "signed_goal_bearing_deg": round(degrees(signed_goal_bearing), 1),
            "rem_due_s": round(rem_due_s, 1),
            "active_sectors": sector_summaries,
            "has_active_threat": bool(threat_particles),
            "ranked_candidates": candidates,
        }

    def format_state_text(self, analysis: dict, *, include_top_candidates: bool = True) -> str:
        """Render the 14-value observation and its deterministic kinematic summary."""
        if analysis["active_sectors"]:
            sectors_text = "; ".join(
                f"s{s['sector']}({s['signed_angle_deg']:+.0f}deg)={s['distance_m']:.1f}m"
                f"(closing={s['closing_mps']:+.1f}m/s)"
                for s in analysis["active_sectors"]
            )
        else:
            sectors_text = "clear_within_100m"
        top3 = ", ".join(
            f"{c['action']}:{c['name']}[v1={c['v1']}m/s,clr={c['min_clr_m']}m,"
            f"herr={c['heading_err_deg']}deg,on_time={c['on_time']}]"
            for c in analysis["ranked_candidates"][:3]
        )
        state_text = (
            f"{analysis['raw_text']}\n"
            f"Harness14D: step={analysis['step']}; goal_dist_m={analysis['goal_dist_m']}; "
            f"signed_goal_bearing_deg={analysis['signed_goal_bearing_deg']:+.1f}; "
            f"rem_schedule_s={analysis['rem_due_s']:+.1f}; threats=[{sectors_text}]"
        )
        return f"{state_text}; top_candidates=[{top3}]" if include_top_candidates else state_text

    def build_request(
        self,
        analysis: dict,
        *,
        top_k: int = 3,
        include_diagnostics: bool = True,
        offered_candidates: list[dict] | None = None,
        include_top_candidates: bool = True,
    ) -> dict:
        """Build a native NanoJev request with pruned or full candidates and diagnostic questions."""
        if not 2 <= top_k <= 9:
            raise ValueError("top_k must be between 2 and 9")
        offered = analysis["ranked_candidates"][:top_k] if offered_candidates is None else offered_candidates
        if not 2 <= len(offered) <= 9:
            raise ValueError("a NanoJev choice requires between 2 and 9 candidates")
        criteria = {
            str(c["action"]): (
                f"{c['name']}: next_speed={c['v1']} m/s, goal_heading_error={c['heading_err_deg']} deg, "
                f"predicted_3s_min_clearance={c['min_clr_m']} m, reaches_goal={c['reaches_goal']}, "
                f"on_time={c['on_time']}."
            )
            for c in offered
        }
        questions: dict = {
            "action": {
                "type": "choice",
                "instructions": HARNESS_POLICY_QUESTION,
                "criteria": criteria,
            }
        }
        if include_diagnostics:
            questions["threat_level"] = {
                "type": "score",
                "instructions": THREAT_SCORE_QUESTION,
                "criteria": list(THREAT_SCORE_CRITERIA),
            }
            questions["safety_margin"] = {
                "type": "boolean",
                "instructions": SAFETY_BOOLEAN_QUESTION,
                "criteria": dict(SAFETY_BOOLEAN_CRITERIA),
            }
        return {
            "states": [{
                "id": "uav",
                "state": self.format_state_text(analysis, include_top_candidates=include_top_candidates),
                "questions": questions,
            }]
        }


def parse_harness_response(response: dict, offered_ids: list[str]) -> dict | None:
    """Validate choice probabilities over offered_ids and optional score/boolean answers."""
    try:
        rows = response["states"]
        if len(rows) != 1 or rows[0]["id"] != "uav":
            return None
        answers = rows[0]["answers"]
        action_ans = answers["action"]
        probs = action_ans["probabilities"]
        choice = action_ans["choice"]
        if action_ans["type"] != "choice" or type(choice) is not str:
            return None
        if set(probs) != set(offered_ids) or choice not in probs:
            return None
        vals = [probs[k] for k in offered_ids]
        if any(type(p) not in (float, int) or not isfinite(p) or not 0.0 <= p <= 1.0 for p in vals):
            return None
        if abs(sum(vals) - 1.0) > 1e-4:
            return None
        if probs[choice] != max(vals):
            return None

        parsed = {
            "choice": int(choice),
            "choice_probability": float(probs[choice]),
            "probabilities": {int(k): float(probs[k]) for k in offered_ids},
            "threat_score": None,
            "threat_level": None,
            "safety_p_true": None,
        }
        if "threat_level" in answers:
            t_ans = answers["threat_level"]
            if t_ans.get("type") == "score":
                parsed["threat_score"] = float(t_ans["score"])
                parsed["threat_level"] = int(t_ans["level"])
        if "safety_margin" in answers:
            s_ans = answers["safety_margin"]
            if s_ans.get("type") == "boolean":
                parsed["safety_p_true"] = float(s_ans["p_true"])
        return parsed
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        return None


class HarnessNanoJevAgent:
    """UAV-JevHarness policy combining 14-value kinematic projection with NanoJev inference."""

    def __init__(
        self,
        evaluate: Callable[[dict], dict],
        *,
        config: EnvConfig | None = None,
        top_k: int = 3,
        shielded: bool = True,
        include_diagnostics: bool = True,
        utility_temperature: float = 0.08,
    ):
        if not 2 <= top_k <= 9:
            raise ValueError("top_k must be between 2 and 9")
        self.evaluate = evaluate
        self.harness = UAVJevHarness(config)
        self.top_k = top_k
        self.shielded = shielded
        self.include_diagnostics = include_diagnostics
        self.utility_temperature = utility_temperature
        self.latencies_ms: list[float] = []
        self.invalid_actions = 0
        self.last_probability: float | None = None
        self.last_diagnostics: dict | None = None
        self.step_diagnostics: list[dict] = []

    def reset(self) -> None:
        self.harness.reset_memory()
        self.step_diagnostics.clear()

    def select_action(self, state) -> int:
        start = perf_counter()
        analysis = self.harness.analyze(state)
        request = self.harness.build_request(
            analysis, top_k=self.top_k, include_diagnostics=self.include_diagnostics
        )
        offered = analysis["ranked_candidates"][:self.top_k]
        offered_ids = [str(c["action"]) for c in offered]
        response = self.evaluate(request)
        self.latencies_ms.append((perf_counter() - start) * 1000.0)

        parsed = parse_harness_response(response, offered_ids)
        if parsed is None:
            self.invalid_actions += 1
            self.last_probability = None
            fallback = offered[0]["action"] if self.shielded else Action.MAINTAIN.value
            self.harness.prev_action = fallback
            return fallback

        if not self.shielded:
            chosen = parsed["choice"]
            self.last_probability = parsed["choice_probability"]
        else:
            # Combine Jev log-probabilities with Harness kinematic safety shield
            best_u = offered[0]["utility"]
            best_action = offered[0]["action"]
            best_joint = -1e18
            for cand in offered:
                act = cand["action"]
                u_diff = cand["utility"] - best_u
                # Only allow candidates within the near-equivalent Pareto tolerance of the top safe maneuver
                if u_diff < -0.05:
                    continue
                p_jev = max(parsed["probabilities"].get(act, 1e-6), 1e-6)
                joint_score = log(p_jev) + (u_diff / max(self.utility_temperature, 1e-3))
                if joint_score > best_joint:
                    best_joint = joint_score
                    best_action = act
            chosen = best_action
            self.last_probability = parsed["probabilities"].get(chosen, parsed["choice_probability"])

        self.harness.prev_action = chosen
        self.last_diagnostics = {
            "step": analysis["step"],
            "chosen_action": chosen,
            "jev_raw_choice": parsed["choice"],
            "jev_probability": self.last_probability,
            "jev_threat_score": parsed["threat_score"],
            "jev_threat_level": parsed["threat_level"],
            "jev_safety_p_true": parsed["safety_p_true"],
            "min_clr_m": next(c["min_clr_m"] for c in offered if c["action"] == chosen),
            "active_sectors": analysis["active_sectors"],
            "offered_actions": [c["action"] for c in offered],
        }
        self.step_diagnostics.append(self.last_diagnostics)
        return chosen


class HarnessOnlyAgent:
    """Attribution control that always executes the kinematic utility leader."""

    def __init__(self, config: EnvConfig | None = None):
        self.harness = UAVJevHarness(config)
        self.invalid_actions = 0
        self.step_diagnostics: list[dict] = []

    def reset(self) -> None:
        self.harness.reset_memory()
        self.step_diagnostics.clear()

    def select_action(self, state) -> int:
        analysis = self.harness.analyze(state)
        candidate = analysis["ranked_candidates"][0]
        chosen = candidate["action"]
        self.harness.prev_action = chosen
        self.step_diagnostics.append({
            "step": analysis["step"],
            "chosen_action": chosen,
            "decision_source": "utility_top",
            "safe_candidate_count": sum(c["safe"] for c in analysis["ranked_candidates"]),
            "offered_actions": [],
            "jev_raw_choice": None,
            "min_clr_m": candidate["min_clr_m"],
        })
        return chosen


class SafeChoiceNanoJevAgent:
    """Let NanoJev choose among safety-passing actions, optionally capped by utility."""

    def __init__(
        self,
        evaluate: Callable[[dict], dict],
        *,
        config: EnvConfig | None = None,
        max_candidates: int | None = None,
    ):
        if max_candidates is not None and not 2 <= max_candidates <= 9:
            raise ValueError("max_candidates must be between 2 and 9")
        self.evaluate = evaluate
        self.harness = UAVJevHarness(config)
        self.max_candidates = max_candidates
        self.invalid_actions = 0
        self.step_diagnostics: list[dict] = []

    def reset(self) -> None:
        self.harness.reset_memory()
        self.step_diagnostics.clear()

    def select_action(self, state) -> int:
        analysis = self.harness.analyze(state)
        ranked = analysis["ranked_candidates"]
        safe_ranked = [c for c in ranked if c["safe"]]
        offered_ranked = safe_ranked[:self.max_candidates]
        offered = sorted(offered_ranked, key=lambda c: c["action"])
        parsed = None
        if not safe_ranked:
            chosen = ranked[0]["action"]
            source = "no_safe_fallback"
        elif len(safe_ranked) == 1:
            chosen = safe_ranked[0]["action"]
            source = "sole_safe"
        else:
            request = self.harness.build_request(
                analysis,
                offered_candidates=offered,
                include_top_candidates=False,
                include_diagnostics=True,
            )
            parsed = parse_harness_response(
                self.evaluate(request), [str(c["action"]) for c in offered]
            )
            if parsed is None:
                self.invalid_actions += 1
                chosen = safe_ranked[0]["action"]
                source = "invalid_fallback"
            else:
                chosen = parsed["choice"]
                source = "nanojev"

        self.harness.prev_action = chosen
        selected = next(c for c in ranked if c["action"] == chosen)
        self.step_diagnostics.append({
            "step": analysis["step"],
            "chosen_action": chosen,
            "decision_source": source,
            "safe_candidate_count": len(safe_ranked),
            "offered_actions": [c["action"] for c in offered] if len(safe_ranked) >= 2 else [],
            "jev_raw_choice": parsed["choice"] if parsed else None,
            "jev_probability": parsed["choice_probability"] if parsed else None,
            "jev_threat_score": parsed["threat_score"] if parsed else None,
            "jev_safety_p_true": parsed["safety_p_true"] if parsed else None,
            "min_clr_m": selected["min_clr_m"],
        })
        return chosen
