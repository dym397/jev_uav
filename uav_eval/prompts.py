"""Versioned UAV prompt inputs for decision models.

Kinematic styles use current idealized sensor values from ``Observation``.
They never query the simulator's future state or episode outcome.
"""

from dataclasses import dataclass
from math import atan2, cos, degrees, hypot, sin

from config import ACTION_SPACE, action_control
from environment.observation import wrap_angle


# first_person_polar / third_person_polar state identical facts (range, bearing,
# relative velocity); only the grammatical perspective differs. The *_semantic
# styles carry derived facts (range rate, bearing rate, arrival vs deadline) as
# graded words with no numbers.
#
# Equal-information pairs, so that one factor changes at a time:
#   paper14 -> paper14_prose          the D3QN's 14 values: fields -> sentences (same numbers)
#   paper14_prose -> paper14_semantic same 14 values: numbers -> graded words
#   first_person_derived -> first_person_semantic   same derived facts: numbers -> words
#   first_person_polar -> first_person_derived      raw relative velocity -> derived rates
#   first_person_polar -> first_person_clock        bearing frame: degrees -> clock positions
#   first_person_polar -> first_person_world        body frame -> world coordinates
#   first_person_polar -> first_person_list         format: prose -> bullet list
STATE_STYLES = ("paper14", "kinematic_fields", "kinematic_prose", "cpa_brief",
                "first_person_polar", "third_person_polar",
                "first_person_semantic", "third_person_semantic",
                "paper14_prose", "paper14_semantic", "first_person_derived",
                "first_person_clock", "first_person_world", "first_person_list")
FIRST_PERSON_STYLES = ("first_person_polar", "first_person_semantic", "first_person_derived",
                       "first_person_clock", "first_person_world", "first_person_list")
SEMANTIC_STYLES = ("first_person_semantic", "third_person_semantic")
PAPER14_STYLES = ("paper14", "paper14_prose", "paper14_semantic")
QUESTION_STYLES = ("balanced", "safety_first")
CPA_HORIZON_S = 10.0


@dataclass(frozen=True)
class PromptSpec:
    state_style: str
    question_style: str


def _number(value: float) -> str:
    return f"{float(value):.3f}".rstrip("0").rstrip(".") if float(value) else "0"


def _kinematic_fields(observation) -> str:
    rows = [
        f"time_s={_number(observation.current_time)}",
        f"due_s={_number(observation.due_time)}",
        f"own_x_m={_number(observation.own_position[0])}",
        f"own_y_m={_number(observation.own_position[1])}",
        f"heading_deg={_number(degrees(observation.own_heading))}",
        f"speed_mps={_number(observation.own_speed)}",
        f"goal_x_m={_number(observation.goal[0])}",
        f"goal_y_m={_number(observation.goal[1])}",
        f"contacts_within_100m={len(observation.contacts)}",
    ]
    for index, contact in enumerate(observation.contacts):
        rows.extend((
            f"contact_{index}_forward_m={_number(contact.forward_m)}",
            f"contact_{index}_left_m={_number(contact.left_m)}",
            f"contact_{index}_rel_forward_mps={_number(contact.relative_forward_mps)}",
            f"contact_{index}_rel_left_mps={_number(contact.relative_left_mps)}",
        ))
    return "; ".join(rows)


def _kinematic_prose(observation) -> str:
    lines = [
        f"At time {_number(observation.current_time)} s, the deadline is "
        f"{_number(observation.due_time)} s. Own UAV position is "
        f"({_number(observation.own_position[0])}, {_number(observation.own_position[1])}) m, "
        f"heading {_number(degrees(observation.own_heading))} deg and speed "
        f"{_number(observation.own_speed)} m/s. The goal is "
        f"({_number(observation.goal[0])}, {_number(observation.goal[1])}) m.",
        f"Detected contacts within 100 m: {len(observation.contacts)}.",
    ]
    for index, contact in enumerate(observation.contacts):
        lines.append(
            f"Contact {index}: relative position (forward {_number(contact.forward_m)}, "
            f"left {_number(contact.left_m)}) m; relative velocity "
            f"(forward {_number(contact.relative_forward_mps)}, "
            f"left {_number(contact.relative_left_mps)}) m/s."
        )
    return "\n".join(lines)


def _side(angle_deg: float) -> str:
    """Body-frame bearing in words; positive angles are to the left."""
    rounded = round(angle_deg)
    if rounded == 0:
        return "dead ahead"
    if abs(rounded) == 180:
        return "directly behind"
    return f"{abs(rounded)} deg to the {'left' if rounded > 0 else 'right'} of the nose"


def _motion(forward_mps: float, left_mps: float) -> str:
    along = "forward" if forward_mps >= 0 else "backward"
    across = "left" if left_mps >= 0 else "right"
    return (f"{_number(abs(forward_mps))} m/s {along} and "
            f"{_number(abs(left_mps))} m/s to the {across}")


def _polar(observation, *, first_person: bool) -> str:
    """Same facts as kinematic_prose in range/bearing form; no derived risk estimates."""
    my = "my" if first_person else "its"
    own_x, own_y = observation.own_position
    goal_dx, goal_dy = observation.goal[0] - own_x, observation.goal[1] - own_y
    goal_bearing = degrees(wrap_angle(atan2(goal_dy, goal_dx) - observation.own_heading))
    remaining = observation.due_time - observation.current_time
    deadline = (f"{_number(remaining)} s remain until {my} arrival deadline" if remaining >= 0
                else f"{my} arrival deadline passed {_number(-remaining)} s ago")
    subject = "I am" if first_person else "The UAV is"
    lines = [
        f"{subject} flying at {_number(observation.own_speed)} m/s. "
        f"{my.capitalize()} goal is {_number(hypot(goal_dx, goal_dy))} m away, "
        f"{_side(goal_bearing)}. {deadline[0].upper() + deadline[1:]}.",
    ]
    if not observation.contacts:
        lines.append(f"No intruder is detected within 100 m of {'me' if first_person else 'the UAV'}.")
    for index, contact in enumerate(observation.contacts):
        bearing = degrees(atan2(contact.left_m, contact.forward_m))
        whose = "me" if first_person else "the UAV"
        lines.append(
            f"Intruder {index + 1} is {_number(hypot(contact.forward_m, contact.left_m))} m from "
            f"{whose}, {_side(bearing)}. Relative to {whose} it moves "
            f"{_motion(contact.relative_forward_mps, contact.relative_left_mps)}."
        )
    return "\n".join(lines)


def _range_words(meters: float) -> str:
    for limit, words in ((15, "extremely close"), (30, "very close"), (60, "close")):
        if meters < limit:
            return words
    return "at a moderate distance"


def _direction_words(angle_deg: float) -> str:
    """Body-frame bearing as a coarse sector; positive angles are to the left."""
    side = "left" if angle_deg > 0 else "right"
    size = abs(angle_deg)
    if size < 5:
        return "straight ahead"
    if size < 30:
        return f"ahead and slightly to the {side}"
    if size < 70:
        return f"ahead to the {side}"
    if size < 110:
        return f"off to the {side}"
    if size < 175:
        return f"behind to the {side}"
    return "directly behind"


def _closing_words(range_rate: float) -> str:
    """range_rate < 0 means the distance is shrinking."""
    if range_rate < -6:
        return "closing in fast"
    if range_rate < -2:
        return "closing in"
    if range_rate < -0.2:
        return "closing in slowly"
    if range_rate <= 0.2:
        return "holding roughly the same distance"
    return "moving away"


def _drift_words(bearing_rate_deg: float) -> str:
    """How fast the intruder's direction is changing right now (line-of-sight rate)."""
    if abs(bearing_rate_deg) < 1:
        return "its direction is staying almost constant"
    side = "left" if bearing_rate_deg > 0 else "right"
    pace = "slowly" if abs(bearing_rate_deg) < 4 else "quickly"
    return f"its direction is {pace} drifting to the {side}"


def _semantic(observation, *, first_person: bool) -> str:
    """Current facts as graded words only; no numbers and no predicted separation."""
    me = "me" if first_person else "the UAV"
    my = "my" if first_person else "its"
    own_x, own_y = observation.own_position
    goal_dx, goal_dy = observation.goal[0] - own_x, observation.goal[1] - own_y
    goal_bearing = degrees(wrap_angle(atan2(goal_dy, goal_dx) - observation.own_heading))
    speed = float(observation.own_speed)
    pace = "slowly" if speed < 3 else "at a moderate speed" if speed < 7 else "fast"
    remaining = observation.due_time - observation.current_time
    if remaining < 0:
        timing = f"{my} arrival deadline has already passed"
    else:
        eta = hypot(goal_dx, goal_dy) / max(speed, 1e-6)
        timing = (f"at this speed {'I' if first_person else 'it'} would arrive "
                  + ("well before the deadline" if eta < 0.85 * remaining
                     else "after the deadline" if eta > 1.15 * remaining else "about on time"))
    subject = "I am" if first_person else "The UAV is"
    lines = [f"{subject} flying {pace}. {my.capitalize()} goal is "
             f"{_direction_words(goal_bearing)}, and {timing}."]
    if not observation.contacts:
        lines.append(f"No intruder is detected near {me}.")
    for index, contact in enumerate(observation.contacts):
        f, l = float(contact.forward_m), float(contact.left_m)
        vf, vl = float(contact.relative_forward_mps), float(contact.relative_left_mps)
        r = max(hypot(f, l), 1e-6)
        range_rate = (f * vf + l * vl) / r
        bearing_rate = degrees((f * vl - l * vf) / (r * r))
        lines.append(
            f"Intruder {chr(ord('A') + index)} is {_range_words(r)}, "
            f"{_direction_words(degrees(atan2(l, f)))}. It is {_closing_words(range_rate)}, "
            f"and seen from {me} {_drift_words(bearing_rate)}."
        )
    return "\n".join(lines)


# The nine paper sectors are 40 deg wide; sector i is centred i * 40 deg left of the nose.
SECTOR_WORDS = ("straight ahead", "ahead to the left", "off to the left", "behind to the left",
                "almost directly behind, slightly left", "almost directly behind, slightly right",
                "behind to the right", "off to the right", "ahead to the right")
COMPASS = ("east", "north-east", "north", "north-west", "west", "south-west", "south", "south-east")


def _paper14_parts(observation):
    values = [float(v) for v in observation.numeric]
    if len(values) != 14:
        raise ValueError("Paper observation requires exactly 14 values")
    heading, speed, goal_ratio, goal_angle, eta_ratio = values[:5]
    return heading, speed, goal_ratio, degrees(wrap_angle(goal_angle)), eta_ratio, values[5:]


def _paper14_prose(observation) -> str:
    """The D3QN's 14 values as sentences: same numbers (angles in degrees), no added facts."""
    heading, speed, goal_ratio, goal_deg, eta_ratio, sectors = _paper14_parts(observation)
    lines = [
        f"The UAV's heading is {_number(degrees(heading))} deg counter-clockwise from the +x axis "
        f"and its speed is {_number(speed)} m/s. Its goal is {_number(goal_ratio)} of the leg length "
        f"away, {_side(goal_deg)}. Its estimated remaining flight time is {_number(eta_ratio)} "
        f"times the leg's deadline (deadline counted from the start of the leg).",
        "Nearest intruder in each 40 deg sector, as a fraction of the 100 m detection radius "
        "(1 means nothing detected):",
    ]
    lines.extend(f"Sector centred {_side(wrap_angle_deg(40 * i))}: {_number(v)}."
                 for i, v in enumerate(sectors))
    return "\n".join(lines)


def wrap_angle_deg(angle_deg: float) -> float:
    return (angle_deg + 180) % 360 - 180


def _share_words(ratio: float) -> str:
    for limit, words in ((0.1, "almost none"), (0.4, "less than half"), (0.6, "about half"),
                         (0.95, "most"), (1.05, "about all")):
        if ratio < limit:
            return words
    return "more than the whole"


def _paper14_semantic(observation) -> str:
    """The same 14 values as graded words; nothing derived beyond what the values state."""
    heading, speed, goal_ratio, goal_deg, eta_ratio, sectors = _paper14_parts(observation)
    compass = COMPASS[round(degrees(heading) % 360 / 45) % 8]
    pace = "slowly" if speed < 3 else "at a moderate speed" if speed < 7 else "fast"
    lines = [f"The UAV is heading roughly {compass} and flying {pace}. {_share_words(goal_ratio).capitalize()} "
             f"of the leg to its goal is still ahead of it, and the goal is {_direction_words(goal_deg)}. "
             f"Its estimated remaining flight time is {_share_words(eta_ratio)} of the leg's deadline "
             f"(deadline counted from the start of the leg)."]
    occupied = [f"{_range_words(100 * v)} {SECTOR_WORDS[i]}" for i, v in enumerate(sectors) if v < 1]
    lines.append("Nearest intruder per direction: " + "; ".join(occupied) + "; every other direction is clear."
                 if occupied else "No intruder is detected in any direction.")
    return "\n".join(lines)


def _contact_rates(contact) -> tuple[float, float, float, float]:
    """(range m, bearing deg left of nose, range rate m/s, bearing rate deg/s left)."""
    f, l = float(contact.forward_m), float(contact.left_m)
    vf, vl = float(contact.relative_forward_mps), float(contact.relative_left_mps)
    r = max(hypot(f, l), 1e-6)
    return r, degrees(atan2(l, f)), (f * vf + l * vl) / r, degrees((f * vl - l * vf) / (r * r))


def _goal_facts(observation):
    """(goal distance m, goal bearing deg left of nose, seconds left until the deadline)."""
    own_x, own_y = observation.own_position
    goal_dx, goal_dy = observation.goal[0] - own_x, observation.goal[1] - own_y
    return (hypot(goal_dx, goal_dy), degrees(wrap_angle(atan2(goal_dy, goal_dx) - observation.own_heading)),
            observation.due_time - observation.current_time)


def _deadline(remaining: float) -> str:
    return (f"{_number(remaining)} s remain until my arrival deadline" if remaining >= 0
            else f"my arrival deadline passed {_number(-remaining)} s ago")


def _first_person_derived(observation) -> str:
    """The facts behind first_person_semantic, as numbers instead of words."""
    distance, goal_deg, remaining = _goal_facts(observation)
    speed = float(observation.own_speed)
    lines = [f"I am flying at {_number(speed)} m/s. My goal is {_side(goal_deg)}; at this speed I would "
             f"arrive in {_number(distance / max(speed, 1e-6))} s, and {_deadline(remaining)}."]
    if not observation.contacts:
        lines.append("No intruder is detected near me.")
    for index, contact in enumerate(observation.contacts):
        r, bearing, range_rate, bearing_rate = _contact_rates(contact)
        trend = (f"shrinking by {_number(-range_rate)} m/s" if range_rate < 0
                 else f"growing by {_number(range_rate)} m/s")
        drift = "left" if bearing_rate >= 0 else "right"
        lines.append(f"Intruder {chr(ord('A') + index)} is {_number(r)} m from me, {_side(bearing)}. "
                     f"The distance is {trend}, and seen from me its direction drifts "
                     f"{_number(abs(bearing_rate))} deg/s to the {drift}.")
    return "\n".join(lines)


def _clock(angle_deg: float) -> int:
    """Body-frame direction (positive = left) as a clock position, 12 = straight ahead."""
    return round(-angle_deg / 30) % 12 or 12


def _first_person_clock(observation) -> str:
    """first_person_polar's facts with directions as clock positions."""
    distance, goal_deg, remaining = _goal_facts(observation)
    lines = [f"I am flying at {_number(observation.own_speed)} m/s. My goal is {_number(distance)} m "
             f"away at my {_clock(goal_deg)} o'clock. {_deadline(remaining)[0].upper() + _deadline(remaining)[1:]}."]
    if not observation.contacts:
        lines.append("No intruder is detected within 100 m of me.")
    for index, contact in enumerate(observation.contacts):
        f, l = float(contact.forward_m), float(contact.left_m)
        vf, vl = float(contact.relative_forward_mps), float(contact.relative_left_mps)
        motion = (f"moves at {_number(hypot(vf, vl))} m/s toward my {_clock(degrees(atan2(vl, vf)))} o'clock"
                  if hypot(vf, vl) else "does not move")
        lines.append(f"Intruder {index + 1} is {_number(hypot(f, l))} m from me at my "
                     f"{_clock(degrees(atan2(l, f)))} o'clock. Relative to me it {motion}.")
    return "\n".join(lines)


def _first_person_world(observation) -> str:
    """first_person_polar's facts in world coordinates instead of body axes."""
    own_x, own_y = observation.own_position
    heading, speed = float(observation.own_heading), float(observation.own_speed)
    fx, fy = cos(heading), sin(heading)
    remaining = observation.due_time - observation.current_time
    lines = [f"I am at ({_number(own_x)}, {_number(own_y)}) m, flying at {_number(speed)} m/s with heading "
             f"{_number(degrees(heading) % 360)} deg counter-clockwise from the +x axis. My goal is at "
             f"({_number(observation.goal[0])}, {_number(observation.goal[1])}) m. "
             f"{_deadline(remaining)[0].upper() + _deadline(remaining)[1:]}."]
    if not observation.contacts:
        lines.append("No intruder is detected within 100 m of me.")
    for index, contact in enumerate(observation.contacts):
        f, l = float(contact.forward_m), float(contact.left_m)
        vf, vl = float(contact.relative_forward_mps), float(contact.relative_left_mps)
        x, y = own_x + f * fx - l * fy, own_y + f * fy + l * fx
        vx, vy = vf * fx - vl * fy + speed * fx, vf * fy + vl * fx + speed * fy
        lines.append(f"Intruder {index + 1} is at ({_number(x)}, {_number(y)}) m with velocity "
                     f"({_number(vx)}, {_number(vy)}) m/s.")
    return "\n".join(lines)


def _first_person_list(observation) -> str:
    """first_person_polar's facts as a bullet list."""
    distance, goal_deg, remaining = _goal_facts(observation)
    lines = [f"- my speed: {_number(observation.own_speed)} m/s",
             f"- my goal: {_number(distance)} m away, {_side(goal_deg)}",
             f"- my deadline: {_deadline(remaining).replace('my arrival deadline', 'deadline')}"]
    if not observation.contacts:
        lines.append("- intruders: none detected within 100 m of me")
    for index, contact in enumerate(observation.contacts):
        f, l = float(contact.forward_m), float(contact.left_m)
        lines.append(f"- intruder {index + 1}: {_number(hypot(f, l))} m from me, {_side(degrees(atan2(l, f)))}; "
                     f"relative to me it moves "
                     f"{_motion(contact.relative_forward_mps, contact.relative_left_mps)}")
    return "\n".join(lines)


def _cpa(contact) -> tuple[float, float]:
    px, py = float(contact.forward_m), float(contact.left_m)
    vx, vy = float(contact.relative_forward_mps), float(contact.relative_left_mps)
    speed_sq = vx * vx + vy * vy
    time = min(max(-(px * vx + py * vy) / speed_sq, 0.0), CPA_HORIZON_S) if speed_sq else 0.0
    return time, hypot(px + vx * time, py + vy * time)


RENDERERS = {"paper14_prose": _paper14_prose, "paper14_semantic": _paper14_semantic,
             "first_person_derived": _first_person_derived, "first_person_clock": _first_person_clock,
             "first_person_world": _first_person_world, "first_person_list": _first_person_list}


def action_criteria() -> dict[str, str]:
    """Describe the actual controls, not merely the D3QN action names."""
    criteria = {}
    for action_id, name in ACTION_SPACE.items():
        yaw, acceleration = action_control(action_id)
        yaw_deg = round(degrees(yaw))
        criteria[str(action_id)] = (
            f"{name}: yaw {yaw_deg:+d} deg/s; "
            f"acceleration {acceleration:+g} m/s^2"
        )
    return criteria


def build_prompt(observation, spec: PromptSpec) -> dict:
    if spec.state_style not in STATE_STYLES:
        raise ValueError(f"Unknown state style: {spec.state_style}")
    if spec.question_style not in QUESTION_STYLES:
        raise ValueError(f"Unknown question style: {spec.question_style}")
    if spec.state_style == "paper14":
        state = observation.state_to_text()
    elif spec.state_style == "kinematic_prose":
        state = _kinematic_prose(observation)
    elif spec.state_style in ("first_person_polar", "third_person_polar"):
        state = _polar(observation, first_person=spec.state_style in FIRST_PERSON_STYLES)
    elif spec.state_style in SEMANTIC_STYLES:
        state = _semantic(observation, first_person=spec.state_style in FIRST_PERSON_STYLES)
    elif spec.state_style in RENDERERS:
        state = RENDERERS[spec.state_style](observation)
    else:
        state = _kinematic_fields(observation)
        if spec.state_style == "cpa_brief":
            estimates = [f"cpa_horizon_s={_number(CPA_HORIZON_S)}"]
            for index, contact in enumerate(observation.contacts):
                time, distance = _cpa(contact)
                estimates.extend((
                    f"contact_{index}_tcpa_s={_number(time)}",
                    f"contact_{index}_dcpa_m={_number(distance)}",
                ))
            state += "; " + "; ".join(estimates)
    priority = (
        "Safety first: minimize collision risk; among comparably safe controls, "
        "reach the goal near its deadline."
        if spec.question_style == "safety_first" else
        "Choose a control that keeps at least 10 m from intruders while "
        "reaching the goal near its deadline."
    )
    instructions = (
        f"{priority} Select one control for the next 1 s. During this step the UAV "
        "moves with its current heading and speed; the selected yaw and acceleration "
        "change heading and speed at the end of the step. Negative yaw turns right, "
        "positive yaw turns left. Use only the stated current observations."
    )
    if spec.state_style in FIRST_PERSON_STYLES:
        # Same content as above, addressed to the aircraft itself.
        priority = (
            "You are this UAV. Safety first: minimize my collision risk; among comparably "
            "safe controls, reach my goal near its deadline."
            if spec.question_style == "safety_first" else
            "You are this UAV. Choose my control so that I keep at least 10 m from intruders "
            "while reaching my goal near its deadline."
        )
        instructions = (
            f"{priority} Select my control for the next 1 s. During this step I keep my current "
            "heading and speed; the selected yaw and acceleration change them at the end of the "
            "step. Negative yaw turns me right, positive yaw turns me left. Use only the stated "
            "current observations."
        )
    return {
        "state": state,
        "questions": {"action": {
            "type": "choice",
            "instructions": instructions,
            "criteria": action_criteria(),
        }},
    }
