"""Normalize native NanoJev and Jev-compatible Choice responses."""

import json
import math
from urllib.request import Request, urlopen

from config import ACTION_SPACE


def normalize_choice(response: dict) -> dict[int, float]:
    try:
        if "states" in response:
            rows = response["states"]
            if len(rows) != 1 or rows[0]["id"] != "uav":
                raise ValueError("Expected one uav state")
            answer = rows[0]["answers"]["action"]
        else:
            answer = response["answers"]["action"]
        if answer.get("type", "choice") != "choice":
            raise ValueError("Expected a Choice answer")
        raw = answer["probabilities"]
        if set(raw) != {str(i) for i in ACTION_SPACE}:
            raise ValueError("Choice probabilities must cover all nine actions")
        values = {}
        for action_id in ACTION_SPACE:
            number = raw[str(action_id)]
            if type(number) not in (float, int) or not math.isfinite(number) or number < 0:
                raise ValueError("Choice probabilities must be finite and nonnegative")
            values[action_id] = float(number)
        if not math.isclose(math.fsum(values.values()), 1.0, rel_tol=0, abs_tol=1e-3):
            raise ValueError("Choice probabilities must sum to one")
        return values
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError("Malformed Choice response") from error


def validate_distribution(raw, keys) -> dict[str, float]:
    """A finite, normalized distribution over exactly the offered option keys."""
    if not isinstance(raw, dict) or set(raw) != set(keys):
        raise ValueError(f"Expected probabilities over {sorted(keys)}, got {raw!r}")
    values = {}
    for key in keys:
        number = raw[key]
        if type(number) not in (float, int) or not math.isfinite(number) or number < 0:
            raise ValueError("Probabilities must be finite and nonnegative")
        values[key] = float(number)
    if not math.isclose(math.fsum(values.values()), 1.0, rel_tol=0, abs_tol=1e-3):
        raise ValueError("Probabilities must sum to one")
    return values


def _laya_checked(response: dict) -> dict:
    usage = response.get("usage") or {}
    # laya drops tokens past its context budget; count that as invalid, not silent loss.
    if usage.get("truncated") or usage.get("state_tokens_dropped"):
        raise ValueError(f"laya truncated the input: {usage}")
    return response


# Every backend also answers one arbitrary Choice question with ask(state, question),
# returning probabilities keyed by the question's criteria keys. Questions are always
# asked one per call so that no model sees another question's text.


class NanoBackend:
    def __init__(self, evaluate, *, model_id: str):
        self.evaluate = evaluate
        self.model_id = model_id

    def predict(self, payload: dict) -> dict[int, float]:
        response = self.evaluate({"states": [{"id": "uav", **payload}]})
        return normalize_choice(response)

    def ask(self, state: str, question: dict) -> dict[str, float]:
        response = self.evaluate({"states": [{"id": "uav", "state": state, "questions": {"q": question}}]})
        answer = response["states"][0]["answers"]["q"]
        return validate_distribution(answer["probabilities"], question["criteria"])


class JevK5Backend:
    """In-process JevK5 (allebee/jevk5): softmax over option-letter logits, one pass."""

    def __init__(self, model, *, model_id: str):
        self.model = model
        self.model_id = model_id

    def predict(self, payload: dict) -> dict[int, float]:
        answer = self.model.decide(payload["state"], payload["questions"]["action"])
        return normalize_choice({"answers": {"action": answer}})

    def ask(self, state: str, question: dict) -> dict[str, float]:
        return validate_distribution(self.model.decide(state, question)["probabilities"], question["criteria"])


def _renormalized(raw: dict) -> dict:
    # decider rounds each probability to 4 decimals; renormalize so the sum check is exact.
    total = math.fsum(raw.values())
    return {k: v / total for k, v in raw.items()} if total > 0 else raw


class DeciderBackend:
    """In-process decider-ai (Mapika/decider): option-letter logits at one answer slot.

    Also serves as the zero-shot LLM-logit control: a stock causal LM loaded through the
    same Decider gets the same prompt layout and readout, at temperature 1 (no config).
    """

    def __init__(self, decider, *, model_id: str):
        self.decider = decider
        self.model_id = model_id

    def ask(self, state: str, question: dict) -> dict[str, float]:
        answer = self.decider.system_one(state, {"q": question})["answers"]["q"]
        return validate_distribution(_renormalized(answer["probabilities"]), question["criteria"])

    def predict(self, payload: dict) -> dict[int, float]:
        answer = self.decider.system_one(payload["state"], payload["questions"])["answers"]["action"]
        answer = {**answer, "probabilities": _renormalized(answer["probabilities"])}
        return normalize_choice({"answers": {"action": answer}})


class LayaBackend:
    """In-process laya agent; predict() already returns a SystemOne-shaped response."""

    def __init__(self, agent, *, model_id: str):
        self.agent = agent
        self.model_id = model_id

    def predict(self, payload: dict) -> dict[int, float]:
        return normalize_choice(_laya_checked(self.agent.predict(payload["state"], payload["questions"])))

    def ask(self, state: str, question: dict) -> dict[str, float]:
        response = _laya_checked(self.agent.predict(state, {"q": question}))
        return validate_distribution(response["answers"]["q"]["probabilities"], question["criteria"])


class SystemOneHTTPBackend:
    def __init__(self, url: str, *, model_id: str | None = None, timeout: float = 60.0):
        self.url = url
        self.model_id = model_id or url
        self.request_model = model_id
        self.timeout = timeout

    def ask(self, state: str, question: dict) -> dict[str, float]:
        body = {"state": state, "questions": {"q": question}}
        if self.request_model is not None:
            body["model"] = self.request_model
        data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        request = Request(self.url, data=data, headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=self.timeout) as response:
            decoded = json.load(response)
        return validate_distribution(decoded["answers"]["q"]["probabilities"], question["criteria"])

    def predict(self, payload: dict) -> dict[int, float]:
        body = {**payload}
        if self.request_model is not None:
            body["model"] = self.request_model
        data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        request = Request(self.url, data=data, headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=self.timeout) as response:
            decoded = json.load(response)
        return normalize_choice(decoded)
