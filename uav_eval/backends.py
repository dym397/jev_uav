"""Normalize native NanoJev and Jev-compatible Choice responses."""

import json
import math
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config import ACTION_SPACE


def _checked_sum(values: dict, sum_tol: float, what: str) -> dict:
    # sum_tol > 1e-3 admits servers that round each probability (the hosted Jev API sends two decimals).
    total = math.fsum(values.values())
    if not math.isclose(total, 1.0, rel_tol=0, abs_tol=sum_tol):
        raise ValueError(f"{what} must sum to one")
    return {key: value / total for key, value in values.items()}


def normalize_choice(response: dict, sum_tol: float = 1e-3) -> dict[int, float]:
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
        return _checked_sum(values, sum_tol, "Choice probabilities")
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError("Malformed Choice response") from error


def validate_distribution(raw, keys, sum_tol: float = 1e-3) -> dict[str, float]:
    """A finite, normalized distribution over exactly the offered option keys."""
    if not isinstance(raw, dict) or set(raw) != set(keys):
        raise ValueError(f"Expected probabilities over {sorted(keys)}, got {raw!r}")
    values = {}
    for key in keys:
        number = raw[key]
        if type(number) not in (float, int) or not math.isfinite(number) or number < 0:
            raise ValueError("Probabilities must be finite and nonnegative")
        values[key] = float(number)
    return _checked_sum(values, sum_tol, "Probabilities")


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
    def __init__(self, url: str, *, model_id: str | None = None, request_model: str | None = None,
                 api_key_env: str | None = None, sum_tol: float = 1e-3, retries: int = 4,
                 timeout: float = 60.0):
        self.url = url
        self.model_id = model_id or url
        # Sent as the request's "model"; some servers accept only their own names (Open-Jev), so the
        # label in our tables (model_id) can differ from it. "" sends no model (the server's default).
        self.request_model = request_model if request_model is not None else model_id
        # The key is read from the environment only, so it never lands in argv, logs or records.
        self.headers = {"Content-Type": "application/json"}
        if api_key_env:
            key = os.environ.get(api_key_env)
            if not key:
                raise ValueError(f"{api_key_env} is not set")
            self.headers["Authorization"] = f"Bearer {key}"
        self.sum_tol = sum_tol
        self.retries = retries
        self.timeout = timeout

    def _post(self, body: dict) -> dict:
        if self.request_model:
            body = {**body, "model": self.request_model}
        data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        for attempt in range(self.retries + 1):
            try:
                with urlopen(Request(self.url, data=data, headers=self.headers), timeout=self.timeout) as response:
                    return json.load(response)
            except (HTTPError, URLError, TimeoutError) as failure:
                # Rate limits, server errors and dropped connections are retried; a 4xx is the request's fault.
                transient = not isinstance(failure, HTTPError) or failure.code == 429 or failure.code >= 500
                if not transient or attempt == self.retries:
                    raise
                time.sleep(2 ** attempt)

    def ask(self, state: str, question: dict) -> dict[str, float]:
        decoded = self._post({"state": state, "questions": {"q": question}})
        return validate_distribution(decoded["answers"]["q"]["probabilities"], question["criteria"], self.sum_tol)

    def predict(self, payload: dict) -> dict[int, float]:
        return normalize_choice(self._post(payload), self.sum_tol)
