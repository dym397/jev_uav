"""One toy Choice question per model; prints the normalized distribution."""
import json
import os
import sys
from time import perf_counter

MODELS = "/home/mydisk1/jev_uav/models"
STATE = "An intruder aircraft is 30 m directly ahead of me and closing at 8 m/s. Nothing is on my left or right."
QUESTION = {"type": "choice", "instructions": "Which control should I choose to avoid a collision?",
            "criteria": {"0": "Turn right", "1": "Maintain heading and speed", "2": "Turn left"}}

which = sys.argv[1]
t = perf_counter()
if which == "jevk5":
    from jevk5 import JevK5
    model = JevK5(f"{MODELS}/JevK5-9B", device="cuda:0")
    load = perf_counter() - t
    out = model.decide(STATE, QUESTION)
elif which.startswith("laya"):
    import laya
    sub = {"laya": None, "laya_td": "typed-decisions"}[which]
    path = f"{MODELS}/laya" if sub is None else f"{MODELS}/laya-typed-decisions"
    agent = laya.load(path)
    load = perf_counter() - t
    out = agent.predict(STATE, {"action": QUESTION})
print(json.dumps({"model": which, "load_s": round(load, 1), "answer": out}, default=str, indent=1)[:2500])
