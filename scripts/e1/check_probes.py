"""Sanity statistics for a generated probe set."""
import collections
import sys
import time

from uav_eval.probes import generate, mirror_action

t = time.time()
P = generate(int(sys.argv[1]) if len(sys.argv) > 1 else 17, per_family=int(sys.argv[2]) if len(sys.argv) > 2 else 24)
print("probes", len(P), "secs", round(time.time() - t, 1))
c = collections.Counter((p["family"], p["variant"], p["truth"]["decision_eligible"]) for p in P)
for k in sorted(c):
    print(k, c[k])
by = {p["id"]: p for p in P}
bases = [p for p in P if p["variant"] == "base"]
bad = sum(sorted(mirror_action(a) for a in p["truth"]["safe_actions"]) != by[p["id"] + "_mirror"]["truth"]["safe_actions"]
          or p["truth"]["conflict_if_hold"] != by[p["id"] + "_mirror"]["truth"]["conflict_if_hold"] for p in bases)
print("mirror label mismatches", bad, "of", len(bases))
cf = [(p["truth"], by[p["id"] + "_cf"]["truth"]) for p in bases]
print("cf paper14 identical", sum(a["paper14"] == b["paper14"] for a, b in cf), "of", len(cf))
print("cf safe set differs", sum(a["safe_actions"] != b["safe_actions"] for a, b in cf))
print("cf approach flips", sum(a["approach"] != b["approach"] for a, b in cf))
print("conflict rate (all)", round(sum(p["truth"]["conflict_if_hold"] for p in P) / len(P), 2))
print("approach", collections.Counter(p["truth"]["approach"] for p in P))
print("side", collections.Counter(p["truth"]["side"] for p in P))
E = [p for p in P if p["truth"]["decision_eligible"]]
print("eligible", len(E), "mean |safe|", round(sum(len(p["truth"]["safe_actions"]) for p in E) / len(E), 2),
      "mean |good|", round(sum(len(p["truth"]["good_actions"]) for p in E) / len(E), 2))
for a in range(9):
    print(f"const {a}: safe {sum(a in p['truth']['safe_actions'] for p in E) / len(E):.2f} "
          f"good {sum(a in p['truth']['good_actions'] for p in E) / len(E):.2f}")
