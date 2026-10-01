"""Does the chosen action depend on the risk? Argmax action split by true and answered conflict."""
import json
import sys
from collections import Counter, defaultdict

probes = {json.loads(line)["id"]: json.loads(line) for line in open(sys.argv[1])}
for d in sys.argv[2:]:
    recs = defaultdict(dict)
    for line in open(d + "/records.jsonl"):
        r = json.loads(line)
        if r["probabilities"]:
            recs[r["style"]][(r["probe_id"], r["question"])] = r["probabilities"]
    for style, got in recs.items():
        split = defaultdict(Counter)
        for pid, p in probes.items():
            act, con = got.get((pid, "action")), got.get((pid, "conflict"))
            if not (act and con):
                continue
            a = max(act, key=act.get)
            said = max(con, key=con.get)
            split["truth=" + str(p["truth"]["conflict_if_hold"])][a] += 1
            split["said=" + said][a] += 1
        print(d.split("/")[-1], style)
        for k in sorted(split):
            print("   ", k, dict(split[k].most_common(4)))
