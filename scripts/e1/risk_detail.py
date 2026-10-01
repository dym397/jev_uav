import json
import sys

for d in sys.argv[1:]:
    s = json.load(open(d + "/summary.json"))
    st = s.get("styles", s)
    for k, v in st.items():
        if not isinstance(v, dict) or "conflict" not in v:
            continue
        out = []
        for q in ("approach", "side", "conflict"):
            x = v[q]
            out.append("%s modal=%.2f first=%.2f pT=%.2f" % (
                q, x.get("modal_answer_share", 0), x.get("picked_first_listed_rate", 0), x.get("mean_p_truth", 0)))
        print(d.split("/")[-1], k, " | ".join(out))
