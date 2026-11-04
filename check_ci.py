import json, glob, sys
metric = sys.argv[1] if len(sys.argv) > 1 else "growth_dice"
nan = float("nan")
for f in sorted(glob.glob("results/eval_h*.json")):
    d = json.load(open(f))
    print(f"\nH={d['horizon']}  metric={metric}  best baseline: {d['best_baseline']}")
    for name, e in d["results"].items():
        if e["kind"] != "model":
            continue
        diff, lo, hi = [x if x is not None else nan
                        for x in e["vs_best_baseline"][metric]]
        verdict = "BETTER" if lo > 0 else ("WORSE" if hi < 0 else "tie")
        print(f"  {name:26s} {diff:+.3f} [{lo:+.3f}, {hi:+.3f}]  {verdict}")
