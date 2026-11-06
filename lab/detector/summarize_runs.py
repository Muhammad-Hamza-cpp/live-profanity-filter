"""Summarize the training runs: pick each run's threshold on the real validation set, then report
its test numbers at that threshold. Groups seeds (name_s1, name_s2, ...) into mean +- std.

Threshold rule (validation only, at the given delay): maximize curse words silenced (%) plus
clean words intact (%); ties go to the higher threshold. The test set is never used to choose.

    python summarize_runs.py ../../data/lambda_backup/files/lab/detector/runs --delay 0.40
"""
import argparse
import re
import statistics as st
from pathlib import Path

LINE = re.compile(r"at ([\d.]+) s: curse silent (\d+)/(\d+)  clean intact (\d+)/(\d+) \([\d.]+%\)"
                  r"  sound-alikes intact (\d+)/(\d+)  muted ([\d.]+) s/min")


def parse(path):
    """eval.py log -> {threshold: {set: {delay: (silent%, clean%, soundalike%, muted s/min)}}}"""
    out, thr, cur = {}, None, None
    for line in open(path, errors="replace"):
        if line.startswith("=== threshold"):
            thr = float(line.split()[2].rstrip(","))
            out[thr] = {}
        elif line.startswith("["):
            cur = line[1:line.index("]")]
            out[thr][cur] = {}
        elif (m := LINE.search(line)) and thr is not None:
            s, n, c, cn, a, an = (int(m[i]) for i in range(2, 8))
            pct = lambda x, y: 100 * x / y if y else float("nan")
            out[thr][cur][float(m[1])] = (pct(s, n), pct(c, cn), pct(a, an), float(m[8]))
    return out


def pick(val, delay):
    score = lambda t: val[t]["real"][delay][0] + val[t]["real"][delay][1]
    return max(sorted(val), key=lambda t: (score(t), t))


def fmt(xs):
    return f"{st.mean(xs):.1f} ± {st.stdev(xs):.1f}" if len(xs) > 1 else f"{xs[0]:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path)
    ap.add_argument("--delay", type=float, default=0.40)
    args = ap.parse_args()
    d = args.delay

    rows = {}
    for run in sorted(args.runs.iterdir()):
        if not (run / "eval_val.log").exists() or not (run / "eval_test.log").exists():
            continue
        val, test = parse(run / "eval_val.log"), parse(run / "eval_test.log")
        t = pick(val, d)
        real, esc = test[t]["real"][d], test[t]["esc"][d]
        rows[run.name] = dict(thr=t, val=val[t]["real"][d][0] + val[t]["real"][d][1],
                              silent=real[0], clean=real[1], alike=real[2], muted=real[3], esc=esc[3])

    print(f"Per run (threshold chosen on validation; test numbers at {d:.2f} s delay)\n")
    print("| run | thr | val score | silenced % | clean intact % | sound-alikes intact % | muted s/min | ESC-50 muted s/min |")
    print("|---|---|---|---|---|---|---|---|")
    for n, r in rows.items():
        print(f"| {n} | {r['thr']:.1f} | {r['val']:.1f} | {r['silent']:.1f} | {r['clean']:.1f} | {r['alike']:.1f} "
              f"| {r['muted']:.1f} | {r['esc']:.1f} |")

    groups = {}
    for n, r in rows.items():
        groups.setdefault(re.sub(r"_s\d+$", "", n), []).append(r)
    print(f"\nPer group (mean ± std over seeds)\n")
    print("| group | seeds | thresholds | silenced % | clean intact % | sound-alikes intact % | muted s/min | ESC-50 muted s/min |")
    print("|---|---|---|---|---|---|---|---|")
    for g, rs in groups.items():
        col = lambda k: [r[k] for r in rs]
        print(f"| {g} | {len(rs)} | {'/'.join(f'{t:.1f}' for t in col('thr'))} | {fmt(col('silent'))} "
              f"| {fmt(col('clean'))} | {fmt(col('alike'))} | {fmt(col('muted'))} | {fmt(col('esc'))} |")

    trained = {n: r for n, r in rows.items() if n != "v3_shipped"}
    best = max(trained, key=lambda n: trained[n]["val"])
    print(f"\nBest run on validation: {best} at threshold {trained[best]['thr']:.1f}")


if __name__ == "__main__":
    main()
