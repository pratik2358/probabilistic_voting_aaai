#!/usr/bin/env python3
"""Unified experiment runner.

Every election is dispatched the same way through run_real_elections.py,
which writes <out-root>/_raw/<election>/{summary.csv,summary.json,
election_metadata.json,true_rcv_order.json}.

Examples
--------
    # Everything still missing a summary.csv
    python run_experiments.py --all --skip-existing

    # One election
    python run_experiments.py --election Alaska_20221108_HouseDistrict18

    # All k<=4 races with the exact-family methods only
    python run_experiments.py --all --k-max 4 --methods exact,fft,multinomial

    # KS variants at sigma=1.0 with several bucket sizes
    python run_experiments.py --all --methods ks \\
        --ks-sigmafactor 1.0 --ks-bucket-sizes 100,400,700,1000
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(ROOT, "elections.json")
RUNNER = os.path.join(ROOT, "run_real_elections.py")
ALL_METHODS = ["exact", "fft", "multinomial", "sampling",
               "mvn", "naive_est", "top_count", "ks"]


def load_manifest():
    with open(MANIFEST) as f:
        return json.load(f)["elections"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument("--all", action="store_true",
                     help="run every election in elections.json")
    sel.add_argument("--election", help="comma-separated election name(s)")
    sel.add_argument("--list", action="store_true",
                     help="print the manifest summary and exit")
    p.add_argument("--k", help="only these candidate counts (comma-separated)")
    p.add_argument("--k-max", type=int, help="only races with k <= this")
    p.add_argument("--methods", default=",".join(ALL_METHODS),
                   help=f"comma-separated methods (default: all = {','.join(ALL_METHODS)})")
    p.add_argument("--data-root", default=os.path.join(ROOT, "data", "dataverse_files"),
                   help="directory holding <election>.csv ballot files")
    p.add_argument("--out-root", default=os.path.join(ROOT, "results"),
                   help="results root; per-election output under <out-root>/_raw/")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip elections whose summary.csv already exists")
    p.add_argument("--timeout", type=int, default=3600,
                   help="per-method timeout in seconds (default 3600)")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--ks-sigmafactor", type=float, default=1.0,
                   help="KS confidence factor (paper canonical: 1.0); "
                        "forwarded to run_real_elections.py")
    p.add_argument("--ks-bucket-sizes", default=None,
                   help="comma-separated bucket sizes, forwarded if set")
    p.add_argument("--smallest-first", action="store_true",
                   help="run in ascending n_bound+U order (quick races first)")
    p.add_argument("--quiet", action="store_true",
                   help="one status line per race only (full output still goes "
                        "to the per-race .run.log)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    elections = load_manifest()

    if args.list:
        from collections import Counter
        print(f"{len(elections)} elections "
              f"(by k: {dict(sorted(Counter(e['k'] for e in elections).items()))})")
        for e in elections:
            print(f"  k={e['k']:>2}  nb={e['n_bound']:>6}  U={e['U']:>6}  "
                  f"{e['name']}  [{'+'.join(e['sources'])}]")
        return

    if args.election:
        wanted = {n.strip() for n in args.election.split(",")}
        byname = {e["name"]: e for e in elections}
        missing = wanted - set(byname)
        if missing:
            sys.exit(f"not in elections.json: {', '.join(sorted(missing))}")
        elections = [byname[n] for n in sorted(wanted)]
    if args.k:
        ks = {int(x) for x in args.k.split(",")}
        elections = [e for e in elections if e["k"] in ks]
    if args.k_max is not None:
        elections = [e for e in elections if e["k"] <= args.k_max]

    raw_dir = os.path.join(args.out_root, "_raw")
    os.makedirs(raw_dir, exist_ok=True)

    todo = []
    for e in elections:
        csv_path = os.path.join(args.data_root, e["name"] + ".csv")
        summary = os.path.join(raw_dir, e["name"], "summary.csv")
        if not os.path.exists(csv_path):
            print(f"MISSING CSV, skipped: {csv_path}")
            continue
        if args.skip_existing and os.path.exists(summary):
            continue
        todo.append((e, csv_path))

    if args.smallest_first:
        todo.sort(key=lambda t: t[0]["n_bound"] + t[0]["U"])

    print(f"{len(todo)} election(s) to run "
          f"({len(elections) - len(todo)} skipped) -> {raw_dir}")
    if args.dry_run:
        for e, _ in todo:
            print(f"  would run k={e['k']:>2} {e['name']}")
        return

    ok = failed = 0
    t_all = time.time()
    for i, (e, csv_path) in enumerate(todo, 1):
        cmd = [sys.executable, RUNNER,
               "--election", csv_path,
               "--outdir", raw_dir,
               "--methods", args.methods,
               "--timeout", str(args.timeout),
               "--repeats", str(args.repeats)]
        if args.ks_sigmafactor is not None:
            cmd += ["--ks-sigmafactor", str(args.ks_sigmafactor)]
        if args.ks_bucket_sizes is not None:
            cmd += ["--ks-bucket-sizes", args.ks_bucket_sizes]

        log_path = os.path.join(raw_dir, f"{e['name']}.run.log")
        print(f"[{i}/{len(todo)}] k={e['k']} {e['name']} ...", flush=True)
        t0 = time.time()
        with open(log_path, "w") as log:
            if args.quiet:
                rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                    cwd=ROOT).returncode
            else:
                # stream the child's output live (method progress, completions)
                # while also writing the per-race log
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT,
                                        cwd=ROOT, text=True)
                for line in proc.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log.write(line)
                rc = proc.wait()
        produced = os.path.exists(os.path.join(raw_dir, e["name"], "summary.csv"))
        status = "ok" if (rc == 0 and produced) else f"FAILED (rc={rc}, see {log_path})"
        print(f"    {status} in {time.time()-t0:.0f}s", flush=True)
        ok += status == "ok"
        failed += status != "ok"

    print(f"\ndone: {ok} ok, {failed} failed in {(time.time()-t_all)/60:.1f} min")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
