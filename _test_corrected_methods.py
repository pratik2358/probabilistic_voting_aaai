#!/usr/bin/env python3
"""Verify the corrected exact methods on real elections.

Runs the three exact-family methods through the compare_ks wiring on
  1. Alaska_20221108_HouseDistrict18 -- the knife-edge race (k=3, U=676)
     where the previous implementations diverged
     (old exact 0.309 / FFT 0.148 / old Mult 0.033), and
  2. NewYorkCity_20230627_CON_CityCouncilD13 -- a decisive regression race
     where all methods must stay at 1.0.

Expected after the corrections (ground truth by independent enumeration):
  ExhaustDP (exact.py)      WP = 0.148418769  (mass 1.0)
  FFTProb   (unchanged)     WP ~ 0.1484       (float32)
  Mult      (memoryless)    WP = 0.148418769  (same model as ExhaustDP; mass 1.0)

Usage:
  python _test_corrected_methods.py [--data-root PATH] [--skip-mult] [--jobs N]

--data-root defaults to <repo>/data/dataverse_files; point it at an existing
copy of the dataverse CSVs if this checkout is code-only.
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "preprocessing"))

GROUND_TRUTH_HD18 = 0.148418769  # memoryless model, independent enumeration

RACES = [
    ("Alaska_20221108_HouseDistrict18", "hd18"),
    ("NewYorkCity_20230627_CON_CityCouncilD13", "regression"),
]


def wp_of(dist, cands, winner):
    from compare_ks import ensure_full_orders, winner_probs_from_order_dist
    return winner_probs_from_order_dist(ensure_full_orders(dist, tuple(cands))).get(winner, 0.0)


def run_race(name, kind, data_root, jobs, skip_mult):
    from preprocessing.parse_election_data import parse_election
    from preprocessing.formats.base_parser import BaseParser
    from compare_ks import (exact_baseline_once, fft_once, normalize_dist,
                            ensure_full_orders, compute_true_rcv_tally)
    from exact import exact_rcv_order_distribution_fair
    from multinomial_parallel import run_parallel_over_orders

    el = parse_election(os.path.join(data_root, f"{name}.csv"))
    cands = list(el.candidates)
    P = BaseParser.compute_first_choice_probs(el.bound_ballots, cands)
    U = len(el.unbound_ballots)

    # True winner: deterministic IRV over ALL cast ballots, mirroring the
    # pipeline (compute_true_rcv_tally returns a PARTIAL order of losers;
    # ensure_full_orders appends the surviving winner -- compare_ks:939-943).
    partial = compute_true_rcv_tally(el.bound_ballots, el.unbound_ballots, cands)
    true_order = list(ensure_full_orders({partial: 1.0}, tuple(cands)).keys())[0]
    winner = true_order[-1]

    print(f"\n== {name}: k={len(cands)} nb={len(el.bound_ballots)} U={U} "
          f"winner={winner}", flush=True)
    failures = []

    # ExhaustDP through the compare_ks wrapper (exercises the parallel path)
    d_par, _, _ = exact_baseline_once(tuple(cands), el.bound_ballots, P, U, jobs)
    wp_par = wp_of(d_par, cands, winner)
    # and the serial path directly
    d_ser = exact_rcv_order_distribution_fair(el.bound_ballots, U, P)
    wp_ser = wp_of(d_ser, cands, winner)
    print(f"  ExhaustDP (corrected)  WP = {wp_par:.9f}  "
          f"(serial {wp_ser:.9f}, mass {sum(d_ser.values()):.9f})", flush=True)
    if abs(wp_par - wp_ser) > 1e-9:
        failures.append(f"exact parallel/serial mismatch: {wp_par} vs {wp_ser}")

    d_fft, _, _ = fft_once(tuple(cands), el.bound_ballots, P, U)
    wp_fft = wp_of(d_fft, cands, winner)
    print(f"  FFTProb   (unchanged)  WP = {wp_fft:.6f}", flush=True)

    wp_mult = None
    if not skip_mult:
        raw = run_parallel_over_orders(candiates=cands, bound_ballots=el.bound_ballots,
                                       p_dict=P, m=U, approx=1e-6, n_jobs=jobs)
        wp_mult = wp_of(normalize_dist(raw), cands, winner)
        print(f"  Mult      (memoryless) WP = {wp_mult:.9f}  "
              f"(raw mass {sum(raw.values()):.6f})", flush=True)

    if kind == "hd18":
        if abs(wp_ser - GROUND_TRUTH_HD18) > 1e-6:
            failures.append(f"exact vs ground truth: {wp_ser:.9f} != {GROUND_TRUTH_HD18}")
        if abs(wp_fft - wp_ser) > 5e-4:
            failures.append(f"fft vs exact: {wp_fft:.6f} vs {wp_ser:.6f}")
        if wp_mult is not None and abs(wp_mult - wp_ser) > 1e-6:
            failures.append(f"mult vs exact: {wp_mult:.9f} vs {wp_ser:.9f}")
    else:  # regression: decisive race, everything at ~1
        for label, v in [("exact", wp_par), ("fft", wp_fft), ("mult", wp_mult)]:
            if v is not None and v < 0.9999:
                failures.append(f"{label} regressed on decisive race: {v:.6f}")

    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root",
                    default=os.path.join(REPO, "data", "dataverse_files"))
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--skip-mult", action="store_true",
                    help="skip the Mult check (slowest part, a few minutes)")
    args = ap.parse_args()

    all_failures = []
    for name, kind in RACES:
        all_failures += run_race(name, kind, args.data_root, args.jobs, args.skip_mult)

    print()
    if all_failures:
        for f in all_failures:
            print(f"FAIL: {f}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
