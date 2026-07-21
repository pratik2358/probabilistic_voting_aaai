#!/usr/bin/env python3
"""
Run compare_ks.py on real-world election data.

This script:
1. Parses election data from various formats
2. Runs compare_ks experiments for each election
3. Saves results in separate directories per election

Usage:
    python run_real_elections.py --election data/dataverse_files/<Election>.csv --outdir results/_raw
(normally invoked via run_experiments.py, which drives the elections.json corpus)
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

# Add preprocessing to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'preprocessing'))

from preprocessing.parse_election_data import ElectionDataParser
from preprocessing.formats.base_parser import ElectionData


def run_election_experiment(
    election: ElectionData,
    outdir: str,
    python_exe: str,
    repeats: int = 10,
    sampling_N: int = 5000,
    mult_eps: float = 1e-6,
    exact_jobs: int = 0,
    randomness: float = 0.0,
    mult_jobs: int = None,
    ks_bucket_size: int = 100,
    ks_bucket_sizes: list = None,
    ks_sigmafactor: float = 0.1,
    sampling_epsilon: float = 0.05,
    sampling_delta: float = 0.05,
    timeout_seconds: int = 3600,
    memory_limit_percent: float = 0.80,
    methods_to_run: list = None
):
    """
    Run compare_ks.py for a single election by creating a temporary wrapper script.
    
    The wrapper script sets up the election data and calls
    experiment_real_ballots_setup from compare_ks.py directly.
    """
    
    # Create election-specific output directory
    election_dir = os.path.join(outdir, election.name)
    os.makedirs(election_dir, exist_ok=True)
    
    # Save election metadata
    import json
    metadata_file = os.path.join(election_dir, 'election_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump({
            'name': election.name,
            'candidates': election.candidates,
            'n_bound': election.n_bound,
            'U': election.U,
            'P': election.P,
            'metadata': election.metadata
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Running experiment for: {election.name}")
    print(f"{'='*80}")
    print(election.summary())
    print(f"\nResults will be saved to: {election_dir}")
    print(f"{'='*80}\n")
    
    # Bucket sizes: a list -> each becomes its own ks_b<size> method row
    _ks_buckets = ks_bucket_sizes if ks_bucket_sizes else [ks_bucket_size]

    # Create temporary wrapper script that calls compare_ks functions directly
    wrapper_script = os.path.join(election_dir, '_temp_run_script.py')
    
    # Calculate the project root (where compare_ks.py is located)
    # This is the directory containing run_real_elections.py
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Determine which experimental setup to use
    use_real_ballots = (election.unbound_ballots is not None and 
                       len(election.unbound_ballots) > 0)
    
    if use_real_ballots:
        # Use experiment_real_ballots_setup with actual unbound ballots
        print(f"  Using experiment_real_ballots_setup (actual unbound ballots)")
        
        with open(wrapper_script, 'w', encoding='utf-8') as f:
            f.write(f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temporary wrapper script for running compare_ks on: {election.name}
Auto-generated - safe to delete after run
Using experiment_real_ballots_setup with actual unbound ballots
"""
import sys
import os
import json

# Add project root to path to import compare_ks
sys.path.insert(0, {repr(project_root)})

from compare_ks import experiment_real_ballots_setup

def main():
    """Main function - required for multiprocessing on macOS."""
    # Load election data
    candidates = {repr(election.candidates)}
    bound_ballots = {repr(election.bound_ballots)}
    unbound_ballots = {repr(election.unbound_ballots)}

    # Run experiment with real unbound ballots
    print("Starting experiment with real unbound ballots...")
    result = experiment_real_ballots_setup(
        outdir={repr(election_dir)},
        candidates=candidates,
        bound_ballots=bound_ballots,
        unbound_ballots=unbound_ballots,
        multinomial_eps={mult_eps},
        multinomial_jobs={repr(mult_jobs)},
        sampling_N={sampling_N},
        repeats={repeats},
        exact_jobs={exact_jobs},
        ks_buckets={_ks_buckets},
        ks_sigmafactor={ks_sigmafactor},
        sampling_epsilon={sampling_epsilon},
        sampling_delta={sampling_delta},
        timeout_seconds={timeout_seconds},
        memory_limit_percent={memory_limit_percent},
        methods_to_run={repr(methods_to_run)}
    )
    
    # Save results to CSV
    import csv
    csv_path = os.path.join({repr(election_dir)}, "summary.csv")
    with open(csv_path, "w", newline="") as csvfile:
        w = csv.writer(csvfile)
        w.writerow([
            "method",
            "time_mean", "time_sd",
            "mem_mb_mean", "mem_mb_sd",
            "ndcg_2_mean", "ndcg_2_sd",
            "ndcg_full_mean", "ndcg_full_sd",
            "expected_ndcg_full_mean", "expected_ndcg_full_sd",
            "matches_rcv_order_mean", "matches_rcv_order_sd",
            "matches_rcv_winner_mean", "matches_rcv_winner_sd",
            "winner_prob_mean", "winner_prob_sd",
            "n_success", "n_failures"
        ])
        w.writerow(result["exact_row"])
        w.writerow(result["fft_row"])
        for row in result["approx_rows"]:
            w.writerow(row)
        for row in result["ks_rows"]:
            w.writerow(row)

    # Also save true RCV info
    true_rcv_path = os.path.join({repr(election_dir)}, "true_rcv_order.json")
    with open(true_rcv_path, "w") as f:
        json.dump(result["true_rcv_info"], f, indent=2)

    print(f"Results saved to: {{csv_path}}")
    print(f"True RCV order saved to: {{true_rcv_path}}")
    print("Experiment complete!")
    return result

if __name__ == '__main__':
    result = main()
''')
    
    else:
        raise ValueError(
            f"{election.name}: parse produced no real unbound ballots. The"
            f" corrected pipeline runs only on real bound/unbound splits (the"
            f" strict parser guarantees this); the legacy simulated-unbound"
            f" path (experiment_one_setup) was removed."
        )

    
    # Make script executable and run it
    os.chmod(wrapper_script, 0o755)
    
    print(f"Running experiment (this may take several minutes)...\n")
    result = subprocess.run(
        [python_exe, wrapper_script],
        capture_output=False,
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Experiment failed with return code {result.returncode}")
    
    print(f"\n✓ Experiment completed successfully!")
    print(f"✓ Results saved to: {election_dir}/summary.csv")
    
    # Optionally delete the temporary script
    # os.remove(wrapper_script)
    
    return election_dir


def main():
    parser = argparse.ArgumentParser(
        description="Run compare_ks experiments on real election data"
    )
    
    parser.add_argument(
        '--election',
        required=True,
        help='Single election data file (CSV)'
    )
    
    # Output
    parser.add_argument(
        '--outdir',
        default='results_real',
        help='Output directory for all results (default: results_real)'
    )
    
    # NOTE: the legacy --U-method/--U-value/--P-method flags were removed.
    # StrictRealBallotParser reads bound/unbound exclusively from the file's
    # explicit ballot-type markings (U = count of real unbound ballots) and
    # the experiment computes P from bound first choices; the flags were
    # accepted but silently ignored.

    # Experiment parameters
    parser.add_argument(
        '--repeats',
        type=int,
        default=10,
        help='Number of repeats for each approximation method'
    )
    parser.add_argument(
        '--sampling-N',
        type=int,
        default=5000,
        help='Samples per order for sampling method'
    )
    parser.add_argument(
        '--mult-eps',
        type=float,
        default=1e-6,
        help='Epsilon for multinomial pruning'
    )
    parser.add_argument(
        '--exact-jobs',
        type=int,
        default=0,
        help='CPU cores for exact computation (0=all)'
    )
    parser.add_argument(
        '--ks-bucket-size',
        type=int,
        default=100,
        help='Bucket size for Kapoor-Staecker method'
    )
    parser.add_argument(
        '--ks-bucket-sizes',
        type=str,
        default=None,
        help='Comma-separated bucket sizes; each recorded as ks_b<size> (overrides --ks-bucket-size)'
    )
    parser.add_argument(
        '--ks-sigmafactor',
        type=float,
        default=0.1,
        help='Sigma factor for Kapoor-Staecker method'
    )
    parser.add_argument(
        '--sampling-epsilon',
        type=float,
        default=0.05,
        help='Epsilon for adaptive sampling (Hoeffding bound width target, default: 0.05)'
    )
    parser.add_argument(
        '--sampling-delta',
        type=float,
        default=0.05,
        help='Delta for adaptive sampling confidence (default: 0.05 = 95%% confidence)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=3600,
        help='Timeout in seconds for each method (default: 3600 = 1 hour)'
    )
    parser.add_argument(
        '--memory-limit',
        type=float,
        default=0.80,
        help='Memory limit as fraction of total RAM (default: 0.80)'
    )
    parser.add_argument(
        '--methods',
        type=str,
        default=None,
        help='Comma-separated list of methods to run (default: all). Options: exact,fft,multinomial,sampling,mvn,naive_est,top_count,ks'
    )
    
    args = parser.parse_args()
    
    # Find Python executable
    python_exe = sys.executable
    
    # Parse election(s)
    election_parser = ElectionDataParser()
    elections = []
    
    elections = [election_parser.parse(args.election)]
    
    if not elections:
        print("No elections found to process!")
        return 1
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    # Parse methods to run
    methods_to_run = None
    if args.methods:
        methods_to_run = [m.strip() for m in args.methods.split(',')]
        print(f"\n🎯 Running only methods: {', '.join(methods_to_run)}")
    else:
        print(f"\n✅ Running all methods (default)")

    ks_bucket_sizes = None
    if args.ks_bucket_sizes:
        ks_bucket_sizes = [int(x.strip()) for x in args.ks_bucket_sizes.split(',')]
    
    # Process each election
    results_summary = []
    for election in elections:
        try:
            result_dir = run_election_experiment(
                election=election,
                outdir=args.outdir,
                python_exe=python_exe,
                repeats=args.repeats,
                sampling_N=args.sampling_N,
                mult_eps=args.mult_eps,
                exact_jobs=args.exact_jobs,
                ks_bucket_size=args.ks_bucket_size,
                ks_bucket_sizes=ks_bucket_sizes,
                ks_sigmafactor=args.ks_sigmafactor,
                sampling_epsilon=args.sampling_epsilon,
                sampling_delta=args.sampling_delta,
                timeout_seconds=args.timeout,
                memory_limit_percent=args.memory_limit,
                methods_to_run=methods_to_run
            )
            results_summary.append((election.name, 'SUCCESS', result_dir))
        except Exception as e:
            print(f"✗ Failed to process {election.name}: {e}")
            results_summary.append((election.name, 'FAILED', str(e)))
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    success_count = sum(1 for _, status, _ in results_summary if status == 'SUCCESS')
    print(f"Processed {len(elections)} elections:")
    print(f"  ✓ Success: {success_count}")
    print(f"  ✗ Failed: {len(elections) - success_count}")
    print(f"\nResults saved in: {args.outdir}/")

    for name, status, info in results_summary:
        symbol = '✓' if status == 'SUCCESS' else '✗'
        print(f"  {symbol} {name}: {info}")


    print(f"\n{'='*80}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
