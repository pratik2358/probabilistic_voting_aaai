from __future__ import annotations
import os, json, csv, time, math, tracemalloc, gc
from typing import Dict, Tuple, List, Optional
from collections import defaultdict, Counter

from exact import exact_rcv_order_distribution_fair_parallel
from fft_algo_opt import rcv_elimination_orders_prob_fft
from multinomial_parallel import run_parallel_over_orders as multinomial_run
from sampling_gpu import mc_over_feasible_orders
from cdf_plot_correction import rcv_elimination_orders_prob_mvn, pad_cdf_orders
from naive_est import naive_estimation
from top_count_est import top_count_last_winner_estimation

from irvprob import PreferenceSchedule, list_to_stringlist, ProbabilityTable
import time

def normalize_dist(d: Dict[Tuple[str, ...], float]) -> Dict[Tuple[str, ...], float]:
    s = sum(d.values())
    if s <= 0:
        return {k: 0.0 for k in d}
    return {k: v / s for k, v in d.items()}

def ensure_full_orders(d: Dict[Tuple[str, ...], float],
                       all_cands: Tuple[str, ...]) -> Dict[Tuple[str, ...], float]:
    m = len(all_cands)
    all_set = set(all_cands)
    out = defaultdict(float)
    for order, p in d.items():
        if not order:
            continue
        if len(order) == m:
            out[order] += p
        elif len(order) == m - 1:
            miss = list(all_set - set(order))
            if len(miss) == 1:
                out[tuple(order) + (miss[0],)] += p
    return dict(out)

def tvd_orders(p: Dict[Tuple[str, ...], float], q: Dict[Tuple[str, ...], float]) -> float:
    keys = set(p.keys()) | set(q.keys())
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)

def tvd_over_support(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p.keys()) | set(q.keys())
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)

def winner_probs_from_order_dist(order_dist: Dict[Tuple[str, ...], float]) -> Dict[str, float]:
    out = defaultdict(float)
    for order, pr in order_dist.items():
        if order:
            out[order[-1]] += pr
    s = sum(out.values())
    if s > 0:
        for k in list(out.keys()):
            out[k] /= s
    return dict(out)

def spearman_perm(order1: Tuple[str, ...], order2: Tuple[str, ...]) -> float:
    if (order1 is None) or (order2 is None):
        return float('nan')
    if len(order1) != len(order2) or set(order1) != set(order2):
        return float('nan')
    n = len(order1)
    r1 = {c: i for i, c in enumerate(order1, start=1)}
    r2 = {c: i for i, c in enumerate(order2, start=1)}
    ssd = sum((r1[c] - r2[c])**2 for c in r1)
    denom = n * (n**2 - 1)
    return 1.0 - 6.0 * ssd / denom if denom > 0 else float('nan')

def top_k_orders(d: Dict[Tuple[str, ...], float], k: int) -> List[Tuple[str, ...]]:
    return [o for (o, _) in sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:k]]

def mean_sd(arr: List[float]) -> Tuple[float, float]:
    if not arr:
        return float('nan'), float('nan')
    m = sum(arr) / len(arr)
    s = (sum((x - m)**2 for x in arr) / (len(arr) - 1))**0.5 if len(arr) > 1 else 0.0
    return m, s

def reverse_keys(d: Dict[Tuple[str, ...], float]) -> Dict[Tuple[str, ...], float]:
    return {tuple(reversed(k)): v for k, v in d.items()}

def validate_and_compute_tvd(
    candidate_set: Tuple[str, ...],
    baseline: Dict[Tuple[str, ...], float],
    subject: Dict[Tuple[str, ...], float],
    method_name: str = "unknown"
) -> Tuple[Dict[Tuple[str, ...], float], float]:
    """
    Validate order format and compute TVD without auto-correction.
    
    IMPORTANT: All methods (exact, fft, multinomial, sampling, mvn, ks) are verified 
    to use (eliminated_first, ..., winner) format. See ORDER_FORMAT_VERIFICATION.md.
    
    This function NO LONGER auto-corrects. If a method returns reversed format,
    it will use the as-is TVD (which will be higher/worse) and log a warning.
    This ensures bugs are visible in the metrics, not hidden by auto-correction.
    
    Args:
        candidate_set: Full set of candidates
        baseline: Baseline distribution (typically exact_ground_truth)
        subject: Distribution to compare (from a method)
        method_name: Name of method for debugging messages
        
    Returns:
        (subject_dist_full, tvd_as_is)
        - subject_dist_full: The original distribution (padded to full orders)
        - tvd_as_is: TVD computed without any reversal
    """
    # Exact returns partial orders like ('C2', 'C3'), KS returns full orders like ('C2', 'C3', 'C1')
    baseline_full = ensure_full_orders(baseline, candidate_set)
    subj_full = ensure_full_orders(subject, candidate_set)
    tvd_as_is = tvd_orders(baseline_full, subj_full)
    
    # Check if reversal would give lower TVD (indicates format bug)
    subj_rev = reverse_keys(subj_full)
    tvd_rev = tvd_orders(baseline_full, subj_rev)
    
    if tvd_rev < tvd_as_is:
        print(f"WARNING: {method_name} has format mismatch!")
        print(f"    TVD(as-is)={tvd_as_is:.6f}, TVD(reversed)={tvd_rev:.6f}")
        print(f"    NOT auto-correcting  TVD will reflect actual mismatch.")
        print(f"    All methods should use (eliminated_first, ..., winner) format.")
        print(f"    This indicates a BUG in {method_name} - please investigate!")
    
    # Always return as-is (no auto-correction)
    return subj_full, tvd_as_is

def time_and_peakmem(func, *args, **kwargs):
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    secs = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, secs, peak / (1024 * 1024)

def ks_once(cands: Tuple[str, ...], bound_ballots, P, U,
            bucketsize: int, bucketcap: Optional[int] = None, sigmafactor: float = 1.0):
    """
    Run KS method using standard ptfrombound (Kapoor-Staecker paper approach).
    Returns full distribution over elimination orders.
    
    Uses only bound ballot information to estimate unbound distribution - does NOT
    use the prior P. This is the method as described in the original KS paper.
    
    Args:
        cands: Tuple of candidates
        bound_ballots: List of bound ballots
        P: Prior probability dict (NOT used by KS - kept for API consistency)
        U: Number of unbound ballots
        bucketsize: Bucket size for discretization
        bucketcap: Optional cap on number of buckets
        sigmafactor: Confidence factor (higher = wider predictions, less confident)
    
    Returns:
        (distribution_dict, time_seconds, memory_mb)
    """
    def _call():
        from collections import Counter, defaultdict
        from irvsimulator import ptfrombound
        
        # Convert ballots to PreferenceSchedule
        votesdict = {}
        counts = Counter(tuple(b) for b in bound_ballots)
        for ballot_tuple, ct in counts.items():
            votesdict[list_to_stringlist(list(ballot_tuple))] = int(ct)
        boundps = PreferenceSchedule(votesdict)
        
        # Build ProbabilityTable using ptfrombound (standard KS method)
        # Uses bound ballot empirical proportions, NOT external prior
        pt = ptfrombound(
            boundps, 
            unboundnumber=int(U), 
            bucketsize=int(bucketsize), 
            sigmafactor=float(sigmafactor)
        )
        if bucketcap is not None:
            pt.bucketcap = int(bucketcap)
        
        # Convert to full elimination order distribution
        def _elim_order_distribution(pt: ProbabilityTable) -> Dict[Tuple[str, ...], float]:
            """Recursively build full elimination-order distribution."""
            cands = pt.candidates[:]
            if len(cands) == 1:
                return {(cands[0],): 1.0}
            
            eps = pt.elimination_probabilities_tieweighted()
            out = defaultdict(float)
            for c in cands:
                p = float(eps.get(c, 0.0))
                if p <= 0.0:
                    continue
                child = pt.eliminate(c)
                child_dist = _elim_order_distribution(child)
                for suffix, psuf in child_dist.items():
                    out[(c,) + suffix] += p * float(psuf)
            return dict(out)
        
        return _elim_order_distribution(pt)
    
    dist, secs, mb = time_and_peakmem(_call)
    return normalize_dist(dist), secs, mb


def exact_baseline_once(cands: Tuple[str, ...], bound_ballots, P, U,
                        exact_jobs: Optional[int] = None):
    """
    Run the exact baseline in parallel.
    - exact_jobs is the requested number of worker processes.
      If exact_jobs in {None, 0} => use all available cores (handled by exact.py).
    """
    def _call():
        n_jobs = None if (exact_jobs is None or exact_jobs == 0) else exact_jobs
        return exact_rcv_order_distribution_fair_parallel(bound_ballots, U, P, n_jobs=n_jobs)
    dist, secs, mb = time_and_peakmem(_call)
    return normalize_dist(dist), secs, mb

def fft_once(cands: Tuple[str, ...], bound_ballots, P, U):
    def _call():
        return rcv_elimination_orders_prob_fft(bound_ballots, P, U)
    dist, secs, mb = time_and_peakmem(_call)
    return normalize_dist(dist), secs, mb

def multinomial_once(cands: Tuple[str, ...], bound_ballots, P, U,
                     approx_eps: float, n_jobs: Optional[int]):
    def _call():
        return multinomial_run(
            candiates=list(cands),
            bound_ballots=bound_ballots,
            p_dict=P,
            m=U,
            approx=approx_eps,
            n_jobs=n_jobs
        )
    dist, secs, mb = time_and_peakmem(_call)
    return normalize_dist(dist), secs, mb

def sampling_once(cands: Tuple[str, ...], bound_ballots, P, U,
                  N_per_order: int,
                  epsilon: Optional[float] = None,
                  sigma: float = 0.5,
                  delta: float = 0.05,
                  N_cap_per_order: Optional[int] = None):
    """
    Run sampling method once.
    
    Args:
        cands: Candidate names
        bound_ballots: Already-counted ballots
        P: Prior probability distribution
        U: Number of unbound ballots
        N_per_order: Base number of samples per order (used if epsilon=None)
        epsilon: Target Hoeffding width for adaptive sampling (None = fixed sampling)
        sigma: Standard deviation parameter for adaptive sampling (default 0.5)
        N_cap_per_order: Maximum samples per order for adaptive sampling (None = use N_per_order)
    
    Returns:
        Tuple of (normalized_distribution, time_seconds, memory_mb, order_stats, meta)
    """
    order_stats_out = {}
    meta_out = {}
    
    def _call():
        nonlocal order_stats_out, meta_out
        # Call with new adaptive parameters
        result = mc_over_feasible_orders(
            list(cands), bound_ballots, P, U,
            N_per_order=N_per_order,
            epsilon=epsilon,
            sigma=sigma,
            N_cap_per_order=N_cap_per_order,
            delta=delta,
            n_jobs_feas=None,
            n_jobs_mc_orders=None,
            seed=None,
            use_torch=False,
            torch_device=None
        )
        
        # New adaptive version returns (order_probs, order_stats, meta)
        if isinstance(result, tuple) and len(result) == 3:
            order_probs, order_stats, meta = result
            order_stats_out = order_stats
            meta_out = meta
            # Extract probabilities from (prob, (lo, hi)) tuples
            return {order: v[0] for order, v in order_probs.items()}
        else:
            # Fallback for old format (shouldn't happen with sampling_gpu_new)
            return {order: v[0] if isinstance(v, tuple) else v for order, v in result.items()}
    
    dist, secs, mb = time_and_peakmem(_call)
    return normalize_dist(dist), secs, mb, order_stats_out, meta_out

def mvn_once(cands: Tuple[str, ...], bound_ballots, P, U, is_samples: int):
    def _call():
        d = rcv_elimination_orders_prob_mvn(
            bound_ballots, P, U,
            continuity_correction=True,
            is_samples=is_samples,
            return_tree=False
        )
        return pad_cdf_orders(d, cands)
    dist, secs, mb = time_and_peakmem(_call)
    return normalize_dist(dist), secs, mb

def naive_once(cands: Tuple[str, ...], bound_ballots, P, U, eps: float = 1e-6):
    """
    Run naive estimator baseline (ignores U and P, uses only bound ballots).
    
    Args:
        cands: Candidate names
        bound_ballots: Already-counted ballots
        P: Prior (unused by naive method)
        U: Number of unbound ballots (unused by naive method)
        eps: Small constant to avoid division by zero (default 1e-6)
    
    Returns:
        Tuple of (normalized_distribution, time_seconds, memory_mb)
    """
    def _call():
        return naive_estimation(bound_ballots, list(cands), eps=eps)
    dist, secs, mb = time_and_peakmem(_call)
    return normalize_dist(dist), secs, mb

def top_count_once(cands: Tuple[str, ...], bound_ballots, P, U):
    """
    Run top-count last-winner estimator (uses only bound ballot first choices).

    Args:
        cands: Candidate names
        bound_ballots: Already-counted ballots
        P: Prior (unused by this method)
        U: Number of unbound ballots (unused by this method)

    Returns:
        Tuple of (normalized_distribution, time_seconds, memory_mb)
    """
    def _call():
        return top_count_last_winner_estimation(bound_ballots, list(cands))
    dist, secs, mb = time_and_peakmem(_call)
    return normalize_dist(dist), secs, mb

KS_BUCKET_SIZES = [1, 10, 50, 100, 200]

def compute_true_rcv_tally(bound_ballots: List[List[str]], 
                           unbound_ballots: List[List[str]], 
                           candidates: List[str]) -> Tuple[str, ...]:
    """
    Compute deterministic RCV outcome by running RCV on ALL actual ballots.
    
    This is the "true" outcome if all ballots were counted - used as reference
    point in correlation stress tests to see if predictions match reality.
    
    Args:
        bound_ballots: Ballots already counted (e.g., Election Day)
        unbound_ballots: Ballots that will arrive later (e.g., Absentee)
        candidates: Full list of candidates
    
    Returns:
        Elimination order tuple (first_eliminated, ..., winner)
    
    Example:
        >>> bound = [["A", "B"], ["B", "A"]]
        >>> unbound = [["C", "A"], ["C", "B"]]
        >>> compute_true_rcv_tally(bound, unbound, ["A", "B", "C"])
        ('B', 'A', 'C')  # B eliminated first, then A, C wins
    """
    from exact import enumerate_all_orders_with_weights
    
    # Combine all ballots
    all_ballots = bound_ballots + unbound_ballots
    
    # Run RCV tally - enumerate_all_orders_with_weights handles ties fairly
    order_dist = enumerate_all_orders_with_weights(all_ballots, candidates)
    
    # Return the order with highest weight (should be 1.0 for deterministic outcome)
    if not order_dist:
        # Fallback: return candidates in original order if RCV fails
        return tuple(candidates)
    
    best_order = max(order_dist.items(), key=lambda kv: kv[1])[0]
    return best_order


def _run_method_with_timeout_helper(method_func, args, kwargs, queue):
    """
    Helper function for running methods in subprocess with timeout.
    Must be at module level for pickling.
    """
    try:
        result = method_func(*args, **kwargs)
        queue.put(('success', result))
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        queue.put(('error', error_msg))


def experiment_real_ballots_setup(
    outdir: str,
    candidates: List[str],
    bound_ballots: List[List[str]],
    unbound_ballots: List[List[str]],
    multinomial_eps: float,
    multinomial_jobs: Optional[int],
    sampling_N: int,
    repeats: int,
    exact_jobs: Optional[int] = None,
    ks_buckets: Optional[List[int]] = None,
    ks_sigmafactor: float = 1.0,
    ks_bucketcap: Optional[int] = None,
    sampling_epsilon: Optional[float] = None,
    sampling_sigma: float = 0.5,
    sampling_delta: float = 0.05,
    sampling_N_cap: Optional[int] = None,
    timeout_seconds: int = 3600,
    memory_limit_percent: float = 0.80,
    methods_to_run: Optional[List[str]] = None
):
    """
    Real election experiment with actual bound and unbound ballots.
    
    Key characteristics:
    1. Ground Truth: True RCV tally from ALL actual ballots (bound + unbound)
    2. Prior P: Estimated from bound ballot first-choice votes
    3. All methods (including Exact) evaluated as methods, not ground truth
    4. Metrics: NDCG, binary winner/order matching, time, memory
    5. Timeout and memory limits for each method
    
    Args:
        outdir: Output directory for results
        candidates: List of candidate names
        bound_ballots: Actual bound ballots (e.g., Election Day + Early Voting)
        unbound_ballots: Actual unbound ballots (e.g., Absentee)
        multinomial_eps: Pruning epsilon for multinomial method
        multinomial_jobs: Worker count for multinomial
        sampling_N: Samples per order for Monte Carlo
        repeats: Repetitions for stochastic methods
        exact_jobs: Worker count for exact method
        ks_buckets: List of KS bucket sizes to test
        ks_sigmafactor: Confidence factor for KS method
        ks_bucketcap: Optional cap on buckets for KS
        sampling_epsilon: Target Hoeffding width for adaptive sampling
        sampling_sigma: Standard deviation for adaptive sampling
        sampling_N_cap: Maximum samples per order
        timeout_seconds: Maximum time per method run (default 3600s = 1 hour)
        memory_limit_percent: Maximum memory usage (default 0.80 = 80%)
        methods_to_run: List of methods to run (default None = all methods).
                       Options: 'exact', 'fft', 'multinomial', 'sampling', 'mvn', 'naive_est', 'ks'
                       Example: ['sampling', 'naive_est'] to run only those two
    
    Returns:
        Dict with rows for CSV output:
        - exact_row, fft_row, approx_rows, ks_rows
        - true_rcv_info: Dict with true RCV outcome
    """
    import psutil
    import multiprocessing as mp
    import queue as pyqueue
    import time as time_module
    from compute_ndcg import compute_ndcg, compute_expected_ndcg
    
    cands = tuple(candidates)
    U = len(unbound_ballots)
    n_bound = len(bound_ballots)
    
    # Get total system memory
    total_memory_gb = psutil.virtual_memory().total / (1024**3)
    memory_limit_gb = total_memory_gb * memory_limit_percent
    
    print(f"System memory: {total_memory_gb:.1f} GB, limit: {memory_limit_gb:.1f} GB ({memory_limit_percent*100:.0f}%)")
    print(f"Timeout per method: {timeout_seconds}s ({timeout_seconds/60:.1f} minutes)")
    
    def check_memory_limit():
        """Check if memory usage exceeds limit."""
        current_memory_gb = psutil.virtual_memory().used / (1024**3)
        if current_memory_gb > memory_limit_gb:
            raise MemoryError(f"Memory usage {current_memory_gb:.1f} GB exceeds limit {memory_limit_gb:.1f} GB")
    
    # Step 1: Compute P_estimated from bound ballots
    print("Computing prior P from bound ballot first-choice votes...")
    from preprocessing.formats.base_parser import BaseParser
    P_estimated = BaseParser.compute_first_choice_probs(bound_ballots, candidates)
    print(f"  P_estimated: {P_estimated}")
    
    # Step 2: Compute True RCV Tally with ALL actual ballots
    print("Computing TRUE RCV TALLY with all actual ballots...")
    true_rcv_order_raw = compute_true_rcv_tally(bound_ballots, unbound_ballots, candidates)
    true_rcv_full_dict = ensure_full_orders({true_rcv_order_raw: 1.0}, cands)
    true_rcv_order = list(true_rcv_full_dict.keys())[0] if true_rcv_full_dict else tuple(cands)
    true_rcv_winner = true_rcv_order[-1] if true_rcv_order else None
    print(f"  True RCV: {true_rcv_order}, winner: {true_rcv_winner}")
    
    # Storage for results
    def _empty_storage():
        return {'runs': [], 'times': [], 'mems': [], 'ndcg_2': [], 'ndcg_full': [],
                'expected_ndcg': [], 'match_order': [], 'match_winner': [],
                'winner_prob': [], 'failures': []}

    results = {
        'exact': _empty_storage(),
        'fft': _empty_storage(),
        'mvn': _empty_storage(),
        'multinomial': _empty_storage(),
        'sampling': _empty_storage(),
        'naive_est': _empty_storage(),
        'top_count': _empty_storage(),
        'ks': {}  # Will be populated per bucket size
    }
    
    def _kill_descendants(proc):
        """Kill the method child's own workers (e.g. Exact's pool) before
        terminating it -- SIGTERM on the child alone orphans them, and the
        orphans keep computing and hold inherited pipe fds open."""
        try:
            for c in psutil.Process(proc.pid).children(recursive=True):
                try:
                    c.kill()
                except Exception:
                    pass
        except Exception:
            pass

    def safe_run_method(method_name, method_func, *args, **kwargs):
        """Run a method with timeout and memory limits using multiprocessing."""
        # Check memory before starting
        try:
            check_memory_limit()
        except MemoryError as e:
            print(f"  💾 {method_name} MEMORY ERROR: {e}")
            return None, "memory"
        
        # Create queue for result
        queue = mp.Queue()
        
        # Start method in subprocess
        process = mp.Process(
            target=_run_method_with_timeout_helper,
            args=(method_func, args, kwargs, queue)
        )
        
        start_time = time_module.time()
        process.start()
        payload = None
        while payload is None:
            try:
                payload = queue.get(timeout=1.0)
            except pyqueue.Empty:

                try:
                    check_memory_limit()
                except MemoryError as e:
                    elapsed = time_module.time() - start_time
                    print(f"  💾 {method_name} MEMORY ERROR after {elapsed:.1f}s: {e}")
                    _kill_descendants(process)
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                        process.join()
                    return None, "memory"
                if time_module.time() - start_time > timeout_seconds:
                    break
                if not process.is_alive():
                    break  # child exited without producing a result
        elapsed = time_module.time() - start_time

        if payload is None and process.is_alive():
            # Timeout occurred
            print(f" {method_name} TIMEOUT after {elapsed:.1f}s")
            _kill_descendants(process)
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
            return None, "timeout"

        # Result (if any) is drained, so the child can flush and exit promptly.
        process.join(timeout=10)
        if process.is_alive():
            _kill_descendants(process)
            process.terminate()
            process.join()

        # Check if method succeeded
        if payload is not None:
            status, result = payload
            if status == 'success':
                # Check memory after completion
                try:
                    check_memory_limit()
                except MemoryError as e:
                    print(f"  💾 {method_name} MEMORY ERROR: {e}")
                    return None, "memory"
                return result, None
            else:
                print(f"{method_name} ERROR: {result}")
                return None, f"error: {result}"
        else:
            # Process exited without putting result
            print(f" {method_name} ERROR: Process exited unexpectedly")
            return None, "error: process_exit"
    
    def collect_metrics(dist_raw, time_s, mem_mb, method_name, storage_dict):
        """Compute and store metrics for a method run."""
        if dist_raw is None:
            return

        dist = ensure_full_orders(dist_raw, cands)
        best_order = max(dist.items(), key=lambda kv: kv[1])[0] if dist else None

        # NDCG metrics
        k_full = len(cands)
        ndcg_2 = compute_ndcg(true_rcv_order, best_order, k=2, order_format="eliminated_to_winner") if best_order else 0.0
        ndcg_full = compute_ndcg(true_rcv_order, best_order, k=k_full, order_format="eliminated_to_winner") if best_order else 0.0
        expected_ndcg = compute_expected_ndcg(true_rcv_order, dist, k=k_full, order_format="eliminated_to_winner") if dist else 0.0

        # Binary matching
        match_order = 1 if best_order == true_rcv_order else 0
        winner_pred = best_order[-1] if best_order else None
        match_winner = 1 if winner_pred == true_rcv_winner else 0

        # Probability assigned to the true winner.
        # Use the padded `dist` (k-tuples ending in the winner) rather than
        # `dist_raw`, because some methods (e.g. Exact) emit (k-1)-tuples of
        # eliminations only — `ensure_full_orders` appends the missing
        # surviving candidate so `order[-1]` is reliably the winner.
        winner_probs = winner_probs_from_order_dist(dist)
        winner_prob = winner_probs.get(true_rcv_winner, 0.0) if true_rcv_winner else float('nan')

        # Store results
        storage_dict['runs'].append(dist)
        storage_dict['times'].append(time_s)
        storage_dict['mems'].append(mem_mb)
        storage_dict['ndcg_2'].append(ndcg_2)
        storage_dict['ndcg_full'].append(ndcg_full)
        storage_dict['expected_ndcg'].append(expected_ndcg)
        storage_dict['match_order'].append(match_order)
        storage_dict['match_winner'].append(match_winner)
        storage_dict['winner_prob'].append(winner_prob)
    
    # Initialize methods to run (default: all methods)
    if methods_to_run is None:
        methods_to_run = ['exact', 'fft', 'multinomial', 'sampling', 'mvn', 'naive_est', 'top_count', 'ks']
    
    print(f"\nMethods to run: {', '.join(methods_to_run)}")
    
    # Step 3: Run Exact method (single run)
    if 'exact' in methods_to_run:
        print("\nRunning Exact method...")
        result, failure = safe_run_method(
            "Exact",
            exact_baseline_once,
            cands, bound_ballots, P_estimated, U, exact_jobs
        )
        if result:
            dist_raw, time_s, mem_mb = result
            collect_metrics(dist_raw, time_s, mem_mb, "Exact", results['exact'])
            print(f" Exact completed in {time_s:.2f}s (memory: {mem_mb:.1f} MB)")
        else:
            results['exact']['failures'].append(failure)
            print(f" Exact failed: {failure}")
    else:
        print("\nSkipping Exact method")
    
    # Step 4: Run FFT method (single run)
    if 'fft' in methods_to_run:
        print("\nRunning FFT method...")
        result, failure = safe_run_method(
            "FFT",
            fft_once,
            cands, bound_ballots, P_estimated, U
        )
        if result:
            dist_raw, time_s, mem_mb = result
            collect_metrics(dist_raw, time_s, mem_mb, "FFT", results['fft'])
            print(f"FFT completed in {time_s:.2f}s (memory: {mem_mb:.1f} MB)")
        else:
            results['fft']['failures'].append(failure)
            print(f"FFT failed: {failure}")
    else:
        print("\nSkipping FFT method")
    
    # Step 5: Run approximation methods with repeats
    # Multinomial is deterministic (given epsilon), so run only once
    if 'multinomial' in methods_to_run:
        print(f"\nRunning Multinomial method (deterministic, single run)...")
        result, failure = safe_run_method(
            "Multinomial",
            multinomial_once,
            cands, bound_ballots, P_estimated, U, multinomial_eps, multinomial_jobs
        )
        if result:
            dist_raw, time_s, mem_mb = result
            collect_metrics(dist_raw, time_s, mem_mb, "Multinomial", results['multinomial'])
            print(f"Multinomial completed in {time_s:.2f}s")
        else:
            results['multinomial']['failures'].append(failure)
            print(f"Multinomial failed: {failure}")
    else:
        print("\nSkipping Multinomial method")
    
    if 'sampling' in methods_to_run:
        print(f"\nRunning Sampling method ({repeats} repeats)...")
        for i in range(repeats):
            result, failure = safe_run_method(
                f"Sampling (run {i+1})",
                sampling_once,
                cands, bound_ballots, P_estimated, U, sampling_N,
                sampling_epsilon, sampling_sigma, sampling_delta, sampling_N_cap
            )
            if result:
                result_tuple = result
                dist_raw, time_s, mem_mb, order_stats, meta = result_tuple
                collect_metrics(dist_raw, time_s, mem_mb, f"Sampling_{i+1}", results['sampling'])
                print(f"Sampling run {i+1}/{repeats} completed in {time_s:.2f}s")
            else:
                results['sampling']['failures'].append(failure)
                print(f"Sampling run {i+1}/{repeats} failed: {failure}")
                if i == 0:
                    print(f"  Skipping remaining {repeats-1} Sampling repeats due to first-run failure")
                    break
    else:
        print("\nSkipping Sampling method")
    
    if 'mvn' in methods_to_run:
        print(f"\nRunning MVN method ({repeats} repeats)...")
        for i in range(repeats):
            result, failure = safe_run_method(
                f"MVN (run {i+1})",
                mvn_once,
                cands, bound_ballots, P_estimated, U, 40000
            )
            if result:
                dist_raw, time_s, mem_mb = result
                collect_metrics(dist_raw, time_s, mem_mb, f"MVN_{i+1}", results['mvn'])
                print(f"MVN run {i+1}/{repeats} completed in {time_s:.2f}s")
            else:
                results['mvn']['failures'].append(failure)
                print(f"MVN run {i+1}/{repeats} failed: {failure}")
                if i == 0:
                    print(f"  Skipping remaining {repeats-1} MVN repeats due to first-run failure")
                    break
    else:
        print("\nSkipping MVN method")
    
    if 'naive_est' in methods_to_run:
        print(f"\nRunning Naive Estimator method ({repeats} repeats)...")
        for i in range(repeats):
            result, failure = safe_run_method(
                f"Naive (run {i+1})",
                naive_once,
                cands, bound_ballots, P_estimated, U, 1e-6
            )
            if result:
                dist_raw, time_s, mem_mb = result
                collect_metrics(dist_raw, time_s, mem_mb, f"Naive_{i+1}", results['naive_est'])
                print(f"Naive run {i+1}/{repeats} completed in {time_s:.2f}s")
            else:
                results['naive_est']['failures'].append(failure)
                print(f"Naive run {i+1}/{repeats} failed: {failure}")
                if i == 0:
                    print(f"  Skipping remaining {repeats-1} Naive repeats due to first-run failure")
                    break
    else:
        print("\nSkipping Naive Estimator method")

    if 'top_count' in methods_to_run:
        print(f"\nRunning Top-Count Last-Winner Estimator ({repeats} repeats)...")
        for i in range(repeats):
            result, failure = safe_run_method(
                f"TopCount (run {i+1})",
                top_count_once,
                cands, bound_ballots, P_estimated, U
            )
            if result:
                dist_raw, time_s, mem_mb = result
                collect_metrics(dist_raw, time_s, mem_mb, f"TopCount_{i+1}", results['top_count'])
                print(f"TopCount run {i+1}/{repeats} completed in {time_s:.2f}s")
            else:
                results['top_count']['failures'].append(failure)
                print(f"TopCount run {i+1}/{repeats} failed: {failure}")
                if i == 0:
                    print(f"  Skipping remaining {repeats-1} TopCount repeats due to first-run failure")
                    break
    else:
        print("\nSkipping Top-Count Estimator method")
    
    # Step 6: Run KS method with different bucket sizes
    if 'ks' in methods_to_run:
        if ks_buckets is None:
            ks_buckets = [100]
        
        print(f"\nRunning KS method with bucket sizes: {ks_buckets}")
        for bs in ks_buckets:
            ks_key = f'ks_b{bs}'
            results['ks'][ks_key] = _empty_storage()
            
            print(f"  Running KS with bucketsize={bs} ({repeats} repeats)...")
            for i in range(repeats):
                result, failure = safe_run_method(
                    f"KS_b{bs} (run {i+1})",
                    ks_once,
                    cands, bound_ballots, P_estimated, U, bs, ks_bucketcap, ks_sigmafactor
                )
                if result:
                    dist_raw, time_s, mem_mb = result
                    collect_metrics(dist_raw, time_s, mem_mb, f"KS_b{bs}_{i+1}", results['ks'][ks_key])
                    print(f" KS_b{bs} run {i+1}/{repeats} completed in {time_s:.2f}s")
                else:
                    results['ks'][ks_key]['failures'].append(failure)
                    print(f" KS_b{bs} run {i+1}/{repeats} failed: {failure}")
                    if i == 0:
                        print(f"    Skipping remaining {repeats-1} KS_b{bs} repeats due to first-run failure")
                        break
    else:
        print("\nSkipping KS method")
    
    print("\nAll methods completed!")
    
    # Step 7: Format output
    def mean_or_nan(xs):
        if not xs:
            return float('nan')
        valid = [x for x in xs if not math.isnan(x)]
        return sum(valid) / len(valid) if valid else float('nan')
    
    def std_or_nan(xs):
        if not xs or len(xs) < 2:
            return float('nan')
        valid = [x for x in xs if not math.isnan(x)]
        if len(valid) < 2:
            return float('nan')
        m = sum(valid) / len(valid)
        var = sum((x - m) ** 2 for x in valid) / (len(valid) - 1)
        return math.sqrt(var)
    
    def format_row(method_name, storage_dict):
        """Format a row for CSV output."""
        n_success = len(storage_dict['runs'])
        n_failure = len(storage_dict['failures'])
        
        return [
            method_name,
            mean_or_nan(storage_dict['times']), std_or_nan(storage_dict['times']),
            mean_or_nan(storage_dict['mems']), std_or_nan(storage_dict['mems']),
            mean_or_nan(storage_dict['ndcg_2']), std_or_nan(storage_dict['ndcg_2']),
            mean_or_nan(storage_dict['ndcg_full']), std_or_nan(storage_dict['ndcg_full']),
            mean_or_nan(storage_dict['expected_ndcg']), std_or_nan(storage_dict['expected_ndcg']),
            mean_or_nan(storage_dict['match_order']), std_or_nan(storage_dict['match_order']),
            mean_or_nan(storage_dict['match_winner']), std_or_nan(storage_dict['match_winner']),
            mean_or_nan(storage_dict['winner_prob']), std_or_nan(storage_dict['winner_prob']),
            n_success, n_failure
        ]
    
    exact_row = format_row("exact", results['exact'])
    fft_row = format_row("fft", results['fft'])
    
    approx_rows = [
        format_row("multinomial", results['multinomial']),
        format_row("sampling", results['sampling']),
        format_row("mvn", results['mvn']),
        format_row("naive_est", results['naive_est']),
        format_row("top_count", results['top_count'])
    ]
    
    ks_rows = [format_row(ks_key, results['ks'][ks_key]) for ks_key in results['ks']]
    
    # True RCV info
    true_rcv_info = {
        "elimination_order": list(true_rcv_order),
        "winner": true_rcv_winner
    }
    
    # Step 8: Save results
    setup_dir = os.path.join(outdir, f"nb{n_bound}_U{U}")
    os.makedirs(setup_dir, exist_ok=True)
    
    csv_path = os.path.join(setup_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
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
        w.writerow(exact_row)
        w.writerow(fft_row)
        for row in approx_rows:
            w.writerow(row)
        for row in ks_rows:
            w.writerow(row)
    
    print(f"\nCSV saved to: {csv_path}")
    
    # Save detailed JSON
    summary = {
        "config": {
            "candidates": list(cands),
            "n_bound": n_bound,
            "U": U,
            "repeats": repeats,
            "timeout_seconds": timeout_seconds,
            "memory_limit_gb": memory_limit_gb
        },
        "P_estimated": P_estimated,
        "true_rcv_tally": true_rcv_info,
        "exact": {
            "runs": [{"|||".join(k): v for k, v in d.items()} for d in results['exact']['runs']],
            "times": results['exact']['times'],
            "mems": results['exact']['mems'],
            "failures": results['exact']['failures']
        },
        "fft": {
            "runs": [{"|||".join(k): v for k, v in d.items()} for d in results['fft']['runs']],
            "times": results['fft']['times'],
            "mems": results['fft']['mems'],
            "failures": results['fft']['failures']
        },
        "multinomial": {
            "runs": [{"|||".join(k): v for k, v in d.items()} for d in results['multinomial']['runs']],
            "times": results['multinomial']['times'],
            "mems": results['multinomial']['mems'],
            "failures": results['multinomial']['failures']
        },
        "sampling": {
            "runs": [{"|||".join(k): v for k, v in d.items()} for d in results['sampling']['runs']],
            "times": results['sampling']['times'],
            "mems": results['sampling']['mems'],
            "failures": results['sampling']['failures']
        },
        "mvn": {
            "runs": [{"|||".join(k): v for k, v in d.items()} for d in results['mvn']['runs']],
            "times": results['mvn']['times'],
            "mems": results['mvn']['mems'],
            "failures": results['mvn']['failures']
        },
        "naive_est": {
            "runs": [{"|||".join(k): v for k, v in d.items()} for d in results['naive_est']['runs']],
            "times": results['naive_est']['times'],
            "mems": results['naive_est']['mems'],
            "failures": results['naive_est']['failures']
        },
        "top_count": {
            "runs": [{"|||".join(k): v for k, v in d.items()} for d in results['top_count']['runs']],
            "times": results['top_count']['times'],
            "mems": results['top_count']['mems'],
            "failures": results['top_count']['failures']
        },
        "ks": {
            ks_key: {
                "runs": [{"|||".join(k): v for k, v in d.items()} for d in results['ks'][ks_key]['runs']],
                "times": results['ks'][ks_key]['times'],
                "mems": results['ks'][ks_key]['mems'],
                "failures": results['ks'][ks_key]['failures']
            }
            for ks_key in results['ks']
        }
    }
    
    json_path = os.path.join(setup_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"JSON saved to: {json_path}")
    
    return {
        "exact_row": exact_row,
        "fft_row": fft_row,
        "approx_rows": approx_rows,
        "ks_rows": ks_rows,
        "true_rcv_info": true_rcv_info
    }


if __name__ == "__main__":
    # macOS multiprocessing safeguards - needed for parallel methods
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    print("compare_ks.py is a library module in the corrected pipeline; "
          "use run_experiments.py to run experiments.")
