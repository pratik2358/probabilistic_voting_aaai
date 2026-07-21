"""Mult -- exact per-order probabilities under the paper's MEMORYLESS model.

At every survivor set S_t the U unbound ballots are redrawn fresh as
Multinomial(U, q(S_t)) with q renormalized over S_t, so for a fixed order pi
the survivor sets are deterministic (S_t = pi[t:]) and

    P(pi) = prod_t P( pi[t] is the fair-tie-broken minimum of
                      bound_tally(S_t) + Multinomial(U, q(S_t)) ),

each round computed by exact enumeration of compositions. This is exactly the
model ExhaustDP (exact.py) and FFTProb compute, so the three exact methods
agree by construction (HD18: 0.148418769, matching ExhaustDP to 9 digits).

This REPLACES the previous coupled cascading-transfer implementation (one
initial Multinomial(U, p) realization tracked forward, the eliminee's unbound
votes redistributed to survivors each round). That process has the same
per-round marginals but correlates rounds through the shared realization --
a different model from the paper's definition, off by 1-3 percentage points
per order on uncertain cases. It also required a water-filling feasibility
prune that harbored the Alaska HD18 stale-tally bug; no transfer step exists
here, so that bug class is structurally impossible.

approx semantics: an order is zeroed if any of its round probabilities falls
below `approx`; every order that survives is computed EXACTLY (no
per-composition pruning, total mass 1 up to the dropped negligible orders).
"""
import math
import os
from typing import List, Dict, Tuple, Optional
from functools import partial
from itertools import permutations
from time import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from tqdm import tqdm


def multinomial_tuples(candidates: List[str], m: int):
    """
    Yield all dicts {c: x_c} such that sum x_c = m and x_c >= 0.
    """
    n = len(candidates)
    if n == 0:
        if m == 0:
            yield {}
        return

    def helper(idx: int, rem: int, acc: Dict[str, int]):
        if idx == n - 1:
            acc[candidates[idx]] = rem
            yield dict(acc)
            return
        for x in range(rem + 1):
            acc[candidates[idx]] = x
            yield from helper(idx + 1, rem - x, acc)
        acc.pop(candidates[idx], None)

    yield from helper(0, m, {})


def compute_bound_tally(active: List[str], bound_ballots: List[List[str]]) -> Dict[str, int]:
    """
    First-available-preference tally of bound ballots restricted to the active set.
    """
    active_set = set(active)
    tally = {c: 0 for c in active}
    for ranking in bound_ballots:
        for pref in ranking:
            if pref in active_set:
                tally[pref] += 1
                break
    return tally


def _log_multinomial_pmf(counts: List[int], probs: List[float]) -> float:
    """
    Stable log PMF for Multinomial(n; probs).
    - Returns -inf if any count>0 has prob==0 (impossible),
      or if total prob == 0 with n>0.
    - Renormalizes probs to their sum to be safe.
    """
    n = sum(counts)
    s = sum(probs)
    if s <= 0.0:
        return 0.0 if n == 0 else float("-inf")
    probs = [p / s for p in probs]
    for k, p in zip(counts, probs):
        if k > 0 and p <= 0.0:
            return float("-inf")
    logcoef = math.lgamma(n + 1.0) - sum(math.lgamma(k + 1.0) for k in counts)
    logp = sum((k * math.log(p)) for k, p in zip(counts, probs) if k > 0 and p > 0.0)
    return logcoef + logp


def order_probability(
    pi: List[str],
    bound_ballots: List[List[str]],
    p_dict: Dict[str, float],
    m: int,
    approx: float = 0.0
) -> float:
    """
    P(elimination order == pi) under the memoryless model: unbound tallies
    u(S) ~ Multinomial(m, q(S)) are redrawn independently at every survivor
    set S. With pi fixed, S_t = pi[t:] deterministically, so P(pi) is the
    product over rounds of the probability that pi[t] is the fair-tie-broken
    minimum, each from its own fresh multinomial.
    """
    k = len(pi)
    total_log_prob = 0.0

    for t in range(k - 1):
        survivors = pi[t:]
        c_t = pi[t]

        bound_tally = compute_bound_tally(survivors, bound_ballots)
        raw_ps = [p_dict.get(c, 0.0) for c in survivors]
        den = sum(raw_ps)
        if den <= 0.0:
            return 0.0
        norm_p = [p / den for p in raw_ps]

        round_prob = 0.0
        for counts in multinomial_tuples(survivors, m):
            T = {c: bound_tally[c] + counts[c] for c in survivors}
            min_votes = min(T.values())
            if T[c_t] != min_votes:
                continue
            tied = [c for c in survivors if T[c] == min_votes]

            logp = _log_multinomial_pmf([counts[c] for c in survivors], norm_p)
            if logp == float("-inf"):
                continue
            round_prob += math.exp(logp) / len(tied)

        if round_prob <= 0.0:
            return 0.0
        if approx > 0.0 and round_prob < approx:
            return 0.0

        total_log_prob += math.log(round_prob)

    return math.exp(total_log_prob)


# print PID once per worker
def _init_worker():
    print(f"[worker-start] PID={os.getpid()}", flush=True)


# return PID along with result
def _order_task(pi_tuple: Tuple[str, ...],
                bound_ballots: List[List[str]],
                p_dict: Dict[str, float],
                m: int,
                approx: float) -> Tuple[Tuple[str, ...], float, int]:
    pid = os.getpid()
    prob = order_probability(list(pi_tuple), bound_ballots, p_dict, m, approx)
    return pi_tuple, prob, pid


# collect & show PIDs
def run_parallel_over_orders(candiates: List[str],
                             bound_ballots: List[List[str]],
                             p_dict: Dict[str, float],
                             m: int,
                             approx: float = 0.0,
                             n_jobs: Optional[int] = None) -> Dict[Tuple[str, ...], float]:
    if n_jobs is None or n_jobs <= 0:
        n_jobs = mp.cpu_count()

    orders = list(permutations(candiates))
    results: Dict[Tuple[str, ...], float] = {}
    pids: set[int] = set()

    task = partial(_order_task,
                   bound_ballots=bound_ballots,
                   p_dict=p_dict,
                   m=m,
                   approx=approx)

    with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_worker) as ex:
        futures = [ex.submit(task, pi) for pi in orders]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Computing orders (parallel)"):
            pi_tuple, prob, pid = f.result()
            results[pi_tuple] = prob
            pids.add(pid)

    print(f"\nWorker PIDs used: {sorted(pids)}")
    return results


if __name__ == "__main__":
    candidates = ["A", "B", "C"]

    bound1 = 28*[list(c) for c in permutations(candidates) if c[0] in ("A")]
    bound2 = 25*[list(c) for c in permutations(candidates) if c[0] in ("B")]
    bound3 = 32*[list(c) for c in permutations(candidates) if c[0] in ("C")]
    bound_ballots = bound1 + bound2 + bound3

    p_dict = {'A': 0.4, 'B': 0.35, 'C': 0.25}
    s = sum(p_dict.values())
    p_dict = {k: v/s for k, v in p_dict.items()}
    m = 2000

    t0 = time()
    results = run_parallel_over_orders(
        candiates=candidates,
        bound_ballots=bound_ballots,
        p_dict=p_dict,
        m=m,
        approx=1e-8,
        n_jobs=None
    )
    elapsed = time() - t0

    results = sorted(results.items(), key=lambda kv: kv[1], reverse=True)
    total = sum([v for k, v in results])
    print(f"\nTime taken (parallel): {elapsed:.2f} seconds")
    print("Order probabilities:")
    for pi, value in results:
        print(f"{pi}: {value:.12f}")
    print(f"Sum over all orders: {total:.12f}")
