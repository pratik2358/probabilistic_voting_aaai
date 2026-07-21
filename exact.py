"""ExhaustDP -- exact elimination-order distribution under the MEMORYLESS model.

CORRECTED IMPLEMENTATION. The previous exact.py materialized each composition
of the U unbound ballots as first-choice-only ("bullet") ballots and simulated
a whole election per composition. Bullet ballots EXHAUST when their candidate
is eliminated, which silently implements a different model than the paper
defines (unbound votes redrawn fresh over the survivors at every round). The
two models coincide whenever every round is decided by a wide margin, but on
knife-edge races with asymmetric transfers they diverge (Alaska HD18 2022:
0.309 exhaust vs 0.148 memoryless winner probability; 0.148418769 is the
independently verified ground truth under the paper's model).

This version implements the paper's Algorithm 1/2 directly:
  - at every survivor set S the U unbound ballots are drawn fresh as
    Multinomial(U, q(S)) with the base prior q renormalized over S;
  - p_min(x | S) is computed by exact enumeration of all compositions of U
    over S with fair (uniform) tie-splitting, and memoized per survivor set;
  - full-order probabilities are products of stage probabilities along a
    prefix DP over survivor sets.

Public API is unchanged so compare_ks.py and all runners work as before:
  exact_rcv_order_distribution_fair(bound_ballots, U, p_dict)
  exact_rcv_order_distribution_fair_parallel(bound_ballots, U, p_dict,
                                             n_jobs=None, batch_size=500)
  enumerate_all_orders_with_weights(ballots, candidates)   # deterministic IRV
The parallel variant shards the composition enumeration of large survivor
sets across a process pool (streamed with a bounded pending window, so memory
stays flat); small sets are computed serially in-process.
"""
from __future__ import annotations
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from functools import lru_cache
import math
import multiprocessing as mp
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


def _lgamma(x: float) -> float:
    return math.lgamma(x)


def multinomial_pmf(counts: List[int], probs: List[float]) -> float:
    """Numerically-stable multinomial pmf using log-gamma.
    pmf = U! / (prod c_i!) * prod p_i^{c_i}
    """
    U = sum(counts)
    if U == 0:
        return 1.0
    if any(c < 0 for c in counts):
        return 0.0
    if len(counts) != len(probs):
        raise ValueError("counts and probs length mismatch")
    if not (0.999999 <= sum(probs) <= 1.000001):
        s = sum(probs)
        if s == 0:
            return 0.0
        probs = [max(0.0, p) / s for p in probs]
    log_num = _lgamma(U + 1)
    log_den = sum(_lgamma(c + 1) for c in counts)
    log_p = sum((c * math.log(p) if c > 0 and p > 0 else (-math.inf if c > 0 else 0.0))
                for c, p in zip(counts, probs))
    val = log_num - log_den + log_p
    if val == -math.inf:
        return 0.0
    return math.exp(val)


def compositions(n: int, k: int) -> Iterator[List[int]]:
    """Yield weak compositions of n into k parts (stars and bars)."""
    if k <= 0:
        return
    if k == 1:
        yield [n]
        return
    def rec(remaining: int, parts_left: int, prefix: List[int]):
        if parts_left == 1:
            yield prefix + [remaining]
            return
        for x in range(remaining + 1):
            yield from rec(remaining - x, parts_left - 1, prefix + [x])
    yield from rec(n, k, [])


def _chunked(iterable: Iterable[List[int]], size: int) -> Iterator[List[List[int]]]:
    buf: List[List[int]] = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


# ---------------------------------------------------------------------------
# Deterministic IRV on a FIXED ballot multiset with fair tie enumeration.
# Used by compare_ks.py to compute the true RCV order from the real ballots.
# Model-independent; carried over unchanged from the previous exact.py.
# ---------------------------------------------------------------------------

def _tally_first_choices(ballots: List[List[str]], active: Tuple[str, ...]) -> Dict[str, int]:
    """Compute first-choice tally restricted to active candidates."""
    active_set = set(active)
    tally = {c: 0 for c in active}
    for b in ballots:
        for pref in b:
            if pref in active_set:
                tally[pref] += 1
                break
    return tally


def _remove_candidate_from_ballots(ballots: List[List[str]], cand: str) -> List[List[str]]:
    out = []
    for b in ballots:
        if cand in b:
            out_b = [x for x in b if x != cand]
            if out_b:
                out.append(out_b)
        else:
            out.append(b)
    return out


@lru_cache(maxsize=None)
def _enumerate_orders_cached(active: Tuple[str, ...],
                            ballots_key: Tuple[Tuple[str, ...], ...]) -> Dict[Tuple[str, ...], float]:
    """Pure functional recursion with memoization.
    ballots_key encodes ballots as a tuple of tuples for cacheability.
    Returns a dict mapping full elimination order (loser->...->winner) to weight.
    """
    active_list = list(active)
    if len(active_list) == 1:
        return {tuple(): 1.0}

    ballots = [list(b) for b in ballots_key]
    tally = _tally_first_choices(ballots, active)
    min_votes = min(tally.values())
    losers = [c for c, v in tally.items() if v == min_votes]

    out: Dict[Tuple[str, ...], float] = defaultdict(float)
    L = len(losers)
    prob_each = 1.0 / L

    for a in losers:
        new_active = tuple(x for x in active if x != a)
        new_ballots = _remove_candidate_from_ballots(ballots, a)
        new_key = tuple(tuple(b) for b in new_ballots)
        sub = _enumerate_orders_cached(new_active, new_key)
        for tail, w in sub.items():
            out[(a,) + tail] += prob_each * w

    return dict(out)


def enumerate_all_orders_with_weights(ballots: List[List[str]],
                                      candidates: List[str]) -> Dict[Tuple[str, ...], float]:
    """Enumerate all elimination orders (loser -> ... -> winner) with fair tie weights
    for a FIXED ballot multiset (no probabilistic unbound votes).
    """
    active = tuple(candidates)
    ballots_key = tuple(tuple(b) for b in ballots)
    _enumerate_orders_cached.cache_clear()  # ensure per-call purity wrt different ballots
    return _enumerate_orders_cached(active, ballots_key)


# ---------------------------------------------------------------------------
# Memoryless-model DP (paper Algorithm 1/2)
# ---------------------------------------------------------------------------

# Below this composition count a survivor set's p_min is computed serially
# in-process; above it, the enumeration is sharded across the worker pool.
_PARALLEL_MIN_COMPOSITIONS = 20_000


def _pmin_batch_task(batch: Iterable[List[int]],
                     b_counts: List[int],
                     q: List[float]) -> Tuple[List[float], float]:
    """Partial fair-tie elimination shares for a batch of compositions.

    Returns (p_partial, weight_sum) where p_partial[i] accumulates
    pmf(u) / |argmin| over compositions u in the batch for each candidate i
    attaining the minimum total b_counts[i] + u[i].
    """
    s = len(b_counts)
    p = [0.0] * s
    wsum = 0.0
    for u in batch:
        w = multinomial_pmf(u, q)
        if w <= 0.0:
            continue
        wsum += w
        totals = [b_counts[i] + u[i] for i in range(s)]
        m = min(totals)
        ties = [i for i in range(s) if totals[i] == m]
        share = w / len(ties)
        for i in ties:
            p[i] += share
    return p, wsum


def _run_dp(bound_ballots: List[List[str]],
            U: int,
            p_dict: Dict[str, float],
            executor: Optional[ProcessPoolExecutor] = None,
            n_jobs: int = 1,
            batch_size: int = 500) -> Dict[Tuple[str, ...], float]:
    cands = tuple(p_dict.keys())

    def bound_tallies(S: Tuple[str, ...]) -> Dict[str, int]:
        Sset = set(S)
        t = {c: 0 for c in S}
        for ballot in bound_ballots:
            for choice in ballot:
                if choice in Sset:
                    t[choice] += 1
                    break
        return t

    pmin_memo: Dict[Tuple[str, ...], Dict[str, float]] = {}

    def pmin(S: Tuple[str, ...]) -> Dict[str, float]:
        """P(x is eliminated first | survivor set S) under fresh
        Multinomial(U, q(S)) unbound tallies with fair tie-splitting."""
        if S in pmin_memo:
            return pmin_memo[S]
        s = len(S)
        b = bound_tallies(S)
        qsum = sum(p_dict.get(c, 0.0) for c in S)
        q = ([p_dict.get(c, 0.0) / qsum for c in S] if qsum > 0
             else [1.0 / s] * s)

        # Sure-loser shortcut: x trails some rival by more than U, so x is
        # the unique minimum for every composition.
        for x in S:
            if b[x] + U < min(b[j] for j in S if j != x):
                out = {c: (1.0 if c == x else 0.0) for c in S}
                pmin_memo[S] = out
                return out

        b_counts = [b[c] for c in S]
        n_comp = math.comb(U + s - 1, s - 1)

        if executor is not None and n_jobs > 1 and n_comp >= _PARALLEL_MIN_COMPOSITIONS:
            # Shard the enumeration; stream batches with a bounded pending
            # window so composition lists are never all materialized at once.
            p = [0.0] * s
            wsum = 0.0
            pending = set()
            max_pending = max(2 * n_jobs, 2)

            def drain(done_futs):
                nonlocal wsum
                for f in done_futs:
                    pp, ww = f.result()
                    wsum += ww
                    for i in range(s):
                        p[i] += pp[i]

            for batch in _chunked(compositions(U, s), batch_size):
                pending.add(executor.submit(_pmin_batch_task, batch, b_counts, q))
                if len(pending) >= max_pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    drain(done)
            drain(as_completed(pending))
        else:
            p, wsum = _pmin_batch_task(compositions(U, s), b_counts, q)

        out = ({c: p[i] / wsum for i, c in enumerate(S)} if wsum > 0
               else {c: 1.0 / s for c in S})
        pmin_memo[S] = out
        return out

    dist: Dict[Tuple[str, ...], float] = {}

    def recurse(prefix: Tuple[str, ...], S: Tuple[str, ...], mass: float):
        if mass <= 0.0:
            return
        if len(S) == 1:
            order = prefix + (S[0],)
            dist[order] = dist.get(order, 0.0) + mass
            return
        p = pmin(S)
        for x in S:
            px = p.get(x, 0.0)
            if px > 0.0:
                child = tuple(c for c in S if c != x)
                recurse(prefix + (x,), child, mass * px)

    recurse((), cands, 1.0)

    # Stage distributions each sum to 1, so total mass is 1 by construction;
    # normalize anyway to defend against floating-point drift (API parity
    # with the previous implementation).
    total = sum(dist.values())
    if total > 0:
        for o in list(dist.keys()):
            dist[o] /= total
    return dist


def exact_rcv_order_distribution_fair(bound_ballots: List[List[str]],
                                      U: int,
                                      p_dict: Dict[str, float]) -> Dict[Tuple[str, ...], float]:
    """Serial exact distribution under the memoryless model with fair ties.

    Parameters
    ----------
    bound_ballots : list of ranked ballots (each a list[str])
    U             : number of unbound ballots redrawn fresh per survivor set
    p_dict        : base prior over candidates (sum 1)
    """
    return _run_dp(bound_ballots, U, p_dict)


def exact_rcv_order_distribution_fair_parallel(bound_ballots: List[List[str]],
                                               U: int,
                                               p_dict: Dict[str, float],
                                               n_jobs: Optional[int] = None,
                                               batch_size: int = 500) -> Dict[Tuple[str, ...], float]:
    """Parallel exact distribution under the memoryless model.

    n_jobs    : number of processes (defaults to mp.cpu_count())
    batch_size: compositions per task; tune to amortize IPC vs. load balance
    """
    if n_jobs is None or n_jobs <= 0:
        n_jobs = mp.cpu_count()
    if n_jobs == 1:
        return _run_dp(bound_ballots, U, p_dict, batch_size=batch_size)
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        return _run_dp(bound_ballots, U, p_dict,
                       executor=ex, n_jobs=n_jobs, batch_size=batch_size)


if __name__ == "__main__":
    import time, argparse

    parser = argparse.ArgumentParser(description="Exact RCV order distribution (memoryless model, fair ties)")
    parser.add_argument("--U", type=int, default=100)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--jobs", type=int, default=0)
    args = parser.parse_args()

    C = ["A", "B", "C"]
    p = {c: 1/3 for c in C}
    bound = [["A", "B", "C"],
             ["B", "C", "A"],
             ["C", "A", "B"]]
    t = time.time()
    if args.parallel:
        dist = exact_rcv_order_distribution_fair_parallel(bound, args.U, p, n_jobs=args.jobs)
    else:
        dist = exact_rcv_order_distribution_fair(bound, args.U, p)
    print("Computed in %.2f seconds" % (time.time() - t))
    print(dist)
    print("Sum of probabilities:", sum(dist.values()))
