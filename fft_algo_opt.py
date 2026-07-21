from collections import defaultdict
from functools import lru_cache
import numpy as np
import random, time

def generate_ballots(candiates, n, priority_order=None, randomness=0):
    ballots = []
    for _ in range(n):
        if priority_order is None:
            priority_order = candiates
            indices = random.sample(range(len(priority_order)), k=random.randint(2, len(candiates)))
            ballot = [priority_order[i] for i in indices]
            priority_order = None
        else:
            indices = random.sample(range(len(priority_order)), k=random.randint(2, len(candiates)))
            ballot = [priority_order[i] for i in sorted(indices)]
        if randomness > 0:
            new_ballot = []
            for c in ballot:
                if random.random() < randomness:
                    cand = random.choice(candiates)
                    while cand in new_ballot:
                        cand = random.choice(candiates)
                    new_ballot.append(cand)
                else:
                    new_ballot.append(c)
            ballot = new_ballot
        ballots.append(ballot)
    return ballots

def rcv_elimination_orders_prob_fft(bound_ballots, base_P, U):
    all_cands = tuple(sorted(set(c for b in bound_ballots for c in b)))
    assert set(all_cands) == set(base_P.keys())

    @lru_cache(maxsize=None)
    def bound_tally_for_subset(S_tuple):
        S = set(S_tuple)
        counts = {c: 0 for c in S_tuple}
        for ballot in bound_ballots:
            for pref in ballot:
                if pref in S:
                    counts[pref] += 1
                    break
        return np.array([counts[c] for c in S_tuple], dtype=int)

    def renorm_q(S_tuple):
        s = sum(base_P[c] for c in S_tuple)
        return np.array([base_P[c] / s for c in S_tuple], dtype=float)

    def feasibility_classify(bound_counts):
        k = len(bound_counts)
        for i in range(k):
            others = np.delete(bound_counts, i)
            if bound_counts[i] + U < others.min(initial=10**9):
                return i, set()
        cannot = set()
        for i in range(k):
            others = np.delete(bound_counts, i)
            if bound_counts[i] > others.min(initial=10**9) + U:
                cannot.add(i)
        return None, cannot

    # --------- FFT multinomial over full grid (no banding) ----------
    def unbound_split_distribution_fft(q, U, k, *, dtype=np.float32, use_memmap=False, tmp_path="/tmp/fftpm"):
        """
        Multinomial pmf over first k-1 coordinates on the full lattice (U+1)^(k-1).
        The kth coordinate is implicit: U - sum_{i<k} n_i.
        """
        if U == 0:
            return np.array(1.0, dtype=dtype).reshape((1,) * (k - 1))
        if k == 1:
            return np.array([1.0], dtype=dtype)

        shape = (U + 1,) * (k - 1)

        # Time-domain one-trial kernel
        if use_memmap:
            K = np.memmap(f"{tmp_path}_K.dat", mode="w+", dtype=dtype, shape=shape)
            K[:] = 0
        else:
            K = np.zeros(shape, dtype=dtype)

        K[(0,) * (k - 1)] += dtype(q[-1])      # assign to implicit last coord
        for ax in range(k - 1):                # unit step along each explicit axis
            if shape[ax] > 1:
                idx = [0] * (k - 1)
                idx[ax] = 1
                K[tuple(idx)] += dtype(q[ax])

        # rFFT
        freq_shape = list(shape)
        if (k - 1) >= 1:
            freq_shape[-1] = freq_shape[-1] // 2 + 1
        freq_shape = tuple(freq_shape)

        if use_memmap:
            F = np.memmap(f"{tmp_path}_F.dat", mode="w+", dtype=np.complex64, shape=freq_shape)
            F_tmp = np.fft.rfftn(K, s=shape)
            F[:] = F_tmp.astype(np.complex64, copy=False)
            del F_tmp
            del K
        else:
            F = np.fft.rfftn(K, s=shape).astype(np.complex64, copy=False)
            del K

        # Exponentiation by squaring, in place: F <- F ** U
        def pow_inplace(A, power):
            B = A.copy()
            A[...] = 1.0 + 0.0j
            p = power
            while p > 0:
                if p & 1:
                    A *= B
                p >>= 1
                if p:
                    B *= B

        pow_inplace(F, U)

        # Back to probabilities
        if use_memmap:
            D = np.memmap(f"{tmp_path}_D.dat", mode="w+", dtype=dtype, shape=shape)
            D_tmp = np.fft.irfftn(F, s=shape).real.astype(dtype, copy=False)
            D[:] = np.clip(D_tmp, 0, None)
            del D_tmp
            del F
        else:
            D = np.fft.irfftn(F, s=shape).real.astype(dtype, copy=False)
            D = np.clip(D, 0, None)
            del F

        s = float(D.sum())
        if s > 0:
            D /= s
        return D

    def first_elimination_distribution_fft(S_tuple):
        k = len(S_tuple)
        bound_counts = bound_tally_for_subset(S_tuple)  # shape (k,)
        q = renorm_q(S_tuple)

        forced_i, cannot_out = feasibility_classify(bound_counts)
        if forced_i is not None:
            out = np.zeros(k)
            out[forced_i] = 1.0
            return out

        if U == 0:
            totals = bound_counts
            minv = totals.min()
            mins = np.where(totals == minv)[0]
            out = np.zeros(k)
            out[mins] = 1.0 / len(mins)
            return out

        D = unbound_split_distribution_fft(q, U, k)
        axes = [np.arange(U + 1)] * (k - 1)
        grids = np.meshgrid(*axes, indexing='ij') if (k - 1) >= 1 else []
        if k - 1 >= 1:
            sum_first = np.zeros_like(D)
            for g in grids:
                sum_first += g
        else:
            sum_first = np.array(0)
        valid_mask = (sum_first <= U)
        if not np.all(valid_mask):
            D = np.where(valid_mask, D, 0.0)

        elim_probs = np.zeros(k)
        totals_first = [ (bound_counts[i] + grids[i]).astype(int) for i in range(k - 1) ]
        totals_last  = (bound_counts[-1] + (U - sum_first)).astype(int)

        stacked = np.stack(totals_first + [totals_last], axis=-1)   # (..., k)
        min_vals = stacked.min(axis=-1)
        for i in range(k):
            at_min = (stacked[..., i] == min_vals)
            share = np.where(at_min, 1.0, 0.0)
            denom = share.copy()
            for j in range(k):
                if j == i:
                    continue
                denom += (stacked[..., j] == min_vals).astype(float)
            denom = np.where(denom > 0, denom, 1.0)
            contrib = D * (share / denom)
            if i in cannot_out:
                contrib = 0.0
            if type(contrib) == float:
                elim_probs[i] += contrib
            else:
                elim_probs[i] += contrib.sum()

        s = elim_probs.sum()
        if s > 0:
            elim_probs /= s
        return elim_probs

    memo = {}

    def recurse(S_tuple):
        # Base case now returns the LAST remaining candidate
        if len(S_tuple) == 1:
            return { (S_tuple[0],): 1.0 }
        if S_tuple in memo:
            return memo[S_tuple]

        p_out = first_elimination_distribution_fft(S_tuple)
        out = defaultdict(float)
        for i, p in enumerate(p_out):
            if p <= 0:
                continue
            child = S_tuple[:i] + S_tuple[i+1:]
            # child_dist already ends with the final survivor
            child_dist = recurse(child)
            c_elim = S_tuple[i]
            for suffix, pc in child_dist.items():
                # prepend this round’s eliminated candidate; suffix ends with the final winner
                out[(c_elim,) + suffix] += p * pc
        memo[S_tuple] = dict(out)
        return memo[S_tuple]

    return recurse(all_cands)

if __name__ == "__main__":
    from itertools import permutations
    candidates = ["A","B","C"]
    n = 100
    # bound1 = int(n/5)*[list(c) for c in permutations(candidates) if c[0] not in ("A")]
    # bound2 = 3*[list(c) for c in permutations(candidates) if c[0] in ("E")]
    # bound3 = 3*[list(c) for c in permutations(candidates) if c[0] in ("C", "D")]
    # bound = bound1 + bound2 + bound3

    bound1 = 28*[list(c) for c in permutations(candidates) if c[0] in ("A")]
    bound2 = 25*[list(c) for c in permutations(candidates) if c[0] in ("B")]
    bound3 = 32*[list(c) for c in permutations(candidates) if c[0] in ("C")]
    # bound4 = 32*[list(c) for c in permutations(candidates) if c[0] in ("D")]
    # bound5 = 20*[list(c) for c in permutations(candidates) if c[0] in ("E")]
    # bound6 = 31*[list(c) for c in permutations(candidates) if c[0] in ("F")]
    # bound7 = 700*[list(c) for c in permutations(candidates) if c[0] in ("G")]
    # bound8 = 800*[list(c) for c in permutations(candidates) if c[0] in ("H")]
    # bound9 = 900*[list(c) for c in permutations(candidates) if c[0] in ("I")]
    # bound10 = 1000*[list(c) for c in permutations(candidates) if c[0] in ("J")]
    bound = bound1 + bound2 + bound3 #+ bound4 + bound5 + bound6

    # P = {"A": 1/5, "B": 1/5, "C": 1/5, "D": 1/5, "E": 1/5}
    # P = {"A": 0.35, "B": 0.30, "C": 0.20, "D": 0.10, "E": 0.05}
    # P = {"A": 0.20, "B": 0.18, "C": 0.15, "D": 0.12, "E": 0.10,
    #      "F": 0.08, "G": 0.07, "H": 0.06, "I": 0.04, "J": 0.03}
    # P = {"A": 0.1, "B": 0.1, "C": 0.1, "D": 0.1, "E": 0.1,
    #      "F": 0.1, "G": 0.1, "H": 0.1, "I": 0.1, "J": 0.1}

    # P = {'A': 8, 'B': 5, 'C': 8, 'D': 4, 'E': 2, 'F': 0.5}
    # P = {'A': 1/5, 'B': 1/3, 'C': 1/6, 'D': 1/7, 'E': 1/8, 'F': 1/6}
    P = {'A': 0.4, 'B': 0.35, 'C': 0.25}
    s = sum(P.values())
    P = {k: v/s for k,v in P.items()}
    U = 10000

    t = time.time()
    dist = rcv_elimination_orders_prob_fft(bound, P, U)
    print("Time:", time.time() - t)
    total = 0.0
    for order in sorted(dist):
        print(f"{order}: {dist[order]:.8f}")
        total += dist[order]
    print("Sum:", total)

    # Winner marginals from full orders (last item is the winner)
    winners = defaultdict(float)
    for order, p in dist.items():
        winners[order[-1]] += p
    print("Winners:")
    for w in sorted(winners):
        print(f"  {w}: {winners[w]:.8f}")
