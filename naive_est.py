"""NaiveEst baseline -- reversed-suffix scheme.

A ballot (favorite first) is REVERSED and matched as a SUFFIX of the
elimination order: the voter's ranked candidates outlast the unranked ones
and are eliminated in reverse preference order, with the favorite surviving
longest. Order weights are PROPORTIONAL to accumulated ballot support,
w(o) = eps + v(o), normalized.

This replaces the original scheme (forward prefix match with inverse
weights 1/(eps + v)), which produced a near-uniform distribution over
unsupported orders. On HD18 / NYC-D13 CON the suffix scheme gives lower TVD
to the exact memoryless distribution (0.677 vs 0.737 and 0.734 vs 0.770).
A reversed-PREFIX match would be wrong for partial ballots: a bullet ballot
[X] would support orders eliminating X first.
"""
from itertools import permutations


def naive_estimation(bound_ballots, candidates, eps=1e-6):
    orders = [list(o) for o in permutations(candidates)]
    order_vals = {tuple(o): 0.0 for o in orders}
    for b in bound_ballots:
        rb = list(reversed(b))
        # reversed ballot as SUFFIX of the elimination order: the voter's
        # ranked candidates are eliminated last, in reverse preference order
        compat = [o for o in orders if rb == o[len(o) - len(rb):]]
        if compat:
            share = 1.0 / len(compat)
            for o in compat:
                order_vals[tuple(o)] += share
    order_vals = {o: eps + v for o, v in order_vals.items()}
    total = sum(order_vals.values())
    return {o: v / total for o, v in order_vals.items()}
