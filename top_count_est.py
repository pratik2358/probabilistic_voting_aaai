# """
# Top-count last-winner estimator.

# Estimates the elimination-order distribution using only the first-choice
# counts from bound ballots.  For each candidate c, the probability that c
# wins equals c's first-choice share among bound ballots.  The prefix (the
# elimination order of non-winners) follows the same top-count ranking:
# candidates with fewer first-choice votes are eliminated first.

# Formally, returns k full orders — one per possible winner w:
#     order = (candidates sorted ascending by top-count, except w) + (w,)
#     P(order) = top_prob[w]

# This runs in O(k log k) instead of O(k!), which matters for large elections.
# The method uses only bound_ballots and candidates; it ignores U and P.
# """

# from itertools import permutations
# from collections import Counter
# from math import factorial
# from typing import Dict, Hashable, Iterable, Sequence, Tuple


# Candidate = Hashable
# Ballot = Sequence[Candidate]
# Order = Tuple[Candidate, ...]


# def top_count_last_winner_estimation(
#     bound_ballots: Iterable[Ballot],
#     candidates: Sequence[Candidate],
# ) -> Dict[Order, float]:
#     """
#     Estimate the elimination-order distribution from top-choice counts.

#     Args:
#         bound_ballots: Iterable of ballots (each ballot is a sequence of
#                        candidates in preference order, possibly partial).
#         candidates:    Complete list of candidates in the race.

#     Returns:
#         Dict mapping k complete elimination orders (one per possible winner)
#         to estimated probabilities.  All probabilities sum to 1.
#     """
#     candidates = list(candidates)
#     k = len(candidates)

#     if k == 0:
#         raise ValueError("candidates must be non-empty")

#     top_counts: Counter[Candidate] = Counter()
#     total_bound = 0

#     for ballot in bound_ballots:
#         if len(ballot) == 0:
#             continue
#         top_candidate = ballot[0]
#         if top_candidate not in candidates:
#             # Skip ballots whose first choice isn't a recognized candidate
#             # (e.g. write-ins, overvotes, or name-format mismatches between
#             #  counting groups).  Treat them the same as blank ballots.
#             continue
#         top_counts[top_candidate] += 1
#         total_bound += 1

#     if total_bound == 0:
#         raise ValueError("No non-empty bound ballots provided")

#     top_probs = {c: top_counts[c] / total_bound for c in candidates}
#     # print(top_probs)

#     # Canonical elimination order: eliminate lowest-top-count candidates first.
#     # One full order per possible winner — O(k log k) instead of O(k!).
#     # Winner probability = top_probs[winner]; prefix is deterministic.

#     denom = factorial(k - 1)

#     order_probs: Dict[Order, float] = {}

#     for winner in candidates:
#         remaining = [c for c in candidates if c != winner]

#         for prefix in permutations(remaining):
#             order = tuple(prefix) + (winner,)
#             order_probs[order] = top_probs[winner] / denom


#     return order_probs

# # bound_ballots=[['A','B','C','D'],['A','B','C','D'],['B','A','C','D']]
# # print(sum(top_count_last_winner_estimation(bound_ballots, ['A','B','C','D']).values()))
# """
# Top-count last-winner estimator.

# Estimates the elimination-order distribution using only the first-choice
# counts from bound ballots.  For each candidate c, the probability that c
# wins equals c's first-choice share among bound ballots.  The prefix (the
# elimination order of non-winners) follows the same top-count ranking:
# candidates with fewer first-choice votes are eliminated first.

# Formally, returns k full orders — one per possible winner w:
#     order = (candidates sorted ascending by top-count, except w) + (w,)
#     P(order) = top_prob[w]

# This runs in O(k log k) instead of O(k!), which matters for large elections.
# The method uses only bound_ballots and candidates; it ignores U and P.
# """

# from itertools import permutations
# from collections import Counter
# from math import factorial
# from typing import Dict, Hashable, Iterable, Sequence, Tuple


# Candidate = Hashable
# Ballot = Sequence[Candidate]
# Order = Tuple[Candidate, ...]


# def top_count_last_winner_estimation(
#     bound_ballots: Iterable[Ballot],
#     candidates: Sequence[Candidate],
# ) -> Dict[Order, float]:
#     """
#     Estimate the elimination-order distribution from top-choice counts.

#     Args:
#         bound_ballots: Iterable of ballots (each ballot is a sequence of
#                        candidates in preference order, possibly partial).
#         candidates:    Complete list of candidates in the race.

#     Returns:
#         Dict mapping k complete elimination orders (one per possible winner)
#         to estimated probabilities.  All probabilities sum to 1.
#     """
#     candidates = list(candidates)
#     k = len(candidates)

#     if k == 0:
#         raise ValueError("candidates must be non-empty")

#     top_counts: Counter[Candidate] = Counter()
#     total_bound = 0

#     top_cands = [b[0] for b in bound_ballots if len(b) > 0 and b[0] in candidates]
#     top_counts.update(top_cands)
#     total_bound = len(top_cands)

#     if total_bound == 0:
#         raise ValueError("No non-empty bound ballots provided")

#     top_probs = {c: top_counts[c] / total_bound for c in candidates}
#     # print(top_probs)

#     # Canonical elimination order: eliminate lowest-top-count candidates first.
#     # One full order per possible winner — O(k log k) instead of O(k!).
#     # Winner probability = top_probs[winner]; prefix is deterministic.

#     denom = factorial(k - 1)

#     order_probs: Dict[Order, float] = {}

#     for winner in candidates:
#         remaining = [c for c in candidates if c != winner]

#         for prefix in permutations(remaining):
#             order = tuple(prefix) + (winner,)
#             order_probs[order] = top_probs[winner] / denom


#     return order_probs

"""
Top-count last-winner estimator.

Estimates the elimination-order distribution using only the first-choice
counts from bound ballots.  For each candidate w, the probability that w
wins equals w's first-choice share among bound ballots; conditional on the
winner, the elimination order of the other k-1 candidates is uniform.

Formally, enumerates all k! full orders:
    P(prefix + (w,)) = top_prob[w] / (k-1)!   for every prefix of the others

Materializing k! orders is only feasible for small fields (roughly k <= 9);
TopCount-S is the sampled equivalent for larger k.
The method uses only bound_ballots and candidates; it ignores U and P.
"""

from itertools import permutations
from collections import Counter
from math import factorial
from typing import Dict, Hashable, Iterable, Sequence, Tuple


Candidate = Hashable
Ballot = Sequence[Candidate]
Order = Tuple[Candidate, ...]


def top_count_last_winner_estimation(
    bound_ballots: Iterable[Ballot],
    candidates: Sequence[Candidate],
) -> Dict[Order, float]:
    """
    Estimate the elimination-order distribution from top-choice counts.

    Args:
        bound_ballots: Iterable of ballots (each ballot is a sequence of
                       candidates in preference order, possibly partial).
        candidates:    Complete list of candidates in the race.

    Returns:
        Dict mapping all k! complete elimination orders to estimated
        probabilities.  All probabilities sum to 1.
    """
    candidates = list(candidates)
    cand_set = set(candidates)
    k = len(candidates)

    if k == 0:
        raise ValueError("candidates must be non-empty")

    top_counts: Counter[Candidate] = Counter()

    top_cands = [b[0] for b in bound_ballots if len(b) > 0 and b[0] in cand_set]
    top_counts.update(top_cands)
    total_bound = len(top_cands)

    if total_bound == 0:
        raise ValueError("No non-empty bound ballots provided")

    top_probs = {c: top_counts[c] / total_bound for c in candidates}

    # Winner probability = top_probs[winner]; conditional on the winner the
    # prefix is uniform over the (k-1)! arrangements of the other candidates.

    denom = factorial(k - 1)

    order_probs: Dict[Order, float] = {}

    for winner in candidates:
        remaining = [c for c in candidates if c != winner]

        for prefix in permutations(remaining):
            order = tuple(prefix) + (winner,)
            order_probs[order] = top_probs[winner] / denom

    return order_probs
# bound_ballots=5000*[['A','B','C','D','E','F','G','H','I','J','K','L'],['G','H','I','J','K','L'],['A','B','C','D','E','F','G'], ['C','A','B','D','E','F','G'],['D','A','B','C','E','F','G'],['F','A','B','C','D','E','G'], ['G','A','B','C','D','E','F'],['B','A','C','D','E','F','G']]
# print(sum(top_count_last_winner_estimation(bound_ballots, ['A','B','C','D','E','F','G','H','I','J','K','L']).values()))
