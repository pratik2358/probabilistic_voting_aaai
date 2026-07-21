from collections import defaultdict
from functools import lru_cache
from typing import List
import random, time
import numpy as np
from itertools import permutations

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import colors as mcolors
import matplotlib
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 20}

matplotlib.rc('font', **font)

from math import exp
from scipy.stats import multivariate_normal as MVN
from scipy.stats import norm as NORM
from scipy.stats import spearmanr, kendalltau
from tqdm import tqdm

# Utilities
def rank_corr(order1, order2):
    """
    order1, order2: lists like ['A','B','C','D'] representing rankings (best to worst)
    Returns: (spearman_rho, kendall_tau) computed on the common items
    """
    order1 = order1[::-1]
    order2 = order2[::-1]

    r1 = {item: i for i, item in enumerate(order1, start=1)}
    r2 = {item: i for i, item in enumerate(order2, start=1)}

    items = [x for x in order1 if x in r2]
    if len(items) < 2:
        raise ValueError("Need at least two overlapping items to compute correlation.")

    x = [r1[i] for i in items]
    y = [r2[i] for i in items]

    rho = spearmanr(x, y).statistic
    tau = kendalltau(x, y).statistic
    return rho, tau


def generate_ballots(candidates: List[str], n: int,
                     priority_order: List[str] = None,
                     randomness: float = 0) -> List[List[str]]:
    ballots = []
    for _ in range(n):
        base_order = priority_order if priority_order is not None else candidates
        k = random.randint(2, len(candidates))
        indices = random.sample(range(len(base_order)), k=k)
        ballot = [base_order[i] for i in sorted(indices)]

        if randomness == 1.0:
            k = random.randint(2, len(candidates))
            ballot = random.sample(candidates, k)
        elif randomness > 0:
            new_ballot = []
            for c in ballot:
                if random.random() < randomness:
                    choices = [cand for cand in candidates if cand not in new_ballot]
                    new_ballot.append(random.choice(choices))
                else:
                    new_ballot.append(c)
            ballot = new_ballot

        ballots.append(ballot)
    return ballots


def pad_cdf_orders(cdf_dist, all_candidates):
    """
    Converts partial CDF orders to full orders by appending the missing candidate.
    """
    full_dist = {}
    all_set = set(all_candidates)
    for order, p in cdf_dist.items():
        missing = all_set - set(order)
        assert len(missing) == 1, f"Expected 1 missing candidate, got {missing}"
        full_order = order + tuple(missing)
        full_dist[full_order] = p
    return full_dist


# MVN orthant RCV (fixed)
def rcv_elimination_orders_prob_mvn(
    bound_ballots,
    base_P,
    U,
    continuity_correction=True,
    is_samples=2000,
    return_tree=False
):
    """
    MVN-based elimination order distribution.
    Fixes:
      - No Bonferroni truncated inclusion-exclusion fallback.
      - Optional importance sampling used only if MVN CDF underflows/errors.
      - No blind renormalization after mixing estimates.
    If return_tree=True, also return edges: { (parent_tuple, child_tuple): branch_prob }.
    """
    all_cands = tuple(sorted(base_P.keys()))
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
        if s <= 0:
            # fallback to uniform if degenerate
            return np.ones(len(S_tuple), dtype=float) / len(S_tuple)
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

    def first_elimination_distribution_mvn(
        S_tuple,
        mvn_floor=1e-12,
        use_importance_sampling=True,   # optional rare-event IS replacement (not combined)
        is_samples=is_samples,
        is_tilt_strength=0.50,
        is_seed=None,
        continuity_correction=True
    ):
        """
        Compute P(c is first eliminated | active S) via MVN orthant.
        If MVN CDF returns < mvn_floor (numerical underflow) or errors,
        optionally replace with a tiny importance-sampling estimate (NOT lower bound).
        """
        rng = np.random.default_rng(is_seed)
        k = len(S_tuple)
        b = bound_tally_for_subset(S_tuple)
        q = renorm_q(S_tuple)

        # Feasibility gates
        forced_i, cannot_out = feasibility_classify(b)
        if forced_i is not None:
            out = np.zeros(k, dtype=np.float64)
            out[forced_i] = 1.0
            return out
        if U == 0:
            totals = b
            minv = totals.min()
            mins = np.where(totals == minv)[0]
            out = np.zeros(k, dtype=np.float64)
            out[mins] = 1.0 / len(mins)
            return out

        # Covariance for unbound counts
        Cov_n = U * (np.diag(q) - np.outer(q, q))
        elim_probs = np.zeros(k, dtype=np.float64)

        def proj_params_for_a(a):
            others = [j for j in range(k) if j != a]
            # D rows: e_a - e_j
            d = np.zeros((k - 1, k), dtype=float)
            for r, j in enumerate(others):
                d[r, a] = 1.0
                d[r, j] = -1.0
            mu = (b[a] - b[others]) + U * (q[a] - q[others])  # mean of D
            Cov_D = d @ Cov_n @ d.T                           # cov of D
            return others, mu, Cov_D

        def rare_event_is_prob_for_a(a, thresh_vec, others):
            """
            Estimate P( D_j <= thresh for ALL j in others ) via IS on the U-count vector.
            q_tilt reduces mass on a, inflating others proportionally.
            """
            if is_samples <= 0:
                return 0.0

            qa = q[a]
            qa_tilt = max(qa * (1.0 - is_tilt_strength), 1e-12)  # shrink a's mass
            rem = 1.0 - qa_tilt
            q_others = q.copy()
            q_others[a] = 0.0
            if 1.0 - qa <= 0:
                return 0.0
            scale = rem / (1.0 - qa)
            q_tilt = q_others * scale
            q_tilt[a] = qa_tilt

            # numeric guards
            q_tilt = np.clip(q_tilt, 1e-16, 1.0)
            q_tilt /= q_tilt.sum()

            log_q = np.log(q + 1e-300)
            log_qt = np.log(q_tilt + 1e-300)

            wsum = 0.0
            for _ in range(is_samples):
                n = rng.multinomial(U, q_tilt)
                # likelihood ratio (multinomial coeff cancels)
                lr = exp(np.dot(n, (log_q - log_qt)))
                # event check
                ok = True
                for idx, j in enumerate(others):
                    Dj = (n[a] - n[j]) + (b[a] - b[j])
                    if Dj > thresh_vec[idx]:
                        ok = False
                        break
                if ok:
                    wsum += lr
            return wsum / max(is_samples, 1)

        for a in tqdm(range(k)):
            if a in cannot_out:
                elim_probs[a] = 0.0
                continue

            others, mu, Cov_D = proj_params_for_a(a)
            m = len(others)
            thresh = np.full(m, -0.5 if continuity_correction else 0.0, dtype=float)

            # (A) Main MVN orthant
            try:
                p_mvn = MVN(mean=mu, cov=Cov_D, allow_singular=True).cdf(thresh)
            except Exception:
                # jitter if numerical issue
                jitter = 1e-10 * np.eye(m)
                p_mvn = MVN(mean=mu, cov=Cov_D + jitter, allow_singular=True).cdf(thresh)

            p_mvn = float(p_mvn)
            if not np.isfinite(p_mvn) or p_mvn < 0:
                p_mvn = 0.0

            # (B) Optional IS replacement ONLY if MVN is too tiny / underflowed
            if (p_mvn < 1e-12) and use_importance_sampling:
                p_is = rare_event_is_prob_for_a(a, thresh, others)
                if np.isfinite(p_is) and p_is >= 0:
                    elim_probs[a] = p_is
                else:
                    elim_probs[a] = p_mvn
            else:
                elim_probs[a] = p_mvn

        # Light renormalization ONLY if already close to 1
        s = elim_probs.sum()
        if 0.95 <= s <= 1.05 and s > 0:
            elim_probs /= s

        return elim_probs

    memo = {}
    edges = {}  # (parent_tuple, child_tuple) -> branch_prob

    def recurse(S_tuple):
        if len(S_tuple) == 1:
            return {(): 1.0}
        if S_tuple in memo:
            return memo[S_tuple]

        p_out = first_elimination_distribution_mvn(
            S_tuple,
            mvn_floor=1e-12,
            use_importance_sampling=True,   # SAFE: only used if MVN underflows
            is_samples=is_samples,
            is_tilt_strength=0.50,
            is_seed=None,                   # do not fix seed to keep variety across runs
            continuity_correction=continuity_correction
        )

        out = defaultdict(float)
        for i, p in enumerate(p_out):
            if p <= 0:
                continue
            child = S_tuple[:i] + S_tuple[i+1:]
            edges[(S_tuple, child)] = float(p)
            child_dist = recurse(child)
            c_elim = S_tuple[i]
            for suffix, pc in child_dist.items():
                out[(c_elim,) + suffix] += p * pc
        memo[S_tuple] = dict(out)
        return memo[S_tuple]

    dist = recurse(all_cands)
    if return_tree:
        return dist, edges
    return dist

# Elimination DAG helpers (plotting)
def accumulate_node_reach(edges, root):
    """
    Given edges {(u,v): p}, compute reach probability for each node:
      reach[root] = 1, reach[v] += reach[u] * p(u->v)
    Returns: reach (dict), layers (dict node->depth), children (dict node->list)
    """
    children = defaultdict(list)
    parents = defaultdict(list)
    nodes = set([root])
    for (u, v), p in edges.items():
        children[u].append((v, p))
        parents[v].append((u, p))
        nodes.add(u); nodes.add(v)

    root_len = len(root)
    layers = {root: 0}
    frontier = [root]
    while frontier:
        nxt = []
        for u in frontier:
            for v, _ in children.get(u, []):
                if v not in layers:
                    layers[v] = root_len - len(v)
                    nxt.append(v)
        frontier = nxt

    order = sorted(nodes, key=lambda x: (layers.get(x, root_len - len(x)), -len(x), x))
    reach = defaultdict(float)
    reach[root] = 1.0
    for u in order:
        for v, p in children.get(u, []):
            reach[v] += reach[u] * p
    return dict(reach), layers, {k: [c for c,_ in v] for k,v in children.items()}


def _layered_positions(layers):
    by_layer = defaultdict(list)
    for node, d in layers.items():
        by_layer[d].append(node)
    pos = {}
    for d in sorted(by_layer):
        row = by_layer[d]
        row = sorted(row, key=lambda t: (len(t), t))
        n = len(row)
        xs = np.linspace(0, 1, n) if n > 1 else np.array([0.5])
        y = 1 - d / max(1, max(layers.values()))
        for x, node in zip(xs, row):
            pos[node] = (x, y)
    return pos


def plot_elim_tree(
    edges,
    root,
    reach,
    layers,
    figsize=(12, 7),
    cmap_name='GnBu',
    edge_cmap_name='OrRd',
    edge_width_range=(1.0, 3.0)
):
    G = nx.DiGraph()
    for (u, v), p in edges.items():
        G.add_edge(u, v, weight=p)

    pos = _layered_positions(layers)

    node_vals = [reach.get(n, 0.0) for n in G.nodes()]
    nmin = float(min(node_vals) if node_vals else 0.0)
    nmax = float(max(node_vals) if node_vals else 1.0)
    if nmax == nmin:
        nmax = nmin + 1.0
    node_norm = mcolors.Normalize(vmin=nmin, vmax=nmax)
    node_cmap = mpl.colormaps[cmap_name]
    node_colors = [node_cmap(node_norm(reach.get(n, 0.0))) for n in G.nodes()]

    def fmt_node(n):
        rem = "{" + ",".join(n) + "}"
        rp = reach.get(n, 0.0)
        return f"{rem}\nP={rp:.4f}"
    labels = {n: fmt_node(n) for n in G.nodes()}

    edge_list = list(G.edges())
    edge_probs = [G[u][v]['weight'] for (u, v) in edge_list]
    if edge_probs:
        emin = float(min(edge_probs))
        emax = float(max(edge_probs))
        if emax == emin:
            emax = emin + 1.0
    else:
        emin, emax = 0.0, 1.0
    edge_norm = mcolors.Normalize(vmin=emin, vmax=emax)
    edge_cmap = mpl.colormaps[edge_cmap_name]

    edge_colors = [edge_cmap(edge_norm(p)) for p in edge_probs]
    edge_widths = list(np.interp(edge_probs, [emin, emax], edge_width_range)) if edge_probs else 1.0

    edge_labels = {(u, v): f"{G[u][v]['weight']:.3f}" for (u, v) in edge_list}

    fig, ax = plt.subplots(figsize=figsize)

    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=900,
        linewidths=0.5, edgecolors='k', ax=ax
    )
    pos_nodes = {n: (x, y + np.random.uniform(0.01, 0.02)) for n, (x, y) in pos.items()}
    nx.draw_networkx_labels(G, pos_nodes, labels=labels, font_size=15, ax=ax,
                            horizontalalignment='right', verticalalignment='bottom')

    nx.draw_networkx_edges(
        G, pos, arrows=True, arrowstyle='-|>', arrowsize=12,
        width=edge_widths, alpha=0.9, edge_color=edge_colors, ax=ax
    )

    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_size=12, ax=ax,
        label_pos=0.3, rotate=True,
        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.75)
    )

    sm_nodes = mpl.cm.ScalarMappable(cmap=node_cmap, norm=node_norm)
    sm_nodes.set_array([])
    cbar_nodes = fig.colorbar(sm_nodes, ax=ax, shrink=0.75, pad=0.02)
    cbar_nodes.set_label("Node reach probability", rotation=270, labelpad=12)

    sm_edges = mpl.cm.ScalarMappable(cmap=edge_cmap, norm=edge_norm)
    sm_edges.set_array([])
    cbar_edges = fig.colorbar(sm_edges, ax=ax, shrink=0.75, pad=0.08)
    cbar_edges.set_label("Edge branch probability", rotation=270, labelpad=12)

    ax.set_axis_off()
    fig.tight_layout()
    plt.savefig("elimination_tree.pdf")
    plt.show()

if __name__ == "__main__":
    candidates = ["A","B","C","D","E"]

    bound1 = 270*[list(c) for c in permutations(candidates) if c[0] in ("A")]
    bound2 = 355*[list(c) for c in permutations(candidates) if c[0] in ("B")]
    bound3 = 360*[list(c) for c in permutations(candidates) if c[0] in ("C")]
    bound4 = 350*[list(c) for c in permutations(candidates) if c[0] in ("D")]
    bound5 = 300*[list(c) for c in permutations(candidates) if c[0] in ("E")]
    bound = bound1 + bound2 + bound3 + bound4 + bound5

    P = {'A': 0.4, 'B': 0.30, 'C': 0.30, 'D': 0.15, 'E': 0.20}
    s = sum(P.values())
    P = {k: v/s for k, v in P.items()}
    U = 20000

    # Compute distribution + tree
    t0 = time.time()
    dist, edges = rcv_elimination_orders_prob_mvn(
        bound, P, U,
        continuity_correction=True,
        is_samples=50000,
        return_tree=True
    )
    dist = pad_cdf_orders(dist, candidates)
    elapsed = time.time() - t0
    print(f"Time: {elapsed:.3f}s")

    # Winner marginals + top orders
    winners = defaultdict(float)
    dist = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
    for order, p in dist:
        winners[order[-1]] += p
        if p > 1e-4:
            print(f"Order: {order}, P={p:.8f}")

    print("Winners:")
    for w in sorted(winners):
        print(f"  {w}: {winners[w]:.8f}")

    # Rank correlations among top-5
    print("Top 5 rank correlations (Spearman rho, Kendall tau):")
    top_orders = [order for order, p in dist[:5]]
    for i in range(len(top_orders)):
        for j in range(i+1, len(top_orders)):
            rho, tau = rank_corr(top_orders[i], top_orders[j])
            print(f"  Orders {i+1} vs {j+1}: Spearman rho={rho:.4f}, Kendall tau={tau:.4f}")

    # Plot DAG
    root = tuple(sorted(P.keys()))
    reach, layers, _ = accumulate_node_reach(edges, root)
    plot_elim_tree(edges, root, reach, layers, figsize=(19, 15), cmap_name='coolwarm', edge_cmap_name='viridis')