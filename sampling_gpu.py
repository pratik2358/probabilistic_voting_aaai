import os
import json
import math
import time
import random
import signal
import traceback
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from itertools import permutations
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm

# Lazy torch import - only load when actually using GPU
_HAS_TORCH = False
_TORCH_IMPORTED = False

def _ensure_torch():
    """Lazy import torch only when needed."""
    global _HAS_TORCH, _TORCH_IMPORTED
    if not _TORCH_IMPORTED:
        try:
            import torch  # noqa: F401
            globals()['torch'] = torch
            _HAS_TORCH = True
        except Exception:
            _HAS_TORCH = False
        _TORCH_IMPORTED = True
    return _HAS_TORCH

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def atomic_write_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)

def load_or_init_results(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"meta": {}, "rows": []}

def make_row_id(num: int, case: str, U: int, n_b: int, N_per_order: int, use_torch: bool) -> str:
    return f"m{num}_case{case}_nb{n_b}_U{U}_N{N_per_order}_torch{int(use_torch)}"

def safe_jsonable(obj: Any) -> Any:
    try:
        import numpy as _np
        if isinstance(obj, (_np.integer,)): return int(obj)
        if isinstance(obj, (_np.floating,)): return float(obj)
        if isinstance(obj, (_np.ndarray,)): return obj.tolist()
    except Exception:
        pass
    if isinstance(obj, tuple): return [safe_jsonable(x) for x in obj]
    if isinstance(obj, dict): return {str(k): safe_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list): return [safe_jsonable(x) for x in obj]
    return obj

def _read_proc_status_value_kb(pid: int, key: str) -> Optional[int]:
    try:
        with open(f"/proc/{pid}/status", "r") as f:
            for line in f:
                if line.startswith(key):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1])
    except Exception:
        return None
    return None

def rss_mb(pid: int) -> float:
    kb = _read_proc_status_value_kb(pid, "VmRSS:")
    if kb is None:
        return -1.0
    return kb / 1024.0

def hwm_mb(pid: int) -> float:
    kb = _read_proc_status_value_kb(pid, "VmHWM:")
    if kb is None:
        return -1.0
    return kb / 1024.0

def save_json(path: str, obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)

def dataset_key(num: int, case: str, U: int, n_b: int) -> str:
    return f"m{num}_case{case}_nb{n_b}_U{U}"

def build_probs(num: int, case: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Same p_bound/p_unbound construction as your earlier loops."""
    if case == 'perfect_match':
        p_bound = {chr(ord("A") + i): random.random() for i in range(num)}
        s = sum(p_bound.values())
        p_bound = {k: v / s for k, v in p_bound.items()}
        p_unbound = p_bound.copy()

    elif case == 'slight_mismatch':
        p_bound = {chr(ord("A") + i): random.random() for i in range(num)}
        s = sum(p_bound.values())
        p_bound = {k: v / s for k, v in p_bound.items()}
        p_unbound = {k: max(0.01, v + random.uniform(-0.05, 0.05)) for k, v in p_bound.items()}
        s2 = sum(p_unbound.values())
        p_unbound = {k: v / s2 for k, v in p_unbound.items()}

    else:  # reverse
        p_bound = {chr(ord("A") + i): random.random() for i in range(num)}
        s = sum(p_bound.values())
        p_bound = {k: v / s for k, v in p_bound.items()}
        items = sorted(p_bound.items(), key=lambda kv: kv[1], reverse=True)
        p_unbound = {k: v for k, v in reversed(items)}
        s2 = sum(p_unbound.values())
        p_unbound = {k: v / s2 for k, v in p_unbound.items()}

    return p_bound, p_unbound

def generate_ballots(P_dict, n: int = 1000):
    p_order = sorted(P_dict.keys(), key=lambda x: [P_dict[x]], reverse=True)
    ballots = []
    for _ in range(n):
        k = random.randint(1, len(p_order))
        ballot = []
        for _i in range(k):
            cand = np.random.choice(
                list(set(p_order) - set(ballot)),
                p=[P_dict[c] / sum([P_dict[x] for x in p_order if x not in ballot]) for c in p_order if c not in ballot],
                replace=False
            )
            ballot.append(cand)
        ballots.append(ballot)
    return ballots

def get_or_create_ballots(
    ballots_root: str,
    num: int,
    case: str,
    n_b: int,
    U: int,
    generate_ballots_fn,
    base_seed: int = 12345,
) -> Tuple[Dict[str, float], Dict[str, float], List[List[str]], List[List[str]]]:
    ensure_dir(ballots_root)
    key = dataset_key(num, case, U, n_b)
    folder = os.path.join(ballots_root, key)
    ensure_dir(folder)

    p_bound_path = os.path.join(folder, "p_bound.json")
    p_unbound_path = os.path.join(folder, "p_unbound.json")
    bound_path = os.path.join(folder, "bound.json")
    unbound_path = os.path.join(folder, "unbound.json")
    meta_path = os.path.join(folder, "meta.json")

    if all(os.path.exists(p) for p in [p_bound_path, p_unbound_path, bound_path, unbound_path]):
        return load_json(p_bound_path), load_json(p_unbound_path), load_json(bound_path), load_json(unbound_path)

    derived_seed = (hash((base_seed, num, case, n_b, U)) % (2**32))
    random.seed(derived_seed)
    try:
        np.random.seed(derived_seed % (2**32 - 1))
    except Exception:
        pass

    p_bound, p_unbound = build_probs(num, case)
    bound = generate_ballots_fn(p_bound, n=n_b)
    unbound = generate_ballots_fn(p_unbound, n=U)

    save_json(p_bound_path, p_bound)
    save_json(p_unbound_path, p_unbound)
    save_json(bound_path, bound)
    save_json(unbound_path, unbound)
    save_json(meta_path, {"num_cands": num, "case": case, "n_b": n_b, "U": U, "seed": int(derived_seed)})

    return p_bound, p_unbound, bound, unbound

def _torch_device():
    if not _ensure_torch():
        return None
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def compute_bound_tally(active: List[str], bound_ballots: List[List[str]]) -> Dict[str, int]:
    active_set = set(active)
    tally = {c: 0 for c in active}
    for ranking in bound_ballots:
        for pref in ranking:
            if pref in active_set:
                tally[pref] += 1
                break
    return tally

def multinomial_draw(n: int, probs: List[float], labels: List[str]) -> Dict[str, int]:
    out = {lab: 0 for lab in labels}
    if n > 0:
        draws = random.choices(labels, weights=probs, k=n)
        for lab in draws:
            out[lab] += 1
    return out

def feasible_order_first_round_only(pi: List[str], bound_ballots: List[List[str]], p_dict: Dict[str, float], m: int) -> bool:
    """
    SAFE feasibility filter: checks only the FIRST elimination step.
    Filtering later steps using only bound tallies is too strict because later rounds
    can benefit from transfers of unbound votes from earlier eliminations.

    This returns True if the first eliminated candidate e=pi[0] can be in the min-tie
    after distributing at most m unbound votes.
    """
    if len(pi) <= 1:
        return True
    active = list(pi)
    e = active[0]
    B = compute_bound_tally(active, bound_ballots)

    total_req = 0
    for o in active[1:]:
        need = max(0, B[e] - B[o])
        if need > 0 and p_dict.get(o, 0.0) == 0.0:
            return False
        total_req += need
    return total_req <= m

def _init_worker_print_pid(verbose: bool = False):
    if verbose:
        print(f"[worker-start] PID={os.getpid()}", flush=True)

def all_candidate_orders(candidates: List[str]) -> List[Tuple[str, ...]]:
    return list(permutations(candidates))

def all_feasible_orders_parallel(
    candidates: List[str],
    bound_ballots: List[List[str]],
    p_dict: Dict[str, float],
    m: int,
    n_jobs: Optional[int] = None,
    verbose_workers: bool = False,
) -> Tuple[List[Tuple[str, ...]], str]:
    """
    Returns a list of orders to evaluate and a mode string.
    Mode is:
      - "first_round_filter" if some orders passed the safe filter,
      - "fallback_all_perms" if none passed (so we evaluate all permutations).
    """
    if n_jobs is None or n_jobs <= 0:
        n_jobs = mp.cpu_count()

    orders = all_candidate_orders(candidates)
    if len(orders) <= 1:
        return orders, "trivial"

    feas: List[Tuple[str, ...]] = []

    # Fast serial path to avoid multiprocessing pickling issues in notebooks.
    if n_jobs == 1:
        for pi in orders:
            if feasible_order_first_round_only(list(pi), bound_ballots, p_dict, m):
                feas.append(pi)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_worker_print_pid, initargs=(verbose_workers,)) as ex:
            fut_map = {ex.submit(feasible_order_first_round_only, list(pi), bound_ballots, p_dict, m): pi for pi in orders}
            for fut in as_completed(fut_map):
                pi = fut_map[fut]
                ok = fut.result()
                if ok:
                    feas.append(pi)

    if feas:
        return sorted(feas), "first_round_filter"
    return sorted(orders), "fallback_all_perms"

def _mc_plain_once(pi: List[str], bound_ballots: List[List[str]], p_norm: Dict[str, float], m: int) -> float:
    """One Monte Carlo sample of the per-order weight under fair tie-breaking; returns w in [0,1]."""
    C = list(pi)
    x0 = multinomial_draw(m, [p_norm[c] for c in C], C)
    active = list(C)
    u = {c: x0[c] for c in active}
    w = 1.0
    for r in range(len(pi) - 1):
        elim = pi[r]
        B = compute_bound_tally(active, bound_ballots)
        T = {c: B[c] + u[c] for c in active}
        mT = min(T.values())
        L = [c for c, v in T.items() if v == mT]
        if elim not in L:
            return 0.0
        w *= 1.0 / len(L)
        x_elim = u[elim]
        active.remove(elim)
        u.pop(elim)
        if x_elim > 0 and active:
            denom = sum(p_norm[c] for c in active)
            if denom == 0.0:
                return 0.0
            probs = [p_norm[c] / denom for c in active]
            add = multinomial_draw(x_elim, probs, active)
            for c in active:
                u[c] += add[c]
    return w

def _bound_tally_tensor(active_list: List[str], bound_ballots: List[List[str]], device) -> "torch.Tensor":
    B = compute_bound_tally(active_list, bound_ballots)
    return torch.tensor([B[c] for c in active_list], dtype=torch.int32, device=device)

def _safe_multinomial_batch(N: int, total_count: int, probs: "torch.Tensor", device: "torch.device") -> "torch.Tensor":
    cpu_probs = probs.detach().to("cpu")
    dist = torch.distributions.Multinomial(total_count=total_count, probs=cpu_probs)
    MAX_CPU_BATCH = 32768
    outs = []
    remain = N
    while remain > 0:
        b = min(MAX_CPU_BATCH, remain)
        outs.append(dist.sample((b,)))
        remain -= b
    u_cpu = torch.cat(outs, dim=0)
    return u_cpu.to(device)

def _torch_mc_once_batch(pi: List[str], bound_ballots: List[List[str]],
                         p_norm: Dict[str, float], m: int, N: int, device=None) -> "torch.Tensor":
    if device is None:
        device = _torch_device()
    active = list(pi)

    p_vec = torch.tensor([p_norm[c] for c in active], dtype=torch.float32, device=device)
    p_vec = p_vec / (p_vec.sum() + 1e-12)

    u = _safe_multinomial_batch(N, m, p_vec, device)
    w = torch.ones(N, dtype=torch.float32, device=device)

    for _ in range(len(pi) - 1):
        elim_idx = 0

        B_t = _bound_tally_tensor(active, bound_ballots, device=device)
        T = u + B_t.unsqueeze(0).to(u.dtype)

        mins = T.min(dim=1, keepdim=True).values
        tie_mask = (T == mins)
        elim_in_tie = tie_mask[:, elim_idx]
        tie_sizes = tie_mask.sum(dim=1).to(w.dtype)

        w = w * torch.where(elim_in_tie, 1.0 / tie_sizes.clamp_min(1.0), torch.zeros_like(w))

        x_elim = u[:, elim_idx].round().to(torch.int64)
        if u.shape[1] == 1:
            break

        p_surv = p_vec[1:]
        denom = p_surv.sum()
        if denom.item() == 0.0:
            w = w * (x_elim == 0).float()
            u = u[:, 1:]
            active.pop(0)
            p_vec = p_surv
            continue

        p_surv = p_surv / denom

        unique_counts, inv = torch.unique(x_elim, return_inverse=True)
        add = torch.zeros((N, len(active) - 1), dtype=u.dtype, device=device)
        cat = torch.distributions.Categorical(probs=p_surv)

        for i, k in enumerate(unique_counts.tolist()):
            if k <= 0:
                continue
            idx = (inv == i).nonzero(as_tuple=True)[0]
            M = idx.numel()
            if M == 0:
                continue
            samp = cat.sample((M, k))
            K = p_surv.numel()
            off = torch.arange(M, device=device).unsqueeze(1) * K
            flat = (samp + off).reshape(-1)
            bc = torch.bincount(flat, minlength=M * K).reshape(M, K).to(add.dtype)
            add[idx] = bc

        u = torch.cat([u[:, 1:]], dim=1) + add
        active.pop(0)
        p_vec = p_surv

    return w

def _empirical_bernstein_halfwidth(var_hat: float, n: int, delta: float, sigma: float) -> float:
    """Empirical Bernstein half-width for bounded [0,1] variables with variance capped by sigma^2."""
    if n <= 1:
        return float("inf")
    logt = math.log(3.0 / max(delta, 1e-12))
    v = min(max(var_hat, 0.0), float(sigma) ** 2)
    return math.sqrt(2.0 * v * logt / n) + 3.0 * logt / (n - 1)

def _hoeffding_halfwidth(n: int, delta: float) -> float:
    if n <= 0:
        return float("inf")
    return math.sqrt(math.log(2.0 / max(delta, 1e-12)) / (2.0 * n))

def mc_order_probability_adaptive(
    pi: List[str],
    bound_ballots: List[List[str]],
    p_dict: Dict[str, float],
    m: int,
    *,
    epsilon: float,
    delta: float = 0.05,
    sigma: float = 0.5,
    N_cap: int = 200_000,
    N_min: int = 200,
    batch: int = 2_000,
    seed: Optional[int] = None,
    use_torch: bool = False,
    torch_device: Optional[str] = None,
) -> Tuple[float, Tuple[float, float], Dict[str, float]]:
    """
    Adaptive sampling until empirical Bernstein half-width <= epsilon (after N_min),
    or until N_cap samples are used.

    Returns: (p_hat, (lo, hi), stats) where stats includes n_used and hw_empbern.
    """
    if seed is not None:
        random.seed(seed)
        try:
            np.random.seed(seed % (2**32 - 1))
        except Exception:
            pass

    s = sum(p_dict.get(c, 0.0) for c in pi)
    if s == 0.0:
        return 0.0, (0.0, 0.0), {"n_used": 0.0, "var": 0.0, "hw_empbern": 0.0, "hw_hoeffding": 0.0}

    p_norm = {c: p_dict.get(c, 0.0) / s for c in pi}

    n = 0
    S = 0.0
    SS = 0.0
    hw = float("inf")

    if use_torch and _ensure_torch():
        dev = torch.device(torch_device) if torch_device else _torch_device()
        while n < N_cap and (n < N_min or hw > epsilon):
            b = min(batch, N_cap - n)
            w = _torch_mc_once_batch(pi, bound_ballots, p_norm, m, b, device=dev)
            S += float(w.sum().item())
            SS += float((w * w).sum().item())
            n += int(b)

            mu = S / n
            var_hat = max(0.0, SS / n - mu * mu)
            var_hat = min(var_hat, 0.25)
            hw = _empirical_bernstein_halfwidth(var_hat, n, delta, sigma)

        mu = S / n if n else 0.0
        var_hat = max(0.0, SS / n - mu * mu) if n else 0.0
        var_hat = min(var_hat, 0.25)
        hw = _empirical_bernstein_halfwidth(var_hat, n, delta, sigma) if n > 1 else 1.0
        return float(mu), (max(0.0, mu - hw), min(1.0, mu + hw)), {
            "n_used": float(n),
            "var": float(var_hat),
            "hw_empbern": float(hw),
            "hw_hoeffding": float(_hoeffding_halfwidth(n, delta)),
        }

    while n < N_cap and (n < N_min or hw > epsilon):
        b = min(batch, N_cap - n)
        for _ in range(b):
            w = _mc_plain_once(pi, bound_ballots, p_norm, m)
            S += w
            SS += w * w
        n += int(b)

        mu = S / n
        var_hat = max(0.0, SS / n - mu * mu)
        var_hat = min(var_hat, 0.25)
        hw = _empirical_bernstein_halfwidth(var_hat, n, delta, sigma)

    mu = S / n if n else 0.0
    var_hat = max(0.0, SS / n - mu * mu) if n else 0.0
    var_hat = min(var_hat, 0.25)
    hw = _empirical_bernstein_halfwidth(var_hat, n, delta, sigma) if n > 1 else 1.0
    return float(mu), (max(0.0, mu - hw), min(1.0, mu + hw)), {
        "n_used": float(n),
        "var": float(var_hat),
        "hw_empbern": float(hw),
        "hw_hoeffding": float(_hoeffding_halfwidth(n, delta)),
    }

def _mc_order_task_adaptive_cpuonly(
    pi_tuple: Tuple[str, ...],
    bound_ballots: List[List[str]],
    p_dict: Dict[str, float],
    m: int,
    epsilon: float,
    delta: float,
    sigma: float,
    N_cap: int,
    seed: Optional[int],
) -> Tuple[Tuple[str, ...], float, Tuple[float, float], Dict[str, float], int]:
    ph, ci, st = mc_order_probability_adaptive(
        list(pi_tuple),
        bound_ballots,
        p_dict,
        m,
        epsilon=epsilon,
        delta=delta,
        sigma=sigma,
        N_cap=N_cap,
        seed=(None if seed is None else int(seed) + (hash(pi_tuple) % 1_000_000)),
        use_torch=False,
    )
    return pi_tuple, float(ph), (float(ci[0]), float(ci[1])), st, os.getpid()

def mc_over_feasible_orders(
    candidates: List[str],
    bound_ballots: List[List[str]],
    p_dict: Dict[str, float],
    m: int,
    N_per_order: int = 10_000,
    *,
    epsilon: Optional[float] = None,
    sigma: float = 0.5,
    N_cap_per_order: Optional[int] = None,
    delta: float = 0.05,
    n_jobs_feas: Optional[int] = None,
    n_jobs_mc_orders: Optional[int] = None,
    seed: Optional[int] = None,
    use_torch: bool = False,
    torch_device: Optional[str] = None,
    verbose_workers: bool = False,
) -> Tuple[Dict[Tuple[str, ...], Tuple[float, Tuple[float, float]]], Dict[Tuple[str, ...], Dict[str, float]], Dict[str, Any]]:
    """
    Returns:
      order_probs: {order: (p_hat, (lo, hi))}
      order_stats: {order: {...}}  (includes n_used/hw_empbern in adaptive mode)
      meta: dict (includes feasible_mode)

    Backward compatible:
      - If epsilon is None => fixed N_per_order sampling (adaptive stats still returned but bounded by N_per_order).
      - If epsilon is not None => adaptive empirical-Bernstein with cap N_cap_per_order (or N_per_order fallback).
    """
    feas_orders, feasible_mode = all_feasible_orders_parallel(
        candidates=candidates,
        bound_ballots=bound_ballots,
        p_dict=p_dict,
        m=m,
        n_jobs=n_jobs_feas,
        verbose_workers=verbose_workers
    )

    meta: Dict[str, Any] = {
        "feasible_mode": feasible_mode,
        "num_orders_sampled": int(len(feas_orders)),
        "adaptive": bool(epsilon is not None),
    }

    if not feas_orders:
        return {}, {}, meta

    cap = int(N_cap_per_order if N_cap_per_order is not None else N_per_order)
    if epsilon is None:
        eps = 0.0
        n_min = cap
    else:
        eps = float(epsilon)
        n_min = 200

    meta.update({"epsilon": float(eps), "sigma": float(sigma), "N_cap_per_order": cap})

    results: Dict[Tuple[str, ...], Tuple[float, Tuple[float, float]]] = {}
    stats_out: Dict[Tuple[str, ...], Dict[str, float]] = {}

    if use_torch and _ensure_torch():
        dev = torch.device(torch_device) if torch_device else _torch_device()
        for pi in feas_orders:
            ph, ci, st = mc_order_probability_adaptive(
                list(pi),
                bound_ballots,
                p_dict,
                m,
                epsilon=eps,
                delta=float(delta),
                sigma=float(sigma),
                N_cap=cap,
                N_min=n_min,
                seed=(None if seed is None else int(seed) + (hash(pi) % 1_000_000)),
                use_torch=True,
                torch_device=str(dev)
            )
            results[pi] = (float(ph), (float(ci[0]), float(ci[1])))
            stats_out[pi] = {k: float(v) for k, v in st.items()}
        return results, stats_out, meta

    if n_jobs_mc_orders is None or n_jobs_mc_orders <= 0:
        n_jobs_mc_orders = mp.cpu_count()

    if n_jobs_mc_orders == 1:
        for pi in feas_orders:
            pi_tuple, ph, ci, st, _pid = _mc_order_task_adaptive_cpuonly(
                pi, bound_ballots, p_dict, m, float(eps), float(delta), float(sigma), cap, seed
            )
            results[pi_tuple] = (float(ph), (float(ci[0]), float(ci[1])))
            stats_out[pi_tuple] = {k: float(v) for k, v in st.items()}
        return results, stats_out, meta

    with ProcessPoolExecutor(max_workers=n_jobs_mc_orders, initializer=_init_worker_print_pid, initargs=(verbose_workers,)) as ex:
        futs = [
            ex.submit(
                _mc_order_task_adaptive_cpuonly,
                pi,
                bound_ballots,
                p_dict,
                m,
                float(eps),
                float(delta),
                float(sigma),
                cap,
                seed
            )
            for pi in feas_orders
        ]
        for f in as_completed(futs):
            pi_tuple, ph, ci, st, _pid = f.result()
            results[pi_tuple] = (float(ph), (float(ci[0]), float(ci[1])))
            stats_out[pi_tuple] = {k: float(v) for k, v in st.items()}

    return results, stats_out, meta