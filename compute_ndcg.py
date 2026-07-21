#!/usr/bin/env python3
"""
NDCG@k metrics for RCV elimination order ranking quality.

Computes Normalized Discounted Cumulative Gain to evaluate how well predicted
elimination orders match ground truth orders, with emphasis on top positions.
"""

import math
from typing import List, Tuple, Dict, Optional


def normalize_ranking(pred_order: List[str], true_order: List[str]) -> List[str]:

    seen = set()
    deduped = []
    true_set = set(true_order)
    
    for cand in pred_order:
        if cand not in seen and cand in true_set:
            deduped.append(cand)
            seen.add(cand)
    
    missing = sorted(true_set - seen)
    
    return deduped + missing


def compute_ndcg(
    true_order: List[str], 
    pred_order: List[str], 
    k: int, 
    order_format: str = "eliminated_to_winner"
) -> float:
    
    if not true_order or not pred_order:
        return 0.0
    
    if order_format == "eliminated_to_winner":
        true_winner_first = list(reversed(true_order))
        pred_winner_first = list(reversed(pred_order))
    else: 
        true_winner_first = list(true_order)
        pred_winner_first = list(pred_order)
    
    pred_winner_first = normalize_ranking(pred_winner_first, true_winner_first)
    
    m = len(true_winner_first)
    

    relevance = {}
    for idx, cand in enumerate(true_winner_first):
        relevance[cand] = m - (idx + 1)
    
    k_eff = min(k, m)
    dcg = 0.0
    
    for j in range(k_eff):
        cand = pred_winner_first[j]
        rel = relevance.get(cand, 0)
        gain = (2 ** rel) - 1 
        discount = math.log2(j + 2)
        dcg += gain / discount
    
    idcg = 0.0
    for j in range(k_eff):
        cand = true_winner_first[j]
        rel = relevance[cand]
        gain = (2 ** rel) - 1
        discount = math.log2(j + 2)
        idcg += gain / discount
    
    if idcg == 0:
        # Only possible if all gains are 0 (e.g., m=1)
        # Return 0 to avoid division by zero
        return 0.0
    
    return dcg / idcg


def compute_ndcg_2_5(
    true_order: List[str], 
    pred_order: List[str], 
    order_format: str = "eliminated_to_winner"
) -> Tuple[float, float]:
    """
    Compute both NDCG@2 and NDCG@5.
    """
    ndcg_2 = compute_ndcg(true_order, pred_order, k=2, order_format=order_format)
    ndcg_5 = compute_ndcg(true_order, pred_order, k=5, order_format=order_format)
    return ndcg_2, ndcg_5


def extract_best_order_from_dist(dist: Dict[str, float]) -> List[str]:
    if not dist:
        return []
    
    best_order_str = max(dist.items(), key=lambda x: x[1])[0]
    
    return best_order_str.split("|||")


def compute_expected_ndcg(
    true_order: List[str],
    pred_dist: Dict[Tuple[str, ...], float],
    k: Optional[int] = None,
    order_format: str = "eliminated_to_winner"
) -> float:
    """
    Compute Expected NDCG@k over a predicted distribution of elimination orders.
    """
    if not true_order or not pred_dist:
        return 0.0
    
    # Default k to full ranking
    if k is None:
        k = len(true_order)
    
    expected_ndcg = 0.0
    
    for order_tuple, prob in pred_dist.items():
        pred_order_list = list(order_tuple)
        ndcg_score = compute_ndcg(true_order, pred_order_list, k=k, order_format=order_format)
        expected_ndcg += prob * ndcg_score
    
    return expected_ndcg


# Expose main API
__all__ = [
    'compute_ndcg',
    'compute_ndcg_2_5',
    'normalize_ranking',
    'extract_best_order_from_dist',
    'compute_expected_ndcg'
]
