def recall_at_k(scores, true_items, k):
    """
    scores: list of (problem_id, predicted_score)
    true_items: set of problems that are actually positive
    """
    top_k = [pid for pid, _ in sorted(scores, key=lambda x: x[1], reverse=True)[:k]]
    hits = len([p for p in top_k if p in true_items])
    return hits / len(true_items) if len(true_items) > 0 else 0.0
