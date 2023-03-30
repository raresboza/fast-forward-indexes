import logging
from sys import maxsize
from collections import defaultdict
from fast_forward.ranking import Ranking


LOGGER = logging.getLogger(__name__)


def interpolate(
    r1: Ranking, r2: Ranking, alpha: float, name: str = None, sort: bool = True, normalization: str = "off"
) -> Ranking:
    """Interpolate scores. For each query-doc pair:
        * If the pair has only one score, ignore it.
        * If the pair has two scores, interpolate: r1 * alpha + r2 * (1 - alpha).

    Args:
        r1 (Ranking): Scores from the first retriever.
        r2 (Ranking): Scores from the second retriever.
        alpha (float): Interpolation weight.
        name (str, optional): Ranking name. Defaults to None.
        sort (bool, optional): Whether to sort the documents by score. Defaults to True.

    Returns:
        Ranking: Interpolated ranking.
    """
    if normalization == "local":
        r1 = normalise_ranking(r1)
        r2 = normalise_ranking(r2)
    elif normalization == "global":
        r1 = normalise_all_ranking(r1)
        r2 = normalise_all_ranking(r2)
    # else don't do anything

    assert r1.q_ids == r2.q_ids
    results = defaultdict(dict)
    for q_id in r1:
        for doc_id in r1[q_id].keys() & r2[q_id].keys():
            results[q_id][doc_id] = (
                alpha * r1[q_id][doc_id] + (1 - alpha) * r2[q_id][doc_id]
            )
    return Ranking(results, name=name, sort=sort, copy=False)


def reciprocal_ranked_fusion(
        r1: Ranking, r2: Ranking, name: str = None, sort: bool = True, normalization: str = "off"
) -> Ranking:
    if normalization == "local":
        r1 = normalise_ranking(r1)
        r2 = normalise_ranking(r2)
    elif normalization == "global":
        r1 = normalise_all_ranking(r1)
        r2 = normalise_all_ranking(r2)
    # else don't do anything

    assert r1.q_ids == r2.q_ids
    results = defaultdict(dict)
    for q_id in r1:
        query_documents = r1[q_id].keys() & r2[q_id].keys()
        r1_q_ids = list(r1[q_id].keys())
        r2_q_ids = list(r2[q_id].keys())
        # r2_q_ids = [doc[0] for doc in sorted(r2[q_id].items(), key=lambda x: x[1], reverse=True)]
        for doc_id in query_documents:
            # compute places of the document in both rankings
            position_in_r1 = r1_q_ids.index(doc_id)
            position_in_r2 = r2_q_ids.index(doc_id)
            # compute the score
            results[q_id][doc_id] = (1.0 / (position_in_r1 + 1)) + (1.0 / (position_in_r2 + 1))
    return Ranking(results, name=name, sort=sort, copy=False)


def normalise_ranking(r: Ranking, name: str = None, sort: bool = True) -> Ranking:
    normalised_result = defaultdict(dict)
    for q_id in r:
        max_score = float(max(r[q_id].items(), key=lambda x: x[1])[1])
        min_score = float(min(r[q_id].items(), key=lambda x: x[1])[1])
        for doc_id in r[q_id].keys():
            normalised_result[q_id][doc_id] = (r[q_id][doc_id] - min_score) / (max_score - min_score)
    return Ranking(normalised_result, name=name, sort=sort, copy=False)


def normalise_all_ranking(r: Ranking, name: str = None, sort: bool = True) -> Ranking:
    max_score = -1
    min_score = maxsize
    for q_id in r:
        local_max = float(max(r[q_id].items(), key=lambda x: x[1])[1])
        max_score = local_max if local_max > max_score else max_score
        local_min = float(min(r[q_id].items(), key=lambda x: x[1])[1])
        min_score = local_min if local_min < min_score else min_score

    normalised_result = defaultdict(dict)
    for q_id in r:
        for doc_id in r[q_id].keys():
            normalised_result[q_id][doc_id] = (r[q_id][doc_id] - min_score) / (max_score - min_score)
    return Ranking(normalised_result, name=name, sort=sort, copy=False)
