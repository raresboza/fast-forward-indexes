import csv
from pathlib import Path

from fast_forward.ranking import Ranking
from fast_forward.index import Mode, InMemoryIndex
from fast_forward.encoder import TCTColBERTQueryEncoder as TCTColBERTQueryEncoderFF
from typing import Dict, List
from ir_measures import read_trec_qrels, calc_aggregate, nDCG, RR, iter_calc, measures


def setup():
    sparse_ranking_2019 = Ranking.from_file(Path("data/msmarco-passage-test2019-sparse10000.txt"))
    sparse_ranking_2019.cut(5000)
    sparse_ranking_2020 = Ranking.from_file(Path("data/msmarco-passage-test2020-sparse10000.txt"))
    sparse_ranking_2020.cut(5000)
    all_ids = set.union(
        *[set(sparse_ranking_2019[q_id].keys()) for q_id in sparse_ranking_2019.q_ids],
        *[set(sparse_ranking_2020[q_id].keys()) for q_id in sparse_ranking_2020.q_ids]
    )
    print(f"indexing {len(all_ids)} documents or passages")
    return sparse_ranking_2019, sparse_ranking_2020


def fusion(sparse_ranking):
    # THIS IS FOR LOADING INDEXES FROM A PICKLE FILE
    index = InMemoryIndex.from_disk(
        index_file=Path("ffindex_passage_2019_2020.pkl"),
        encoder=TCTColBERTQueryEncoderFF("castorini/tct_colbert-msmarco"),
        mode=Mode.PASSAGE,
    )

    # RRF
    with open(
        "data/msmarco-test2019-queries.tsv",
        encoding="utf-8",
        newline=""
    ) as fp:
        queries = {q_id: q for q_id, q in csv.reader(fp, delimiter="\t")}
    print(f"loaded {len(queries)} queries")

    alpha = 0.2
    result = index.get_scores(
        sparse_ranking,
        queries,
        alpha=alpha,
        cutoff=10,
        early_stopping=False,
        rrf=True,
    )

    qrels = list(read_trec_qrels("data/2019qrels-pass.txt"))
    print(
        "BM25",
        calc_aggregate([nDCG@10, RR(rel=2)@10], qrels, sparse_ranking.run)
    )
    print(
        f"BM25, TCTColBERT (alpha={alpha})",
        calc_aggregate([nDCG@10, RR(rel=2)@10], qrels, result[alpha].run)
    )


def cc(sparse_ranking):
    # THIS IS FOR LOADING INDEXES FROM A PICKLE FILE
    index = InMemoryIndex.from_disk(
        index_file=Path("ffindex_passage_2019_2020.pkl"),
        encoder=TCTColBERTQueryEncoderFF("castorini/tct_colbert-msmarco"),
        mode=Mode.PASSAGE,
    )

    # Convex combination
    with open(
            "data/msmarco-test2019-queries.tsv",
            encoding="utf-8",
            newline=""
    ) as fp:
        queries = {q_id: q for q_id, q in csv.reader(fp, delimiter="\t")}
    print(f"loaded {len(queries)} queries")

    alpha = 0.2
    result = index.get_scores(
        sparse_ranking,
        queries,
        alpha=alpha,
        cutoff=10,
        early_stopping=False
    )

    qrels = list(read_trec_qrels("data/2019qrels-pass.txt"))
    print(
        "BM25",
        calc_aggregate([nDCG @ 10, RR(rel=2) @ 10], qrels, sparse_ranking.run)
    )
    print(
        f"BM25, TCTColBERT (alpha={alpha})",
        calc_aggregate([nDCG @ 10, RR(rel=2) @ 10], qrels, result[alpha].run)
    )


def testing(ranking: Ranking, queries: Dict[str, str], qrels: List, alpha: float, rrf: bool = False,
            normalization: str = "off"):
    # THIS IS FOR LOADING INDEXES FROM A PICKLE FILE
    index = InMemoryIndex.from_disk(
        index_file=Path("ffindex_passage_2019_2020.pkl"),
        encoder=TCTColBERTQueryEncoderFF("castorini/tct_colbert-msmarco"),
        mode=Mode.PASSAGE,
    )

    result = index.get_scores(
        ranking,
        queries,
        alpha=alpha,
        cutoff=10,
        early_stopping=False,
        rrf=rrf,
        normalization=normalization,
    )

    print(
        "BM25",
        calc_aggregate([nDCG @ 10, RR(rel=2) @ 10], qrels, ranking.run)
    )
    print(
        f"BM25, TCTColBERT (alpha={alpha})",
        calc_aggregate([nDCG @ 10, RR(rel=2) @ 10], qrels, result[alpha].run)
    )
    return calc_aggregate([measures.MAP @ 10], qrels, ranking.run), calc_aggregate([measures.MAP @ 10], qrels, result[alpha].run)


def testing_individual(ranking: Ranking, queries: Dict[str, str], qrels: List, alpha: float, rrf: bool = False,
                       normalization: str = "off"):
    index = InMemoryIndex.from_disk(
        index_file=Path("ffindex_passage_2019_2020.pkl"),
        encoder=TCTColBERTQueryEncoderFF("castorini/tct_colbert-msmarco"),
        mode=Mode.PASSAGE,
    )

    result = index.get_scores(
        ranking,
        queries,
        alpha=alpha,
        cutoff=10,
        early_stopping=False,
        rrf=rrf,
        normalization=normalization,
    )

    regular_ranking = list(iter_calc([nDCG @ 10, RR(rel=2) @ 10, measures.MAP @ 10], qrels, ranking.run))
    our_ranking = list(iter_calc([nDCG @ 10, RR(rel=2) @ 10, measures.MAP @ 10], qrels, result[alpha].run))

    for m in sorted(our_ranking, key=lambda x: int(x[0])):
        with open("data/rankings/change_me.txt", 'a') as f:
            print(m, file=f)

    # for m in sorted(regular_ranking, key=lambda x: int(x[0])):
    #     with open("data/rankings/bm25_only_ranking.txt", 'a') as f:
    #         print(m, file=f)


if __name__ == "__main__":
    sparse_ranking_2019, sparse_ranking_2020 = setup()

    # Load queries
    with open(
            "data/msmarco-test2019-queries.tsv",
            encoding="utf-8",
            newline=""
    ) as fp:
        queries19 = {q_id: q for q_id, q in csv.reader(fp, delimiter="\t")}
    qrels19 = list(read_trec_qrels("data/2019qrels-pass.txt"))

    with open(
            "data/msmarco-test2020-queries.tsv",
            encoding="utf-8",
            newline=""
    ) as fp:
        queries20 = {q_id: q for q_id, q in csv.reader(fp, delimiter="\t")}
    qrels20 = list(read_trec_qrels("data/2020qrels-pass.txt"))

    # hyper parameter tuning
    # alpha_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # for alpha in alpha_range:
    #     combined_ranking_results = testing(ranking=sparse_ranking_2019, queries=queries19, qrels=qrels19, alpha=alpha, rrf=False, normalization="off")
    #     print(combined_ranking_results)

    # Tune the test here
    testing_individual(
        ranking=sparse_ranking_2020,
        queries=queries20,
        qrels=qrels20,
        alpha=0.2,
        rrf=True,
        normalization="global"
    )
