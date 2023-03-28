import csv
from pathlib import Path

import numpy as np
from fast_forward.ranking import Ranking
from fast_forward.index import Mode, InMemoryIndex
from fast_forward.encoder import TCTColBERTQueryEncoder as TCTColBERTQueryEncoderFF
from pyserini.search.faiss import FaissSearcher, TctColBertQueryEncoder
from ir_measures import read_trec_qrels, calc_aggregate, nDCG, RR


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
        calc_aggregate([nDCG@10, RR(rel=2)@10], qrels, sparse_ranking_2019.run)
    )
    print(
        f"BM25, TCTColBERT (alpha={alpha})",
        calc_aggregate([nDCG@10, RR(rel=2)@10], qrels, result[alpha].run)
    )


if __name__ == "__main__":
    sparse_ranking_2019, sparse_ranking_2020 = setup()
    fusion(sparse_ranking_2019)
