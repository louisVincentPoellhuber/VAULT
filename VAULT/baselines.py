import logging
import random

from model.modeling_utils import *
from model.data_handler import VaultDataLoader

 
import os
import json
import tempfile
import shutil
from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import LuceneIndexer
from beir.retrieval.evaluation import EvaluateRetrieval
import subprocess
from tqdm import tqdm

logging.basicConfig( 
    encoding="utf-8", 
    filename=f"bm25.log", 
    filemode="a", 
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level = logging.INFO
    )

def corpus_iterator(corpus):
    if hasattr(corpus, "items"):
        yield from corpus.items()
    else:
        yield from iter(corpus)

def stream_and_index(corpus, index_dir, batch_size=100000):
    """
    Stream corpus documents into temporary JSONL shards, index each immediately, and delete.
    """

    indexer = LuceneIndexer(index_dir)
    batch = []
    for docid, doc in tqdm(corpus_iterator(corpus)):
        text = (doc.get("title", "") + " " + doc.get("text", "")).strip()
        batch.append({
            "id":str(docid), 
            "contents":text
            })

        if len(batch) >= batch_size:
            indexer.add_batch_dict(batch)
            batch.clear()
    indexer.add_batch_dict(batch)
    batch.clear()

    indexer.close()

def get_bm25_results(index_dir, queries):
    searcher = LuceneSearcher(index_dir)

    results = {}
    for qid, query in queries.items():
        hits = searcher.search(query, k=100)
        results[qid] = {hit.docid: hit.score for hit in hits}

    return results


def evaluate(index_dir, results, qrels): 
    k_values = [1, 5, 10, 20, 100] # Change if you want
    evaluator = CustomEvaluateRetrieval(None)
    ndcg, _map, recall, precision, mrr = evaluator.evaluate(qrels, results, k_values)
    
    metrics_path = os.path.join(index_dir, f"metrics.txt")
    with open(metrics_path, "w") as metrics_file:
        metrics_file.write("Retriever evaluation for k in: {}".format(k_values))
        metrics_file.write(f"\nNDCG: {ndcg}\nRecall: {recall}\nPrecision: {precision}\nMAP: {_map}\nMRR: {mrr}\n")


def run_baseline(datasets, baseline, baseline_dir, batch_size=100000):
    for dataset in datasets:
        print(f"\n=== Processing {dataset} ===")
        corpus, queries, qrels = VaultDataLoader(dataset).load("test")

        index_dir = os.path.join(baseline_dir, dataset)
        os.makedirs(index_dir, exist_ok=True)
        if baseline=="bm25":
            stream_and_index(corpus, index_dir, batch_size=batch_size)
            results = get_bm25_results(index_dir, queries)
        # NOTE: Add another baseline in an elif if you want.

        # Evaluate retrieval
        evaluate(index_dir, results, qrels)


if __name__ == "__main__":
    # Example datasets
    datasets = ["hotpotqa", "nq", "wikir", "scidocs", "cord19", "doris-mae"] 

    vault_dir = os.getenv("VAULT_DIR")
    if vault_dir == None: # TODO: Remove
        vault_dir = "/Tmp/lvpoellhuber/datasets/vault"

    benchmark_dir = os.path.join(vault_dir, "benchmarking")
    os.makedirs(benchmark_dir, exist_ok=True)

    baselines = ["bm25"]

    for baseline in baselines:
        baseline_dir = os.path.join(benchmark_dir, baseline)
        os.makedirs(baseline_dir, exist_ok=True)
        run_baseline(datasets, baseline, baseline_dir, batch_size=100000)



   

# log_message(f"========================= Evaluating BM25.=========================")

# storage_path = os.path.join(STORAGE_DIR, "datasets/vault")
# vault_datasets =["wikir", "nq", "hotpotqa", "doris-mae"]

# for dataset in vault_datasets:
#     corpus, queries, qrels = VaultDataLoader(dataset).load(split="test")


#     index_path = "/Tmp/lvpoellhuber/datasets/msmarco-doc/bm25index" 

#     #### Intialize ####
#     # (1) True - Delete existing index and re-index all documents from scratch
#     # (2) False - Load existing index
#     initialize = True  # False

#     #### Sharding ####
#     # (1) For datasets with small corpus (datasets ~ < 5k docs) => limit shards = 1
#     # SciFact is a relatively small dataset! (limit shards to 1)
#     number_of_shards = 1
#     model = LuceneBM25(index_path=index_path)

#     # (2) For datasets with big corpus ==> keep default configuration
#     # model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
#     retriever = EvaluateRetrieval(model)

#     #### Retrieve dense results (format of results is identical to qrels)
#     results = retriever.retrieve(corpus, queries)

#     #### Evaluate your retrieval using NDCG@k, MAP@K ...
#     log_message(f"Retriever evaluation for k in: {retriever.k_values}")
#     ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

#     #### Retrieval Example ####
#     query_id, scores_dict = random.choice(list(results.items()))
#     log_message(f"Query : {queries[query_id]}\n")

#     with open("bm25_metrics.txt", "w") as metrics_file:
#         metrics_file.write("Retriever evaluation for k in: {}".format(retriever.k_values))
#         metrics_file.write(f"\nNDCG: {ndcg}\nRecall: {recall}\nPrecision: {precision}\n")

#         top_k = 10

#         query_id, ranking_scores = random.choice(list(results.items()))
#         scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
#         metrics_file.write("Query : %s\n" % queries[query_id])

#         for rank in range(top_k):
#             doc_id = scores_sorted[rank][0]
#             # Format: Rank x: ID [Title] Body
#             metrics_file.write("Rank %d: %s [%s] - %s\n" % (rank+1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))
