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

def get_bm25_results(index_dir, queries, k=100):
    searcher = LuceneSearcher(index_dir)

    results = {}
    for qid, query in queries.items():
        hits = searcher.search(query, k)
        results[qid] = {hit.docid: hit.score for hit in hits}

    return results


def evaluate(index_dir, results, qrels): 
    k_values = [1, 3, 5, 10, 100, 1000] # Change if you want
    evaluator = CustomEvaluateRetrieval(None)
    ndcg, _map, recall, precision, mrr = evaluator.evaluate(qrels, results, k_values)
    
    metrics_path = os.path.join(index_dir, f"metrics.txt")
    with open(metrics_path, "w") as metrics_file:
        metrics_file.write("Retriever evaluation for k in: {}".format(k_values))
        metrics_file.write(f"\nNDCG: {ndcg}\nRecall: {recall}\nPrecision: {precision}\nMAP: {_map}\nMRR: {mrr}\n")


def run_baseline(datasets, baseline_dir, batch_size=100000):
    for dataset in datasets:
        log_message(f"\n========================= Baseline: {dataset} =========================", print_message=True)
        corpus, queries, qrels = VaultDataLoader(dataset).load("test")

        index_dir = os.path.join(baseline_dir, dataset)
        os.makedirs(index_dir, exist_ok=True)
        stream_and_index(corpus, index_dir, batch_size=batch_size)
        results = get_bm25_results(index_dir, queries)

        # Evaluate retrieval
        evaluate(index_dir, results, qrels)


if __name__ == "__main__":
    # Example datasets
    datasets = ["wikipedia", "trec-covid"] 
    # datasets = ["hotpotqa", "nq", "wikir", "trec-covid", "doris-mae", "wikipedia"] 

    vault_dir = os.getenv("VAULT_DIR")
    if vault_dir == None: # TODO: Remove
        vault_dir = "/Tmp/lvpoellhuber/datasets/vault"

    benchmark_dir = os.path.join(vault_dir, "benchmarking")
    os.makedirs(benchmark_dir, exist_ok=True)


    baseline_dir = os.path.join(benchmark_dir, "bm25")
    os.makedirs(baseline_dir, exist_ok=True)
    run_baseline(datasets, baseline_dir, batch_size=100000)


