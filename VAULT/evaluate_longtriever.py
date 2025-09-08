import os
import random
import json
import importlib.util
import sys
if importlib.util.find_spec("faiss") is not None:
    import faiss
import dotenv
dotenv.load_dotenv()
from tqdm import tqdm

from model.arguments import DataTrainingArguments, ModelArguments
from model.data_handler import DataCollatorForEvaluatingLongtriever, DataCollatorForEvaluatingHierarchicalLongtriever,DataCollatorForEvaluatingBert, StreamedDataLoader
from model.modeling_retriever import LongtrieverRetriever,BertRetriever
from model.modeling_longtriever import Longtriever
from model.modeling_hierarchical import HierarchicalLongtriever
from model.modeling_utils import *


from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import FlatIPFaissSearch
from streamed_search import StreamedFlatIPFaissSearch
from transformers import AutoTokenizer, HfArgumentParser,TrainingArguments,BertModel

STORAGE_DIR = os.getenv("STORAGE_DIR")

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) <=1 :
        # model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        model_args, data_args, training_args = parser.parse_json_file(json_file="/u/poellhul/Documents/Masters/benchmarkIR-slurm/src/longtriever/configs/streaming_test.json", allow_extra_keys=True)
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    model_args: ModelArguments
    data_args: DataTrainingArguments
    training_args: TrainingArguments

    # Main arguments
    model_path = training_args.output_dir
    batch_size = training_args.per_device_eval_batch_size
    task = data_args.task
    exp_name = training_args.run_name
    
    log_note_path = os.path.join(model_path, "slurm_ids.txt")
    with open(log_note_path, "a") as log_file:
        slurm_id = os.environ["SLURM_JOB_ID"] if "SLURM_JOB_ID" in os.environ else "local"
        log_file.write(f"Evaluating Job Slurm ID: {slurm_id}; Computer: {os.uname()[1]}\n")

    log_message(f"========================= Evaluating run {exp_name}.=========================")

    tokenizer=AutoTokenizer.from_pretrained(data_args.tokenizer_name)
    if model_args.model_type=="longtriever":
        data_collator=DataCollatorForEvaluatingLongtriever(
                tokenizer,
                data_args.max_query_length,
                data_args.max_corpus_length, 
                data_args.max_corpus_sent_num
            )
    elif model_args.model_type=="hierarchical":
        data_collator=DataCollatorForEvaluatingHierarchicalLongtriever(
                tokenizer,
                data_args.max_query_length,
                data_args.max_corpus_length,
                data_args.max_corpus_sent_num, 
                start_separator = model_args.ablation_config.get("start_separator", False), 
                text_separator = model_args.ablation_config.get("text_separator", True), 
                end_separator = model_args.ablation_config.get("end_separator", False)
            )
    elif model_args.model_type=="bert":
        data_collator=DataCollatorForEvaluatingBert( 
                tokenizer,
                data_args.max_query_length,
                data_args.max_corpus_length,
                data_args.max_corpus_sent_num
            )
        
        
    log_message("Loading model.")
    if model_args.model_type=="longtriever":
        encoder = Longtriever.from_pretrained(
                model_args.model_name_or_path, 
                ablation_config=model_args.ablation_config
            )
        model = LongtrieverRetriever(
                model=encoder, 
                normalize=data_args.normalize,
                loss_function=data_args.loss_function, 
                data_collator=data_collator
            ) 
    elif model_args.model_type=="hierarchical":
        encoder = HierarchicalLongtriever.from_pretrained(
                model_args.model_name_or_path, 
                ablation_config=model_args.ablation_config, 
                pooling_strategy=model_args.pooling_strategy
            )
        model = LongtrieverRetriever(
                model=encoder, 
                normalize=data_args.normalize,
                loss_function=data_args.loss_function, 
                data_collator=data_collator
            )     
    elif model_args.model_type=="bert":
        encoder = BertModel.from_pretrained(
                model_args.model_name_or_path
            )
        model = BertRetriever(
                model=encoder, 
                normalize=data_args.normalize,
                loss_function=data_args.loss_function, 
                data_collator=data_collator
            )     
    model.eval()

    if (not model_args.ablation_config.get("inter_block_encoder", True)) or (model_args.model_type=="bert"):
        corpus_chunk_size = 25000
    else:
        corpus_chunk_size = 50000


    data_path = os.path.join(STORAGE_DIR, "datasets", task)
    if task=="nq": 
        data_path = os.path.join(data_path, "nq")

    if data_args.streaming:
        faiss_search = StreamedFlatIPFaissSearch(model, batch_size=batch_size, corpus_chunk_size=corpus_chunk_size) 
        corpus, queries, qrels = StreamedDataLoader(
            corpus_file=data_args.corpus_file, 
            query_file=data_args.query_file, 
            qrels_file=data_args.qrels_file).load(split="test")
    else:
        faiss_search = FlatIPFaissSearch(model, batch_size=batch_size, corpus_chunk_size=corpus_chunk_size) 
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    if data_args.min_corpus_len>0:
        if data_args.streaming:
            raise Exception("Cannot filter corpus with streaming corpus.")
        new_qrels = {}
        new_corpus = {}
        for qid, doc in qrels.items():
            docid = list(doc.keys())[0]
            document = corpus[docid]
            if len(document["title"].split(" ")) + len(document["text"].split(" ")) > data_args.min_corpus_len:
                new_qrels[qid] = {docid:1}
        print(f"Original corpus size: {len(qrels)}, Filtered corpus size: {len(new_qrels)}")
        qrels = new_qrels

        for docid in tqdm(corpus.keys()):
            document = corpus[docid]
            if len(document["title"].split(" ")) + len(document["text"].split(" ")) > data_args.min_corpus_len:
                new_corpus[docid]=document
        corpus = new_corpus

    if training_args.overwrite_output_dir or not os.path.exists(os.path.join(model_path, "default.flat.tsv")):
        log_message("Indexing.")
        faiss_search.index(corpus=corpus, score_function="dot")
        log_message("Saving.")
        faiss_search.save(model_path, prefix="default")
    else:
        faiss_search.load(model_path, prefix="default")
        log_message("Already indexed, loading.")
        
    # retriever = EvaluateRetrieval(faiss_search, score_function="dot")
    retriever = CustomEvaluateRetrieval(faiss_search, score_function="dot")

    log_message("Retrieving.")
    results = retriever.retrieve(corpus, queries)

    log_message(results)

    log_message(qrels)

    log_message("Evaluating.")
    ndcg, _map, recall, precision, mrr = retriever.evaluate(qrels, results, retriever.k_values)
    
    metrics_path = os.path.join(model_path, f"{task}_metrics.txt")
    with open(metrics_path, "w") as metrics_file:
        metrics_file.write("Retriever evaluation for k in: {}".format(retriever.k_values))
        metrics_file.write(f"\nNDCG: {ndcg}\nRecall: {recall}\nPrecision: {precision}\nMAP: {_map}\nMRR: {mrr}\n")

        top_k = 10

        query_id, ranking_scores = random.choice(list(results.items()))
        scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
        metrics_file.write("Query : %s\n" % queries[query_id])

        for rank in range(top_k):
            doc_id = scores_sorted[rank][0]
            # Format: Rank x: ID [Title] Body
            metrics_file.write("Rank %d: %s [%s] - %s\n" % (rank+1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))


if __name__ == "__main__":    
    
    main()