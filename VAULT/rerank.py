import os
import random
import json
import importlib.util
import dotenv
dotenv.load_dotenv()
from tqdm import tqdm
import sys

from bm25 import get_bm25_results, stream_and_index

from model.arguments import DataTrainingArguments, ModelArguments
from model.data_handler import DataCollatorForEvaluatingLongtriever, DataCollatorForEvaluatingHierarchicalLongtriever,DataCollatorForEvaluatingBert, VaultDataLoader
from model.modeling_retriever import LongtrieverRetriever,BertRetriever,SiameseRetriever
from model.modeling_longtriever import Longtriever
from model.modeling_hierarchical import HierarchicalLongtriever
from model.modeling_utils import *


from beir.retrieval.evaluation import EvaluateRetrieval
from transformers import AutoTokenizer, HfArgumentParser,TrainingArguments,BertModel,DPRContextEncoder, DPRQuestionEncoder

STORAGE_DIR = os.getenv("STORAGE_DIR")

def get_data_collator(model_args, data_args):
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
                data_args.max_corpus_sent_num
            )
    elif (model_args.model_type=="bert") or (model_args.model_type=="dpr"):
        data_collator=DataCollatorForEvaluatingBert( 
                tokenizer,
                data_args.max_query_length,
                data_args.max_corpus_length,
                data_args.max_corpus_sent_num, 
                model_args.output_passage_embeddings
            )
            
    # elif model_args.model_type=="YOUR_MODEL"  # You can implement your data collator here. 
        
    return data_collator
        

def get_model(model_args, data_args):
    data_collator = get_data_collator(model_args, data_args)

    log_message("Loading model.")
    if model_args.model_type=="longtriever":
        encoder = Longtriever.from_pretrained(
                model_args.model_name_or_path
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
                segments=model_args.segments,
                output_passage_embeddings=model_args.output_passage_embeddings
            )
        model = LongtrieverRetriever(
                model=encoder,
                normalize=data_args.normalize,
                loss_function=data_args.loss_function,
                data_collator=data_collator, 
                output_passage_embeddings=model_args.output_passage_embeddings
            )     
    elif model_args.model_type=="bert":
        encoder = BertModel.from_pretrained(
                model_args.model_name_or_path
            )
        model = BertRetriever(
                model=encoder, 
                normalize=data_args.normalize,
                loss_function=data_args.loss_function, 
                data_collator=data_collator, 
                output_passage_embeddings=model_args.output_passage_embeddings
            )     
    elif model_args.model_type=="dpr":
        ctx_encoder = DPRContextEncoder.from_pretrained(model_args.ctx_model_name_or_path)
        q_encoder = DPRQuestionEncoder.from_pretrained(model_args.q_model_name_or_path)
        model = SiameseRetriever(ctx_encoder=ctx_encoder,
                                 q_encoder=q_encoder,
                                normalize=data_args.normalize,
                                loss_function=data_args.loss_function, 
                                data_collator=data_collator
                                )
    model.eval()

    return model

def rerank(model, index_dir, corpus, queries, top_k=100):
    bm25_results = get_bm25_results(index_dir, queries, top_k)

    results = {}
    for qid in tqdm(bm25_results.keys(), desc="Reranking queries"):
        top_docids = list(bm25_results[qid].keys())
        sub_corpus = [{"_id":docid, "text":corpus[docid]["text"], "title":corpus[docid]["title"]} for docid in top_docids]

        reranked_results = model.rerank(query=queries[qid], corpus=sub_corpus, batch_size=64, verbose=False)

        results[qid] = reranked_results
    
    return results

def retrieve_and_eval(corpus, queries, qrels, model, bm25_index_dir, training_args):
    
    # retriever = EvaluateRetrieval(faiss_search, score_function="dot")
    retriever = CustomEvaluateReranking(model, k_values=[1,3,5,10,100,1000]) # No score_function cuz rerank only implements dot function. 

    log_message("Retrieving.")
    results = rerank(model, bm25_index_dir, corpus, queries, top_k=retriever.top_k)

    # Sanitize results and qrels
    results = {str(qid): {str(docid): float(score) for docid, score in docs.items()}
       for qid, docs in results.items()}
    qrels = {str(qid): {str(docid): int(rel) for docid, rel in docs.items()}
            for qid, docs in qrels.items()}

    log_message("Evaluating.")
    ndcg, _map, recall, precision, mrr = retriever.evaluate(qrels, results, retriever.k_values)
    

    metrics_path = os.path.join(training_args.output_dir, f"{training_args.run_name}_metrics.txt")
    with open(metrics_path, "w") as metrics_file:
        metrics_file.write("Retriever evaluation for k in: {}".format(retriever.k_values))
        metrics_file.write(f"\nNDCG: {ndcg}\nRecall: {recall}\nPrecision: {precision}\nMAP: {_map}\nMRR: {mrr}\n")

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) <=1 :
        # TODO: Remove configs maybe?
        config_path = os.path.join(os.getcwd(), os.path.join("configs", "bert_passage.json"))
        model_args, data_args, training_args = parser.parse_json_file(json_file=config_path, allow_extra_keys=True)
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    model_args: ModelArguments
    data_args: DataTrainingArguments
    training_args: TrainingArguments

    os.makedirs(training_args.output_dir, exist_ok=True)

    # TODO: Remove
    log_note_path = os.path.join(training_args.output_dir, "slurm_ids.txt")
    with open(log_note_path, "a") as log_file:
        slurm_id = os.environ["SLURM_JOB_ID"] if "SLURM_JOB_ID" in os.environ else "local"
        log_file.write(f"Evaluating Job Slurm ID: {slurm_id}; Computer: {os.uname()[1]}\n")

    log_message(f"========================= Indexing run {training_args.run_name}.=========================")

    model = get_model(model_args, data_args)

    dataloader = VaultDataLoader(data_args.task)
    corpus, queries, qrels = dataloader.load(split="test")
    
    bm25_index_dir = os.path.join(training_args.output_dir, "bm25_index")
    if not os.path.exists(bm25_index_dir):
        stream_and_index(corpus, bm25_index_dir, batch_size=100000)

    log_message(f"========================= Reranking: {training_args.run_name}.=========================")
    retrieve_and_eval(corpus, queries, qrels, model, bm25_index_dir, training_args)

if __name__ == "__main__":    
    
    main()