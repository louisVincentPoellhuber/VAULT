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
from model.data_handler import DataCollatorForEvaluatingLongtriever, DataCollatorForEvaluatingHierarchicalLongtriever,DataCollatorForEvaluatingBert, VaultDataLoader
from model.modeling_retriever import LongtrieverRetriever,BertRetriever,SiameseRetriever
from model.modeling_longtriever import Longtriever
from model.modeling_hierarchical import HierarchicalLongtriever
from model.modeling_utils import *


from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import FlatIPFaissSearch
from search import StreamedFlatIPFaissSearch, CustomFlatIPFaissSearch
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
                data_collator=data_collator
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


def get_dataloader(model, model_args, data_args, training_args):
    # inter_block_encoder is always True for hierarchical and longtriever models
    if (model_args.model_type=="bert") or (model_args.model_type=="dpr"):
        corpus_chunk_size = 10000 # 25000
    else:
        corpus_chunk_size = 50000

    dataloader = VaultDataLoader(data_args.task)
    corpus, queries, qrels = dataloader.load(split="test")

    if dataloader.corpus_file.endswith("db"):
        faiss_search = StreamedFlatIPFaissSearch(model, batch_size=training_args.eval_batch_size, corpus_chunk_size=corpus_chunk_size) 
    else:
        faiss_search = CustomFlatIPFaissSearch(model, batch_size=training_args.eval_batch_size, corpus_chunk_size=corpus_chunk_size) 

    return faiss_search, corpus, queries, qrels


def index_corpus(corpus, faiss_search, training_args):
    if training_args.overwrite_output_dir or not os.path.exists(os.path.join(training_args.output_dir, "default.flat.tsv")):
        log_message("Indexing.")
        faiss_search.index(corpus=corpus, score_function="dot")
        log_message("Saving.")
        faiss_search.save(training_args.output_dir, prefix="default")
    else:
        faiss_search.load(training_args.output_dir, prefix="default")
        log_message("Already indexed, loading.")


def retrieve_and_eval(corpus, queries, qrels, faiss_search, training_args):
    
    # retriever = EvaluateRetrieval(faiss_search, score_function="dot")
    retriever = CustomEvaluateRetrieval(faiss_search, score_function="dot")

    log_message("Retrieving.")
    results = retriever.retrieve(corpus, queries)

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
        config_path = os.path.join(os.getcwd(), os.path.join("configs", "bert_test.json"))
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

    log_message(f"========================= Evaluating run {training_args.run_name}.=========================")

    model = get_model(model_args, data_args)

    faiss_search, corpus, queries, qrels = get_dataloader(model, model_args, data_args, training_args)

    index_corpus(corpus, faiss_search, training_args)
        
    log_message(f"========================= Results for: {training_args.run_name}.=========================")
    retrieve_and_eval(corpus, queries, qrels, faiss_search, training_args)

if __name__ == "__main__":    
    
    main()