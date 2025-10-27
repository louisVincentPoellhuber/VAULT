import comet_ml
import logging
import os
import sys
import dotenv
import json
dotenv.load_dotenv()
import warnings

from model.data_handler import DatasetForFineTuning,DataCollatorForFineTuningLongtriever,DataCollatorForFineTuningHierarchicalLongtriever,DatasetForPretraining,LongtrieverCollator,DataCollatorForFineTuningBert
from model.modeling_longtriever import Longtriever, LongtrieverForPretraining
from model.modeling_hierarchical import HierarchicalLongtriever
from model.modeling_retriever import LongtrieverRetriever, BertRetriever
from model.modeling_utils import *

from model.arguments import DataTrainingArguments, ModelArguments
from trainer import PreTrainer
import transformers
from transformers import AutoTokenizer,HfArgumentParser,TrainingArguments,BertModel
from transformers.trainer_utils import is_main_process
import torch

def setup_distributed():
    log_message("Setting up distributed training. This message should not appear on Slurm.")
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        torch.distributed.init_process_group(
            backend="nccl",  # or "gloo" for CPU
            init_method="env://"
        )
    else:
        print("Not running in distributed mode")

def get_model(model_args, data_args):
    
    log_message("Loading model.")
    if model_args.model_type=="longtriever":
        encoder = Longtriever.from_pretrained(
                model_args.model_name_or_path,
                doc_token_init=model_args.doc_token_init
            )
        model = LongtrieverRetriever(
                model=encoder,
                normalize=data_args.normalize,
                loss_function=data_args.loss_function
            )
    elif model_args.model_type=="hierarchical":
        encoder = HierarchicalLongtriever.from_pretrained(
                model_args.model_name_or_path,
                segments=model_args.segments,
                doc_token_init=model_args.doc_token_init,
                output_attentions=model_args.output_attentions,
                output_passage_embeddings=model_args.output_passage_embeddings
            )
        model = LongtrieverRetriever(
                model=encoder,
                normalize=data_args.normalize,
                loss_function=data_args.loss_function, 
                output_passage_embeddings=model_args.output_passage_embeddings
            )     
    elif model_args.model_type=="bert":
        encoder = BertModel.from_pretrained(
                model_args.model_name_or_path
            )
        model = BertRetriever(
                model=encoder, 
                normalize=data_args.normalize,
                loss_function=data_args.loss_function
            )     
        
    return model

def get_dataset(model_args, data_args, training_args):
    dataset = DatasetForFineTuning(data_args)
    log_message(f"Streaming data: {dataset.streaming}", print_message=True)

    return dataset

def get_data_collator(model_args, data_args):
    
    # Create data collator & model
    log_message("Loading dataset.")
    tokenizer=AutoTokenizer.from_pretrained(data_args.tokenizer_name)

    if model_args.model_type=="longtriever":
        log_message("Loading dataset for fine-tuning")
        data_collator=DataCollatorForFineTuningLongtriever(
                tokenizer=tokenizer,
                max_query_length=data_args.max_query_length,
                max_corpus_length=data_args.max_corpus_length,
                max_corpus_sent_num=data_args.max_corpus_sent_num
            )
    elif model_args.model_type=="hierarchical":
        log_message("Loading dataset for fine-tuning")
        data_collator=DataCollatorForFineTuningHierarchicalLongtriever(
                tokenizer=tokenizer,
                max_query_length=data_args.max_query_length,
                max_corpus_length=data_args.max_corpus_length,
                max_corpus_sent_num=data_args.max_corpus_sent_num
            )
    elif model_args.model_type=="bert":
        log_message("Loading dataset for fine-tuning")
        data_collator=DataCollatorForFineTuningBert(
                tokenizer,
                data_args.max_query_length,
                data_args.max_corpus_length,
                data_args.max_corpus_sent_num
            )
    else:
        raise ValueError(f"Model type {model_args.model_type} not supported")
    

    return tokenizer, data_collator

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) <=1 :
        config_path = os.path.join(os.getcwd(), os.path.join("configs", "hier_passage.json"))
        # config_path = "/u/poellhul/Documents/Masters/VAULT/configs/pubmedbert_test.json"
        model_args, data_args, training_args = parser.parse_json_file(json_file=config_path, allow_extra_keys=True)
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
    log_message(f"========================= Finetuning {training_args.run_name} =========================")

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    model_args: ModelArguments
    data_args: DataTrainingArguments
    training_args: TrainingArguments

    training_args.remove_unused_columns=False
    training_args.seed=43

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    log_message(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    log_message(f"Training parameters {training_args}")

    model = get_model(model_args, data_args)
    dataset = get_dataset(model_args, data_args, training_args)
    tokenizer, data_collator = get_data_collator(model_args, data_args)

    # Initialize our Trainer
    trainer = PreTrainer(
        model=model,                                                                       
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    if (os.getenv("SLURM_JOB_ID")==None) and (os.getenv("EXP_NAME")==None):
        setup_distributed()

    # Train
    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    main()