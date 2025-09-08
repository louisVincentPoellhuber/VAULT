from dataclasses import dataclass, field
from typing import Optional, Union
import os
import json

@dataclass
class DataTrainingArguments:
    exp_name: Optional[str] = field(default="test-doc")
    task: Optional[str] = field(default="msmarco-doc")
    tokenizer_name: Optional[str] = field(default="bert-base-uncased")
    query_file: Optional[str] = field(default=f"{os.getenv("STORAGE_DIR")}/datasets/msmarco-doc/queries.jsonl")
    corpus_file: Optional[str] = field(default=f"{os.getenv("STORAGE_DIR")}/datasets/msmarco-doc/corpus.jsonl")
    qrels_file: Optional[str] = field(default=f"{os.getenv("STORAGE_DIR")}/datasets/msmarco-doc/qrels/train.tsv")
    nqrels_file: Optional[str] = field(default=f"{os.getenv("STORAGE_DIR")}/datasets/msmarco-doc/nqrels/train.tsv")
    max_query_length: Optional[int] = field(default=512)
    max_corpus_length: Optional[int] = field(default=512)
    max_corpus_sent_num: Optional[int] = field(default=8)
    encoder_mlm_probability: Optional[float] = field(default=0.3)
    decoder_mlm_probability: Optional[float] = field(default=0.5)
    normalize: Optional[bool] = field(default=False)
    loss_function: Optional[str] = field(default="contrastive")
    min_corpus_len: Optional[int] = field(default=0)
    base_model: Optional[str] = field(default="bert-base-uncased")
    base_model_postfix: Optional[str] = field(default="true")
    negatives: Optional[bool] = field(default=False)
    streaming: Optional[bool] = field(default=False)

@dataclass
class ModelArguments:
    model_type: Optional[str] = field(default="longtriever")
    model_name_or_path: Optional[str] = field(default=f"{os.getenv("STORAGE_DIR")}/models/longtriever/pretrained/bert-base-uncased")
    ablation_config: Optional[str]  = field(default_factory=lambda:'{"inter_block_encoder":true, "doc_token":true, "start_separator": false, "text_separator": true, "end_separator": false, "segments": false, "cls_position": "first"}')
    doc_token_init: Optional[str] = field(default="default") # default, zero, cls
    output_attentions: Optional[bool] = field(default=False)
    pooling_strategy: Optional[str] = field(default="mean")
    
    def __post_init__(self):
        if isinstance(self.ablation_config, str):
            try:
                self.ablation_config = json.loads(self.ablation_config)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON for ablation_config: {self.ablation_config}") from e