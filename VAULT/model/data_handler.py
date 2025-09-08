from modeling_utils import *
import os
import json
from dataclasses import dataclass
import torch.utils.data.dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase, DataCollatorForWholeWordMask
from datasets import load_dataset,concatenate_datasets,load_from_disk
from modeling_utils import tensorize_batch
import nltk
import sqlite3
from beir.datasets.data_loader import GenericDataLoader
from typing import Union, List
import dotenv
dotenv.load_dotenv()

nltk.download('punkt')

if JOBID == None: JOBID = "debug"
logger = logging.getLogger(__name__)


class DatasetForPretraining(torch.utils.data.Dataset):
    def __init__(self, args):
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)

        cached_path = './cached_data/train'
        if os.path.exists(cached_path):
            print(f'loading dataset from {cached_path}')
            self.dataset = load_from_disk(dataset_path=cached_path)
            return

        def book_tokenize_function(examples):
            return tokenizer(examples["text"], add_special_tokens=False, truncation=False,
                             return_attention_mask=False,return_token_type_ids=False,verbose=False)

        target_length = (args.max_corpus_length - tokenizer.num_special_tokens_to_add(pair=False))*args.max_corpus_sent_num

        def book_pad_each_line(examples):
            texts = []
            blocks = []
            curr_block = []
            for sent in examples['input_ids']:
                if len(curr_block)+len(sent) >= target_length and curr_block:
                    blocks.append(curr_block)
                    curr_block = []
                    if len(blocks)>=args.max_corpus_sent_num:
                        texts.append(blocks)
                        blocks=[]
                curr_block.extend(sent)
            if len(curr_block) > 0:
                blocks.append(curr_block)
            if len(blocks) > 0:
                texts.append(blocks)
            return {'token_ids': texts} # {'token_ids':[[[int]]]]}

        bookcorpus = load_dataset('bookcorpus', split='train')
        tokenized_bookcorpus = bookcorpus.map(book_tokenize_function, num_proc=64, remove_columns=["text"])
        processed_bookcorpus = tokenized_bookcorpus.map(book_pad_each_line, num_proc=64, batched=True,
                                                        batch_size=1000, remove_columns=["input_ids"])

        def wiki_tokenize_function(examples):
            sentences = nltk.sent_tokenize(examples["text"])
            return tokenizer(sentences, add_special_tokens=False, truncation=False, return_attention_mask=False,
                             return_token_type_ids=False,verbose=False)
            # return {'input_ids':[[int]]}

        def wiki_pad_each_line(examples):
            texts = []
            for sents in examples['input_ids']:
                blocks = []
                curr_block = []
                for sent in sents:
                    if len(curr_block)+len(sent) >= target_length and curr_block:
                        blocks.append(curr_block)
                        curr_block = []
                        if len(blocks)>=args.max_corpus_sent_num:
                            texts.append(blocks)
                            blocks=[]
                    curr_block.extend(sent)
                if len(curr_block) > 0:
                    blocks.append(curr_block)
                if len(blocks) > 0:
                    texts.append(blocks)
            return {'token_ids': texts} # {'token_ids':[[[int]]]]}

        # wiki = load_dataset("wikipedia", "20220301.en", split="train")
        # wiki = wiki.remove_columns("title")
        # tokenized_wiki = wiki.map(wiki_tokenize_function, num_proc=64, remove_columns=["text"])
        # processed_wiki = tokenized_wiki.map(wiki_pad_each_line, num_proc=64, batched=True, batch_size=1000,
        #                                     remove_columns=["input_ids"])

        # bert_dataset = concatenate_datasets([processed_bookcorpus, processed_wiki])
        bert_dataset = processed_bookcorpus
        self.dataset = bert_dataset

        self.dataset.save_to_disk(dataset_path=cached_path)

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)


@dataclass
class LongtrieverCollator(DataCollatorForWholeWordMask):
    max_corpus_sent_num: int = 5
    max_corpus_length: int = 512
    encoder_mlm_probability: float = 0.15
    decoder_mlm_probability: float = 0.15

    def __call__(self, examples):
        encoder_input_ids_batch = []
        encoder_attention_mask_batch = []
        encoder_labels_batch=[]
        decoder_input_ids_batch=[]
        decoder_matrix_attention_mask_batch = []
        decoder_labels_batch=[]

        block_len=self.max_corpus_length - self.tokenizer.num_special_tokens_to_add(False)

        for e in examples:

            input_ids_blocks = []
            attention_mask_blocks = []
            encoder_mlm_mask_blocks=[]
            matrix_attention_mask_blocks=[]
            decoder_labels_blocks=[]

            for token_ids in e['token_ids']:
                input_ids_block = self.tokenizer.build_inputs_with_special_tokens(token_ids[:block_len])
                tokens_block = [self.tokenizer._convert_id_to_token(tid) for tid in input_ids_block]
                self.mlm_probability = self.encoder_mlm_probability
                encoder_mlm_mask_block = self._whole_word_mask(tokens_block)

                self.mlm_probability = self.decoder_mlm_probability
                matrix_attention_mask_block = []
                for i in range(len(tokens_block)):
                    decoder_mlm_mask = self._whole_word_mask(tokens_block)
                    decoder_mlm_mask[i] = 1
                    matrix_attention_mask_block.append(decoder_mlm_mask)

                input_ids_blocks.append(torch.tensor(input_ids_block))
                attention_mask_blocks.append(torch.tensor([1] * len(input_ids_block)))
                input_ids_block[0] = -100
                input_ids_block[-1] = -100
                decoder_labels_blocks.append(torch.tensor(input_ids_block))

                encoder_mlm_mask_blocks.append(torch.tensor(encoder_mlm_mask_block))
                matrix_attention_mask_blocks.append(1 - torch.tensor(matrix_attention_mask_block))

            input_ids_blocks = tensorize_batch(input_ids_blocks, self.tokenizer.pad_token_id)
            attention_mask_blocks = tensorize_batch(attention_mask_blocks, 0)
            origin_input_ids_blocks = input_ids_blocks.clone()
            encoder_mlm_mask_blocks = tensorize_batch(encoder_mlm_mask_blocks, 0)
            encoder_input_ids_blocks, encoder_labels_blocks = self.torch_mask_tokens(input_ids_blocks, encoder_mlm_mask_blocks)
            decoder_labels_blocks = tensorize_batch(decoder_labels_blocks, -100)
            matrix_attention_mask_blocks = tensorize_batch(matrix_attention_mask_blocks, 0)

            encoder_input_ids_batch.append(encoder_input_ids_blocks)
            encoder_attention_mask_batch.append(attention_mask_blocks)
            encoder_labels_batch.append(encoder_labels_blocks)
            decoder_input_ids_batch.append(origin_input_ids_blocks)
            decoder_matrix_attention_mask_batch.append(matrix_attention_mask_blocks)
            decoder_labels_batch.append(decoder_labels_blocks)

        encoder_input_ids_batch=tensorize_batch(encoder_input_ids_batch,self.tokenizer.pad_token_id)
        encoder_attention_mask_batch=tensorize_batch(encoder_attention_mask_batch,0)
        encoder_labels_batch=tensorize_batch(encoder_labels_batch,-100)
        decoder_input_ids_batch=tensorize_batch(decoder_input_ids_batch,self.tokenizer.pad_token_id)
        decoder_matrix_attention_mask_batch=tensorize_batch(decoder_matrix_attention_mask_batch,0)
        decoder_labels_batch=tensorize_batch(decoder_labels_batch,-100)


        batch = {
            "encoder_input_ids_batch": encoder_input_ids_batch,
            "encoder_attention_mask_batch": encoder_attention_mask_batch,
            "encoder_labels_batch": encoder_labels_batch,
            "decoder_input_ids_batch": decoder_input_ids_batch,
            "decoder_matrix_attention_mask_batch": decoder_matrix_attention_mask_batch,  # [B,N,L,L]
            "decoder_labels_batch": decoder_labels_batch,
        }

        return batch
    


class StreamingCorpus():
    def __init__(self, file_path):
        self.connection = sqlite3.connect(file_path)
        self.connection.execute("PRAGMA journal_mode=WAL;")         # better concurrency & fewer locks
        self.connection.execute("PRAGMA synchronous=NORMAL;")       # faster writes (safe enough)
        self.connection.execute("PRAGMA temp_store=MEMORY;")        # temp tables in RAM
        self.connection.execute("PRAGMA mmap_size=30000000000;") 
        self.connection.row_factory = sqlite3.Row  # to access columns by name
        self.cursor = self.connection.cursor()

        self.cursor.execute("SELECT id FROM articles")
        ids = [row[0] for row in self.cursor.fetchall()]
        self.ids = ids

    def __getitem__(self, page_id):
        self.cursor.execute("SELECT id, title, text, url FROM articles WHERE id = ?", (int(page_id),))
        row = self.cursor.fetchone()

        if row:
            return {
                "title": row["title"], 
                "text": row["text"]
                }
        else:
            return None, None
        
    def __len__(self):
        return len(self.ids)
    
    def keys(self):
        return self.ids

    def __iter__(self):
        self.cursor.execute("SELECT id, title, text FROM articles")
        for row in self.cursor:
            yield row["id"], {"title": row["title"], "text": row["text"]}
    
class StreamedDataLoader(GenericDataLoader):
    def load(self, split="test") -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, int]]]:
        self.check(fIn=self.corpus_file, ext="db")
        self.check(fIn=self.query_file, ext="jsonl")
        self.check(fIn=self.qrels_file, ext="tsv")
 
        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d Documents.", len(self.corpus))

        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()

        if os.path.exists(self.qrels_file):
            self._load_qrels()
            self.queries = {qid: self.queries[qid] for qid in self.qrels}
            logger.info("Loaded %d %s Queries.", len(self.queries), split.upper())
            logger.info("Query Example: %s", list(self.queries.values())[0])

        return self.corpus, self.queries, self.qrels
    
    def load_corpus(self) -> dict[str, dict[str, str]]:
        self.check(fIn=self.corpus_file, ext="jsonl")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d Documents.", len(self.corpus))

        return self.corpus
    

    def _load_corpus(self):
        self.corpus = StreamingCorpus(self.corpus_file)


class VaultDataLoader(GenericDataLoader):
    def __init__(
        self,
        dataset_name: str = None,
        data_folder: str = "/Tmp/lvpoellhuber/datasets/vault", #os.getenv("VAULT_DIR"), # TODO: Export VAULT dir to environment vars
        corpus_file: str = "corpus.jsonl",
        query_file: str = "queries.jsonl",
        qrels_folder: str = "qrels",
        qrels_file: str = "",
    ):
        self.corpus = {}
        self.queries = {}
        self.qrels = {}

        self.datasets = [
            "wikir",
            "hotpotqa",
            "nq",
            "scidocs",
            "doris-mae",
            "nfcorpus", 
            "cord19"
        ]
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} does not exist in VAULT. Please select a dataset from: wikir, hotpotqa, nq, scidocs, doris-mae, nfcorpus, cord19, bioasq.")

        if dataset_name:
            dataset_path = os.path.join(data_folder, dataset_name)

            # NQ and HotPotQA rely on a common Wikipedia corpus, instead of having separate corpuses. 
            if dataset_name=="nq" or dataset_name=="hotpotqa":
                corpus_path = os.path.join(data_folder, "wikipedia")
            else:
                corpus_path = dataset_path

            # Figure out if the corpus file's extension is JSON or DB
            for file in os.listdir(corpus_path):
                if file.endswith(("corpus.jsonl", "corpus.db")):
                    self.corpus_file = os.path.join(corpus_path, file)
            if corpus_file == None:
                raise ValueError(f"No corpus file found. Please run preprocessing scripts.")

            self.streamed = self.corpus_file.endswith("corpus.db")

            self.query_file = os.path.join(dataset_path, query_file) 
            self.qrels_folder = os.path.join(dataset_path, qrels_folder) 
        else:
            self.corpus_file = corpus_file
            self.query_file = query_file
            self.qrels_folder = None

        self.qrels_file = qrels_file

    def load(self, split="test") -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, int]]]:
        self.qrels_file = os.path.join(self.qrels_folder, split + ".tsv")
        
        if self.streamed:
            self.dataloader = StreamedDataLoader(
                corpus_file=self.corpus_file, 
                query_file=self.query_file, 
                qrels_file=self.qrels_file
            )
        else:
            self.dataloader = GenericDataLoader(
                corpus_file=self.corpus_file, 
                query_file=self.query_file, 
                qrels_file=self.qrels_file
            )

        return self.dataloader.load(split)
    
    
class DatasetForFineTuning(torch.utils.data.Dataset):
    def __init__(self, args):
        
        def load_jsonl(file_path):
            d={}
            with open(file_path,encoding="utf-8")as df:
                for line in df:
                    query=json.loads(line)
                    d[query['_id']]=query
            return d

        self.streaming = args.streaming
        self.id2query=load_jsonl(args.query_file)
        if self.streaming: 
            self.streaming_corpus=StreamingCorpus(args.corpus_file)
        else:
            self.id2corpus=load_jsonl(args.corpus_file)
        self.dataset=open(args.qrels_file,encoding="utf-8").readlines()[1:]
        original_dataset_len=len(self.dataset)
        if args.min_corpus_len>0:
            if self.streaming:
                raise Exception("Cannot filter corpus with streaming corpus.")
            self.dataset=[line for line in self.dataset if len(self.id2corpus[line.split('\t')[1]].get("text"," ").split(" "))+len(self.id2corpus[line.split('\t')[1]].get("title"," ").split(" "))>=args.min_corpus_len]
            log_message(f"Filtered corpus with min length {args.min_corpus_len}, {len(self.dataset)} samples left, {original_dataset_len} originally")

    def __getitem__(self, item):
        query_id, corpus_id, score=self.dataset[item].split('\t')
        if self.streaming:
            item = self.streaming_corpus.__getitem__(corpus_id)
            if item is not None:
                corpus_title_str = item.get("title","")
                corpus_text_str = item.get("text","")
                query_str=self.id2query[query_id].get("text","")
                corpus_str=corpus_title_str+' '+corpus_text_str if len(corpus_title_str)>0 else corpus_text_str
                return [query_str,corpus_str]
            else:
                return ["", ""]
        else:
            if corpus_id in self.id2corpus:
                query_str=self.id2query[query_id].get("text","")
                corpus_title_str=self.id2corpus[corpus_id].get("title","")
                corpus_text_str=self.id2corpus[corpus_id].get("text","")
                corpus_str=corpus_title_str+' '+corpus_text_str if len(corpus_title_str)>0 else corpus_text_str
                return [query_str,corpus_str]
            else:
                return ["", ""]
            


    def __len__(self):
        return len(self.dataset)


class DatasetForFineTuningNegatives(DatasetForFineTuning):
    def __init__(self, args):
        super().__init__(args)
        self.positives=open(args.qrels_file,encoding="utf-8").readlines()[1:]
        self.negatives=open(args.nqrels_file,encoding="utf-8").readlines()[1:]

        if args.min_corpus_len>=0:
            log_message(f"Cannot filter corpus with negatives.", logging.ERROR)
            
    def __getitem__(self, item):
        query_id, positive_id, score=self.positives[item].split('\t')
        _query_id, negative_id, score=self.negatives[item].split('\t')
        assert query_id==_query_id
        
        if (positive_id in self.id2corpus) and (negative_id in self.id2corpus):
            query_str=self.id2query[query_id].get("text","")

            positive_title_str=self.id2corpus[positive_id].get("title","")
            positive_text_str=self.id2corpus[positive_id].get("text","")
            positive_str=positive_title_str+' '+positive_text_str if len(positive_title_str)>0 else positive_text_str

            negative_title_str=self.id2corpus[negative_id].get("title","")
            negative_text_str=self.id2corpus[negative_id].get("text","")
            negative_str=negative_title_str+' '+negative_text_str if len(negative_title_str)>0 else negative_text_str

            return [query_str,positive_str,negative_str]
        else:
            return ["", ""]

    def __len__(self):
        return len(self.positives)

@dataclass
class DataCollatorForFineTuningLongtriever:
    tokenizer:PreTrainedTokenizerBase
    max_query_length:int
    max_corpus_length:int
    max_corpus_sent_num:int
    align_right:bool=False
    negatives:bool=False
    def __post_init__(self):
        if isinstance(self.tokenizer,str):
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer)
        elif isinstance(self.tokenizer,PreTrainedTokenizerBase):
            pass
        else:
            raise TypeError

    def tokenize(self,string):
        sentences = nltk.sent_tokenize(string)
        if not sentences:
            sentences = ["."]
        results = self.tokenizer(sentences, add_special_tokens=False, truncation=False, return_attention_mask=False,
                                 return_token_type_ids=False, verbose=False)

        block_len = self.max_corpus_length - self.tokenizer.num_special_tokens_to_add(False)
        input_ids_blocks = []
        attention_mask_blocks = []
        curr_block = []
        for input_ids_sent in results['input_ids']:
            if len(curr_block) + len(input_ids_sent) >= block_len and curr_block:
                input_ids_blocks.append(
                    torch.tensor(self.tokenizer.build_inputs_with_special_tokens(curr_block[:block_len])))
                attention_mask_blocks.append(torch.tensor([1] * len(input_ids_blocks[-1])))
                curr_block = []
                if len(input_ids_blocks) >= self.max_corpus_sent_num:
                    break
            curr_block.extend(input_ids_sent)
        if len(curr_block) > 0:
            input_ids_blocks.append(
                torch.tensor(self.tokenizer.build_inputs_with_special_tokens(curr_block[:block_len])))
            attention_mask_blocks.append(torch.tensor([1] * len(input_ids_blocks[-1])))
        input_ids_blocks = tensorize_batch(input_ids_blocks, self.tokenizer.pad_token_id, align_right=self.align_right)
        attention_mask_blocks = tensorize_batch(attention_mask_blocks, 0, align_right=self.align_right)
        return {
            "input_ids_blocks": input_ids_blocks,
            "attention_mask_blocks": attention_mask_blocks,
        }

    def __call__(self, examples):
        # tic = time.time()
        query_input_ids_batch = []
        query_attention_mask_batch = []
        corpus_input_ids_batch = []
        corpus_attention_mask_batch = []
        
        if self.negatives:
            negative_input_ids_batch = []
            negative_attention_mask_batch = []

        for e in examples:
            if self.negatives:
                query_str, corpus_str, negative_str=e

                negative_results=self.tokenize(negative_str)
                negative_input_ids_batch.append(negative_results['input_ids_blocks'])
                negative_attention_mask_batch.append(negative_results['attention_mask_blocks'])

            else:
                query_str, corpus_str=e

            query_results=self.tokenize(query_str)
            query_input_ids_batch.append(query_results['input_ids_blocks'])
            query_attention_mask_batch.append(query_results['attention_mask_blocks'])

            corpus_resutls=self.tokenize(corpus_str)
            corpus_input_ids_batch.append(corpus_resutls['input_ids_blocks'])
            corpus_attention_mask_batch.append(corpus_resutls['attention_mask_blocks'])

        query_input_ids_batch = tensorize_batch(query_input_ids_batch, self.tokenizer.pad_token_id, align_right=self.align_right)  # [B,N,L]
        query_attention_mask_batch = tensorize_batch(query_attention_mask_batch, 0, align_right=self.align_right)  # [B,N,L]
        corpus_input_ids_batch=tensorize_batch(corpus_input_ids_batch,self.tokenizer.pad_token_id, align_right=self.align_right) #[B,N,L]
        corpus_attention_mask_batch=tensorize_batch(corpus_attention_mask_batch,0, align_right=self.align_right) #[B,N,L]

        # if corpus_input_ids_batch.shape[2] == 512:
        #     pass

        if self.negatives:
            negative_input_ids_batch=tensorize_batch(negative_input_ids_batch,self.tokenizer.pad_token_id, align_right=self.align_right) #[B,N,L]
            negative_attention_mask_batch=tensorize_batch(negative_attention_mask_batch,0, align_right=self.align_right) #[B,N,L]

            batch = {
                "query_input_ids": query_input_ids_batch, #[B,N,L]
                "query_attention_mask": query_attention_mask_batch, #[B,N,L]
                "corpus_input_ids": corpus_input_ids_batch, #[B,N,L]
                "corpus_attention_mask": corpus_attention_mask_batch, #[B,N,L]
                "neg_input_ids": negative_input_ids_batch, #[B,N,L]
                "neg_attention_mask": negative_attention_mask_batch, #[B,N,L]
            }
        else:
            batch = {
                "query_input_ids": query_input_ids_batch, #[B,N,L]
                "query_attention_mask": query_attention_mask_batch, #[B,N,L]
                "corpus_input_ids": corpus_input_ids_batch, #[B,N,L]
                "corpus_attention_mask": corpus_attention_mask_batch, #[B,N,L]
            }
        
        return batch
    
    
@dataclass
class DataCollatorForFineTuningHierarchicalLongtriever:
    tokenizer:PreTrainedTokenizerBase
    max_query_length:int
    max_corpus_length:int
    max_corpus_sent_num:int
    align_right:bool=False
    negatives:bool=False
    start_separator:bool=False
    text_separator:bool=True
    end_separator:bool=False
    def __post_init__(self):
        if isinstance(self.tokenizer,str):
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer)
        elif isinstance(self.tokenizer,PreTrainedTokenizerBase):
            pass
        else:
            raise TypeError
     
    def tokenize(self,string):
        sentences = nltk.sent_tokenize(string)
        if not sentences:
            sentences = ["."]
        results = self.tokenizer(sentences, add_special_tokens=False, truncation=False, return_attention_mask=False,
                                 return_token_type_ids=False, verbose=False)

        num_special_tokens = 1 + self.start_separator + self.text_separator + self.end_separator
        block_len = self.max_corpus_length - num_special_tokens - (self.max_corpus_sent_num - 1)
        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id
        
        input_ids_blocks = []
        attention_mask_blocks = []
        curr_block = []

        for input_ids_sent in results['input_ids']:

            if len(curr_block) + len(input_ids_sent) >= block_len and curr_block:
                # Start with CLS token
                pre_tokens = [cls_token_id] 
                if self.start_separator: # Add a separator at the beginning if True
                    pre_tokens += [sep_token_id] 
                block_input_ids = pre_tokens + curr_block[:block_len]
                if self.text_separator: # Add a separator at the end of the text if True
                    block_input_ids += [sep_token_id]
                block_input_ids = torch.tensor(block_input_ids)

                input_ids_blocks.append(block_input_ids)
                attention_mask_blocks.append(torch.tensor([1] * len(block_input_ids))) # To account for the extra sep token I'll add at the end
                
                curr_block = []
                if len(input_ids_blocks) >= self.max_corpus_sent_num:
                    break
            curr_block.extend(input_ids_sent)

        if len(curr_block) > 0:
            pre_tokens = [cls_token_id] 
            if self.start_separator: # Add a separator at the beginning if True
                pre_tokens += [sep_token_id] 
            block_input_ids = pre_tokens + curr_block[:block_len]
            if self.text_separator: # Add a separator at the end of the text if True
                block_input_ids += [sep_token_id]
            block_input_ids = torch.tensor(block_input_ids)  

            input_ids_blocks.append(block_input_ids)
            attention_mask_blocks.append(torch.tensor([1] * len(block_input_ids)))
        
        input_ids_blocks = tensorize_batch(input_ids_blocks, self.tokenizer.pad_token_id, align_right=self.align_right)
        attention_mask_blocks = tensorize_batch(attention_mask_blocks, 0, align_right=self.align_right)

        return {
            "input_ids_blocks": input_ids_blocks,
            "attention_mask_blocks": attention_mask_blocks,
        }

    def __call__(self, examples):
        # tic = time.time()
        query_input_ids_batch = []
        query_attention_mask_batch = []
        corpus_input_ids_batch = []
        corpus_attention_mask_batch = []
        
        if self.negatives:
            negative_input_ids_batch = []
            negative_attention_mask_batch = []
        
        sep_token_id = self.tokenizer.sep_token_id


        for e in examples:
            if self.negatives:
                query_str, corpus_str, negative_str=e

                negative_results=self.tokenize(negative_str)
                negative_input_ids_batch.append(negative_results['input_ids_blocks'])
                negative_attention_mask_batch.append(negative_results['attention_mask_blocks'])

            else:
                query_str, corpus_str=e

            query_results=self.tokenize(query_str)
            query_input_ids_batch.append(query_results['input_ids_blocks'])
            query_attention_mask_batch.append(query_results['attention_mask_blocks'])

            corpus_resutls=self.tokenize(corpus_str)
            corpus_input_ids_batch.append(corpus_resutls['input_ids_blocks'])
            corpus_attention_mask_batch.append(corpus_resutls['attention_mask_blocks'])

        query_input_ids_batch = tensorize_batch(query_input_ids_batch, self.tokenizer.pad_token_id, align_right=self.align_right)  # [B,N,L]
        query_attention_mask_batch = tensorize_batch(query_attention_mask_batch, 0, align_right=self.align_right)  # [B,N,L]
        corpus_input_ids_batch=tensorize_batch(corpus_input_ids_batch,self.tokenizer.pad_token_id, align_right=self.align_right) #[B,N,L]
        corpus_attention_mask_batch=tensorize_batch(corpus_attention_mask_batch,0, align_right=self.align_right) #[B,N,L]
        
        if self.end_separator:
            query_final_sep_token = torch.tensor([sep_token_id]).repeat(query_input_ids_batch.shape[0], query_input_ids_batch.shape[1], 1)
            query_input_ids_batch = torch.cat([query_input_ids_batch, query_final_sep_token], dim=2) # Add the last sep token
            query_final_sep_token_mask= query_attention_mask_batch[:, :, :1]
            query_attention_mask_batch = torch.cat([query_attention_mask_batch, query_final_sep_token_mask], dim=2) # Add the last sep token
            
            corpus_final_sep_token = torch.tensor([sep_token_id]).repeat(corpus_input_ids_batch.shape[0], corpus_input_ids_batch.shape[1], 1)
            corpus_input_ids_batch = torch.cat([corpus_input_ids_batch, corpus_final_sep_token], dim=2) # Add the last sep token
            corpus_final_sep_token_mask= corpus_attention_mask_batch[:, :, :1]
            corpus_attention_mask_batch = torch.cat([corpus_attention_mask_batch, corpus_final_sep_token_mask], dim=2) # Add the last sep token
    
        
        if self.negatives:
            negative_input_ids_batch=tensorize_batch(negative_input_ids_batch,self.tokenizer.pad_token_id, align_right=self.align_right) #[B,N,L]
            negative_attention_mask_batch=tensorize_batch(negative_attention_mask_batch,0, align_right=self.align_right) #[B,N,L]
            
            if self.end_separator:
                negative_final_sep_token = torch.tensor([sep_token_id]).repeat(negative_input_ids_batch.shape[0], negative_input_ids_batch.shape[1], 1)
                negative_input_ids_batch = torch.cat([negative_input_ids_batch, negative_final_sep_token], dim=2) # Add the last sep token
                negative_final_sep_token_mask= negative_attention_mask_batch[:, :, :1]
                negative_attention_mask_batch = torch.cat([negative_attention_mask_batch, negative_final_sep_token_mask], dim=2) # Add the last sep token
            
            batch = {
                "query_input_ids": query_input_ids_batch, #[B,N,L]
                "query_attention_mask": query_attention_mask_batch, #[B,N,L]
                "corpus_input_ids": corpus_input_ids_batch, #[B,N,L]
                "corpus_attention_mask": corpus_attention_mask_batch, #[B,N,L]
                "neg_input_ids": negative_input_ids_batch, #[B,N,L]
                "neg_attention_mask": negative_attention_mask_batch, #[B,N,L]
            }
        else:
            batch = {
                "query_input_ids": query_input_ids_batch, #[B,N,L]
                "query_attention_mask": query_attention_mask_batch, #[B,N,L]
                "corpus_input_ids": corpus_input_ids_batch, #[B,N,L]
                "corpus_attention_mask": corpus_attention_mask_batch, #[B,N,L]
            }
        
       
        return batch
    
    
@dataclass
class DataCollatorForFineTuningBert:
    tokenizer:PreTrainedTokenizerBase
    max_query_length:int
    max_corpus_length:int
    align_right:bool=False
    def __post_init__(self):
        if isinstance(self.tokenizer,str):
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer)
        elif isinstance(self.tokenizer,PreTrainedTokenizerBase):
            pass
        else:
            raise TypeError

    def __call__(self, examples):      
        queries = [e[0] for e in examples]
        corpus = [e[1] for e in examples]
        tokenized_queries = self.tokenizer(
                queries, 
                padding=True,
                truncation=True,
                max_length=self.max_query_length, 
                return_tensors="pt",
            )
        tokenized_corpus = self.tokenizer(
                corpus, 
                padding=True,
                truncation=True,
                max_length=self.max_corpus_length, 
                return_tensors="pt",
            )
        batch = {
            "query_input_ids": tokenized_queries["input_ids"], #[B,N,L]
            "query_attention_mask": tokenized_queries["attention_mask"], #[B,N,L]
            "corpus_input_ids": tokenized_corpus["input_ids"], #[B,N,L]
            "corpus_attention_mask": tokenized_corpus["attention_mask"], #[B,N,L]
        }
        
        return batch
    
@dataclass
class DataCollatorForEvaluatingLongtriever:
    tokenizer:PreTrainedTokenizerBase
    max_query_length:int
    max_corpus_length:int
    max_corpus_sent_num:int
    align_right:bool=False
    def __post_init__(self):
        if isinstance(self.tokenizer,str):
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer)
        elif isinstance(self.tokenizer,PreTrainedTokenizerBase):
            pass
        else:
            raise TypeError

    def tokenize(self,string):
        sentences = nltk.sent_tokenize(string)
        if not sentences:
            sentences = ["."]
        results = self.tokenizer(sentences, add_special_tokens=False, truncation=False, return_attention_mask=False,
                                 return_token_type_ids=False, verbose=False)

        block_len = self.max_corpus_length - self.tokenizer.num_special_tokens_to_add(False)
        input_ids_blocks = []
        attention_mask_blocks = []
        curr_block = []
        for input_ids_sent in results['input_ids']:
            if len(curr_block) + len(input_ids_sent) >= block_len and curr_block:
                input_ids_blocks.append(
                    torch.tensor(self.tokenizer.build_inputs_with_special_tokens(curr_block[:block_len])))
                attention_mask_blocks.append(torch.tensor([1] * len(input_ids_blocks[-1])))
                curr_block = []
                if len(input_ids_blocks) >= self.max_corpus_sent_num:
                    break
            curr_block.extend(input_ids_sent)
        if len(curr_block) > 0:
            input_ids_blocks.append(
                torch.tensor(self.tokenizer.build_inputs_with_special_tokens(curr_block[:block_len])))
            attention_mask_blocks.append(torch.tensor([1] * len(input_ids_blocks[-1])))
        input_ids_blocks = tensorize_batch(input_ids_blocks, self.tokenizer.pad_token_id, align_right=self.align_right)
        attention_mask_blocks = tensorize_batch(attention_mask_blocks, 0, align_right=self.align_right)
        return {
            "input_ids_blocks": input_ids_blocks,
            "attention_mask_blocks": attention_mask_blocks,
        }

    def __call__(self, examples):
        input_ids_batch = []
        attention_mask_batch = []
        for e in examples:
            results=self.tokenize(e)
            input_ids_batch.append(results['input_ids_blocks'])
            attention_mask_batch.append(results['attention_mask_blocks'])

        input_ids_batch=tensorize_batch(input_ids_batch,self.tokenizer.pad_token_id, align_right=self.align_right) #[B,N,L]
        attention_mask_batch=tensorize_batch(attention_mask_batch,0, align_right=self.align_right) #[B,N,L]


        batch = {
            "input_ids": input_ids_batch, #[B,N,L]
            "attention_mask": attention_mask_batch, #[B,N,L]
        }

        return batch
    
    
@dataclass
class DataCollatorForEvaluatingHierarchicalLongtriever:
    tokenizer:PreTrainedTokenizerBase
    max_query_length:int
    max_corpus_length:int
    max_corpus_sent_num:int
    align_right:bool=False
    start_separator:bool=False
    text_separator:bool=True
    end_separator:bool=False
    def __post_init__(self):
        if isinstance(self.tokenizer,str):
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer)
        elif isinstance(self.tokenizer,PreTrainedTokenizerBase):
            pass
        else:
            raise TypeError

    def tokenize(self,string):
        sentences = nltk.sent_tokenize(string)
        if not sentences:
            sentences = ["."]
        results = self.tokenizer(sentences, add_special_tokens=False, truncation=False, return_attention_mask=False,
                                 return_token_type_ids=False, verbose=False)

        num_special_tokens = 1 + self.start_separator + self.text_separator + self.end_separator
        block_len = self.max_corpus_length - num_special_tokens - (self.max_corpus_sent_num - 1)
        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id
        
        input_ids_blocks = []
        attention_mask_blocks = []
        curr_block = []

        for input_ids_sent in results['input_ids']:

            if len(curr_block) + len(input_ids_sent) >= block_len and curr_block:
                # Start with CLS token
                pre_tokens = [cls_token_id] 
                if self.start_separator: # Add a separator at the beginning if True
                    pre_tokens += [sep_token_id] 
                block_input_ids = pre_tokens + curr_block[:block_len]
                if self.text_separator: # Add a separator at the end of the text if True
                    block_input_ids += [sep_token_id]
                block_input_ids = torch.tensor(block_input_ids)

                input_ids_blocks.append(block_input_ids)
                attention_mask_blocks.append(torch.tensor([1] * len(block_input_ids))) # To account for the extra sep token I'll add at the end
                
                curr_block = []
                if len(input_ids_blocks) >= self.max_corpus_sent_num:
                    break
            curr_block.extend(input_ids_sent)

        if len(curr_block) > 0:
            pre_tokens = [cls_token_id] 
            if self.start_separator: # Add a separator at the beginning if True
                pre_tokens += [sep_token_id] 
            block_input_ids = pre_tokens + curr_block[:block_len]
            if self.text_separator: # Add a separator at the end of the text if True
                block_input_ids += [sep_token_id]
            block_input_ids = torch.tensor(block_input_ids)  

            input_ids_blocks.append(block_input_ids)
            attention_mask_blocks.append(torch.tensor([1] * len(block_input_ids)))
        
        input_ids_blocks = tensorize_batch(input_ids_blocks, self.tokenizer.pad_token_id, align_right=self.align_right)
        attention_mask_blocks = tensorize_batch(attention_mask_blocks, 0, align_right=self.align_right)

        return {
            "input_ids_blocks": input_ids_blocks,
            "attention_mask_blocks": attention_mask_blocks,
        }

    def __call__(self, examples):
        sep_token_id = self.tokenizer.sep_token_id

        input_ids_batch = []
        attention_mask_batch = []
        for e in examples:
            results=self.tokenize(e)
            input_ids_batch.append(results['input_ids_blocks'])
            attention_mask_batch.append(results['attention_mask_blocks'])

        input_ids_batch=tensorize_batch(input_ids_batch,self.tokenizer.pad_token_id, align_right=self.align_right) #[B,N,L]
        attention_mask_batch=tensorize_batch(attention_mask_batch,0, align_right=self.align_right) #[B,N,L]

        
        if self.end_separator:
            final_sep_token = torch.tensor([sep_token_id]).repeat(input_ids_batch.shape[0], input_ids_batch.shape[1], 1)
            input_ids_batch = torch.cat([input_ids_batch, final_sep_token], dim=2) # Add the last sep token
            final_sep_token_mask= attention_mask_batch[:, :, :1]
            attention_mask_batch = torch.cat([attention_mask_batch, final_sep_token_mask], dim=2) # Add the last sep token

        batch = {
            "input_ids": input_ids_batch, #[B,N,L]
            "attention_mask": attention_mask_batch, #[B,N,L]
        }

        return batch
    
    
@dataclass
class DataCollatorForEvaluatingBert:
    tokenizer:PreTrainedTokenizerBase
    max_query_length:int
    max_corpus_length:int
    align_right:bool=False
    def __post_init__(self):
        if isinstance(self.tokenizer,str):
            self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer)
        elif isinstance(self.tokenizer,PreTrainedTokenizerBase):
            pass
        else:
            raise TypeError

    def __call__(self, examples):      
        tokenized_examples = self.tokenizer(
                examples, 
                padding=True,
                truncation=True,
                max_length=self.max_query_length, 
                return_tensors="pt",
            )
        batch = {
            "input_ids": tokenized_examples["input_ids"],
            "attention_mask": tokenized_examples["attention_mask"]
        }
        
        return batch
    