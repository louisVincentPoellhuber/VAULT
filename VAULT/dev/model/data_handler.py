from .modeling_utils import *
import os
import json
from dataclasses import dataclass
import torch.utils.data.dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase, DataCollatorForWholeWordMask
from datasets import load_dataset,concatenate_datasets,load_from_disk
from .modeling_utils import tensorize_batch
import nltk
import sqlite3
from beir.datasets.data_loader import GenericDataLoader
from typing import Union, List
import faiss
import dotenv
dotenv.load_dotenv()

nltk.download('punkt')

if JOBID == None: JOBID = "debug"
logger = logging.getLogger(__name__)


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
        self.cursor.execute("SELECT id, title, text, url FROM articles WHERE id = ?", (page_id,))
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
        data_folder: str = None, 
        corpus_file: str = "corpus.jsonl",
        query_file: str = "queries.jsonl",
        qrels_folder: str = "qrels",
        qrels_file: str = None,
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
            "cord19", 
            "wikipedia"
        ]

        data_folder = os.getenv("VAULT_DIR")
        if data_folder is None:
            raise ValueError("Please set the VAULT_DIR environment variable to the root folder of the VAULT datasets.")

        if dataset_name:
            if dataset_name.split("_")[0] not in self.datasets:
                raise ValueError(f"Dataset {dataset_name} does not exist in VAULT. Please select a dataset from: wikir, hotpotqa, nq, scidocs, doris-mae, nfcorpus, cord19, bioasq.")
            
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

            self.query_file = os.path.join(dataset_path, query_file) 
            self.qrels_folder = os.path.join(dataset_path, qrels_folder) 
        else:
            self.corpus_file = corpus_file
            self.query_file = query_file
            self.qrels_folder = None

        self.streamed = self.corpus_file.endswith("corpus.db")
        self.qrels_file = qrels_file

    def load(self, split="test") -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, int]]]:
        if self.qrels_file is None:
            self.qrels_file = os.path.join(self.qrels_folder, split + ".tsv")
        
        if self.streamed:
            self.dataloader = StreamedDataLoader(
                corpus_file=self.corpus_file, 
                query_file=self.query_file, 
                qrels_file=self.qrels_file
            )
            return self.dataloader.load(split)
        else:
            self.dataloader = GenericDataLoader(
                corpus_file=self.corpus_file, 
                query_file=self.query_file, 
                qrels_file=self.qrels_file
            )

            return self.dataloader.load_custom()
           
    
    
class DatasetForFineTuning(torch.utils.data.Dataset):
    def __init__(self, args):
        
        def load_jsonl(file_path):
            d={}
            with open(file_path,encoding="utf-8")as df:
                for line in df:
                    query=json.loads(line)
                    d[query['_id']]=query
            return d

        def get_id_mappings(mapping_file):
            id2faiss = {}
            with open(mapping_file, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i>0:
                        docid, faiss_id = line.strip().split("\t", 1)
                        base_id, section_id = docid.split("-")
                        if base_id not in id2faiss:
                            id2faiss[base_id] = []
                        id2faiss[base_id].append(int(faiss_id))
            return id2faiss

        self.id2query=load_jsonl(args.query_file)
        self.index = faiss.read_index(args.index_dir+"/default.flat.faiss")
        self.id2faiss = get_id_mappings(args.index_dir+"/default.flat.tsv")
        self.dataset=open(args.qrels_file,encoding="utf-8").readlines()[1:]
   
    def __getitem__(self, item):
        query_id, corpus_id, score=self.dataset[item].split('\t')
        if corpus_id in self.id2faiss:
            corpus_embeds = []
            for faiss_id in self.id2faiss[corpus_id]:
                faiss_vec = self.index.reconstruct(faiss_id)
                faiss_vec = torch.tensor(faiss_vec)
                corpus_embeds.append(faiss_vec)
            query_str=self.id2query[query_id].get("text","")
            return [query_str,corpus_embeds]
        else:
            return ["", torch.rand((1, 768))]
            


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
        corpus_embeds_batch = []
        
        for e in examples:
            query_str, corpus_embeds=e
            corpus_embeds_batch.append(tensorize_batch(corpus_embeds, 0))

            query_results=self.tokenize(query_str)
            query_input_ids_batch.append(query_results['input_ids_blocks'])
            query_attention_mask_batch.append(query_results['attention_mask_blocks'])

            dummy_input = "[PAD] " * 512 
            corpus_results=self.tokenize(dummy_input)
            input_ids = corpus_results['input_ids_blocks'].repeat(len(corpus_embeds), 1)
            corpus_input_ids_batch.append(input_ids)
            attention_mask = torch.zeros_like(corpus_results["attention_mask_blocks"])
            attention_mask[:, 0] = 1 
            attention_mask[:, -1] = 1
            attention_mask = attention_mask.repeat(len(corpus_embeds), 1)
            corpus_attention_mask_batch.append(attention_mask)

        query_input_ids_batch = tensorize_batch(query_input_ids_batch, self.tokenizer.pad_token_id, align_right=self.align_right)  # [B,N,L]
        query_attention_mask_batch = tensorize_batch(query_attention_mask_batch, 0, align_right=self.align_right)  # [B,N,L]
        corpus_input_ids_batch=tensorize_batch(corpus_input_ids_batch,self.tokenizer.pad_token_id, align_right=self.align_right) #[B,N,L]
        corpus_attention_mask_batch=tensorize_batch(corpus_attention_mask_batch,0, align_right=self.align_right) #[B,N,L]
        corpus_embeds_batch=tensorize_batch(corpus_embeds_batch,0) #[B,D]

        batch = {
            "query_input_ids": query_input_ids_batch, #[B,N,L]
            "query_attention_mask": query_attention_mask_batch, #[B,N,L]
            "corpus_input_ids": corpus_input_ids_batch, #[B,N,L]
            "corpus_embeds": corpus_embeds_batch, #[B, D]
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
     
    def tokenize(self,string, block_len):
        sentences = nltk.sent_tokenize(string)
        if not sentences:
            sentences = ["."]
        results = self.tokenizer(sentences, add_special_tokens=False, truncation=False, return_attention_mask=False,
                                 return_token_type_ids=False, verbose=False)

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
    
    def blockify_inputs(self,embeds):
        num_special_tokens = 2 # CLS and SEP
        nb_blocks = min(len(embeds)//self.max_corpus_length+1, self.max_corpus_sent_num) # This time I KNOW how many "words" there are : there is no tokenization here
        block_len = min(self.max_corpus_length - num_special_tokens - (nb_blocks - 1), len(embeds)) # Maximum heuristic, OR just the number of input tokens, if it's small enough
        attention_mask_blocks = []

        block_embeds = []
        for i in range(0, len(embeds), block_len):
            curr_block_embeds = embeds[i:i+block_len]
            curr_block = tensorize_batch(curr_block_embeds, 0)
            block_embeds.append(curr_block)
            attention_mask_blocks.append(torch.tensor([1] * (len(curr_block_embeds)+2)))
            if len(block_embeds) >= self.max_corpus_sent_num:
                break

        return {
            "block_embeds": block_embeds,
            "attention_mask_blocks": attention_mask_blocks,
            "block_len": block_len,
            "nb_blocks": nb_blocks
        }

    def __call__(self, examples):
        # tic = time.time()
        query_input_ids_batch = []
        query_attention_mask_batch = []
        corpus_input_ids_batch = []
        corpus_attention_mask_batch = []
        corpus_embeds_batch = []
        
        for e in examples:
            query_str, corpus_embeds=e

            blockify_results=self.blockify_inputs(corpus_embeds)
            corpus_embeds_batch.append(tensorize_batch(blockify_results["block_embeds"], 0))
            attention_mask = tensorize_batch(blockify_results["attention_mask_blocks"], 0)
            corpus_attention_mask_batch.append(attention_mask)
            block_len = blockify_results["block_len"]
            nb_blocks = blockify_results["nb_blocks"]

            query_results=self.tokenize(query_str, block_len)
            query_input_ids_batch.append(query_results['input_ids_blocks'])
            query_attention_mask_batch.append(query_results['attention_mask_blocks'])

            dummy_input = "[PAD] " * len(corpus_embeds)
            corpus_results=self.tokenize(dummy_input, block_len)
            corpus_input_ids = corpus_results['input_ids_blocks'].repeat(nb_blocks, 1)
            corpus_input_ids_batch.append(corpus_input_ids)          



        query_input_ids_batch = tensorize_batch(query_input_ids_batch, self.tokenizer.pad_token_id, align_right=self.align_right)  # [B,N,L]
        query_attention_mask_batch = tensorize_batch(query_attention_mask_batch, 0, align_right=self.align_right)  # [B,N,L]
        corpus_input_ids_batch=tensorize_batch(corpus_input_ids_batch,self.tokenizer.pad_token_id, align_right=self.align_right) #[B,N,L]
        corpus_attention_mask_batch=tensorize_batch(corpus_attention_mask_batch,0, align_right=self.align_right) #[B,N,L]
        corpus_embeds=tensorize_batch(corpus_embeds_batch,0) #[B,D]

        batch = {
            "query_input_ids": query_input_ids_batch, #[B,N,L]
            "query_attention_mask": query_attention_mask_batch, #[B,N,L]
            "corpus_input_ids": corpus_input_ids_batch, #[B,N,L]
            "corpus_embeds": corpus_embeds, #[B, D]
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
    