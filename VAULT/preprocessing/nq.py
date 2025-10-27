from utils import *
from wikipedia import *
import nltk

class NQProcessor(DatasetProcessor):
    def __init__(self, datapath, overwrite, suffix=""):
        super().__init__(datapath, "nq"+suffix, overwrite)
        self.wikipedia_processor = WikipediaProcessor(datapath, overwrite)
        self.download_dir = self.wikipedia_processor.download_dir
        self.wikipedia_dir = self.wikipedia_processor.dataset_dir
        self.subsets = ["train", "dev"]
        self.norm_subset_name = {
                "train": "train",
                "dev": "test"
            }
        log_message("The NQ uses 'train' and 'dev' split. These will be renamed to 'train' and 'test' respectively for consistency.", print_message=True)
     
    def download(self):
        ir_datasets.load("natural-questions")

    
    def _create_id_mappings(self):
        id_mappings_path = os.path.join(self.dataset_dir, "id_mappings.json")

        if os.path.exists(id_mappings_path) and not self.overwrite:
            with open(id_mappings_path, "r", encoding="utf-8") as f:
                id_mappings = json.load(f)
        else:
            id_mappings = {}
            missing_docs = []
            title_to_id = {}

            wikipedia_db_path = os.path.join(self.wikipedia_dir, "corpus.db")
            connection = sqlite3.connect(wikipedia_db_path)
            cursor = connection.cursor()

            # cursor.execute("SELECT id, title FROM articles")
            
            for article_id, title in tqdm(cursor.execute("SELECT id, title FROM articles"), desc="Collecting all ID-Title pairs from Wikipedia database."):
                title_to_id[title] = article_id

            dataset = ir_datasets.load(f"natural-questions")
            for doc in tqdm(dataset.docs_iter(), desc="Creating ID mappings for NQ documents."):
                if doc.document_title in title_to_id:
                    nq_docid = doc.doc_id.split("-")[0]
                    if nq_docid not in id_mappings:
                        id_mappings[nq_docid] = title_to_id[doc.document_title]
                else:
                    missing_docs.append(doc.doc_id.split("-")[0])

            missing_docs = list(set(missing_docs))
            with open(os.path.join(self.dataset_dir, "missing_docs.txt"), "w", encoding="utf-8") as f:
                for doc_id in missing_docs:
                    f.write(f"{doc_id}\n")
            with open(id_mappings_path, "w", encoding="utf-8") as f:
                json.dump(id_mappings, f, indent=4)
            
        self.id_mappings = id_mappings

    def process_corpus(self):
        # Check if Wikipedia files are there
        if len(os.listdir(self.download_dir))>1:
            self.wikipedia_processor.process_corpus()
        else:
            raise Exception(f"No extracted Wikipedia folders found at {self.download_dir}. Please run 'process_wikipedia.sh' first.")
            
        self._create_id_mappings()

        corpus_id_file = os.path.join(self.dataset_dir, "corpus_ids.csv")
        if not os.path.exists(corpus_id_file) or self.overwrite:
            dataset = ir_datasets.load("natural-questions")
            wiki_ids = set()
            with open(corpus_id_file, "w") as ids_out:
                for doc in tqdm(dataset.docs_iter()):
                    nq_id = doc.doc_id.split("-")[0]
                    if nq_id in self.id_mappings:
                        wiki_id = self.id_mappings[nq_id]
                        if wiki_id not in wiki_ids:
                            ids_out.write(str(wiki_id)+"\n")
                            wiki_ids.add(wiki_id)
        else:
            log_message(f"Corpus ID file already exists, skipping. ")


    def process_queries(self):
        queries_path = os.path.join(self.dataset_dir, "queries.jsonl")

        if self.overwrite or not os.path.exists(queries_path):
            log_message(f"Processing queries into {queries_path}.", print_message=True)
            with open(queries_path, "a", encoding="utf-8") as f:
                for subset in self.subsets:
                    dataset = ir_datasets.load(f"natural-questions/{subset}")
                    for query in dataset.queries_iter():
                        obj = {
                            "_id": query.query_id,
                            "text": query.text
                        }
                        f.write(json.dumps(obj) + "\n")
        else:
            log_message(f"Queries already exist at {queries_path}. Skipping query processing.", print_message=True)

    def process_qrels(self):
        log_message(f"Processing qrels into {self.qrel_dir}.", print_message=True)

        db_path = os.path.join(self.wikipedia_dir, "corpus.db")
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()

        for subset in self.subsets:
            dataset = ir_datasets.load(f"natural-questions/{subset}")
            subset = self.norm_subset_name[subset]
            qrel_path = os.path.join(self.qrel_dir, f"{subset}.tsv")

            if self.overwrite or not os.path.exists(qrel_path):
                with open(qrel_path, "w", encoding="utf-8") as f:
                    for qrel in dataset.qrels_iter():
                        nq_docid = qrel.doc_id.split("-")[0]
                        if nq_docid in self.id_mappings:
                            doc_id = self.id_mappings[nq_docid]
                            cursor.execute("SELECT id, title, text, url FROM articles WHERE id = ?", (doc_id,))
                            if cursor.fetchone() is not None:
                                f.write(f"{qrel.query_id}\t{doc_id}\t{qrel.relevance}\n")
            else:
                log_message(f"Qrels for {subset} already exist at {qrel_path}. Skipping qrel processing.", print_message=True)
        
        connection.close()

class NQShortProcessor(NQProcessor):
    def __init__(self, datapath, overwrite):
        super().__init__(datapath, overwrite, "_short") 

    
    def process_corpus(self):
        corpus_path = os.path.join(self.dataset_dir, "corpus.jsonl")

        if self.overwrite or not os.path.exists(corpus_path):
            with open(corpus_path, "w", encoding="utf-8") as f:
                dataset = ir_datasets.load(f"natural-questions")
                for doc in tqdm(dataset.docs_iter()):
                    doc_obj = {
                        "_id": doc.doc_id,
                        "text": doc.text, 
                        "title": doc.document_title
                    }
                    f.write(json.dumps(doc_obj) + "\n")
        else:
            log_message(f"Dataset already processed. Skipping.")
    
    def process_qrels(self):
        log_message(f"Processing qrels into {self.qrel_dir}.", print_message=True)

        for subset in self.subsets:
            dataset = ir_datasets.load(f"natural-questions/{subset}")
            qrel_path = os.path.join(self.qrel_dir, f"{self.norm_subset_name[subset]}.tsv")

            if self.overwrite or not os.path.exists(qrel_path):
                with open(qrel_path, "w", encoding="utf-8") as f:
                    for qrel in dataset.qrels_iter():
                        f.write(f"{qrel.query_id}\t{qrel.doc_id}\t{qrel.relevance}\n")
            else:
                log_message(f"Qrels for {subset} already exist at {qrel_path}. Skipping qrel processing.", print_message=True)


class NQDocumentPassageProcessor(NQProcessor):
    def __init__(self, datapath, overwrite, max_corpus_length=512, max_corpus_sent_num=8):
        super().__init__(datapath, overwrite, "_doc-passage")
        self.max_corpus_length = max_corpus_length
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # No need for complicated tokenization, it's just for chunking
        id_mappings_path = os.path.join(self.dataset_dir.replace("_doc-passage", ""), "id_mappings.json")
        with open(id_mappings_path, "r", encoding="utf-8") as f:
                self.id_mappings = json.load(f)
        
        self.corpus_ids = set(self.id_mappings.values())


    def _chunk_text(self, text, max_nb_chunks=8):
        sentences = nltk.sent_tokenize(text)
        if not sentences:
            sentences = ["."]

        # Simulate tokenization to count tokens and correctly split passages
        results = self.tokenizer(sentences, add_special_tokens=False, truncation=False, return_attention_mask=False,
                                 return_token_type_ids=False, verbose=False)

        num_special_tokens = 2  # 1 CLS + 1 SEP (text_separator)
        block_len = self.max_corpus_length - num_special_tokens 
        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id

        input_ids_blocks = []
        curr_block = []
        curr_passage = ""
        passages = []

        for i, input_ids_sent in enumerate(results['input_ids']):

            if len(curr_block) + len(input_ids_sent) >= block_len and curr_block:
                # start_separator=False, text_separator=True
                block_input_ids = [cls_token_id] + curr_block[:block_len] + [sep_token_id]
                block_input_ids = torch.tensor(block_input_ids)

                input_ids_blocks.append(block_input_ids)
                passages.append(curr_passage.strip())

                curr_block = []
                curr_passage = ""

            curr_block.extend(input_ids_sent)
            curr_passage += sentences[i]+" "

            if len(passages) >= max_nb_chunks:
                return passages

        if len(curr_block) > 0:
            # start_separator=False, text_separator=True
            block_input_ids = [cls_token_id] + curr_block[:block_len] + [sep_token_id]
            block_input_ids = torch.tensor(block_input_ids)

            input_ids_blocks.append(block_input_ids)
            passages.append(curr_passage.strip())

        return passages

    def process_corpus(self):
        corpus_path = os.path.join(self.dataset_dir, "corpus.jsonl")
        cursor = self.wikipedia_processor.cursor

        if not os.path.exists(corpus_path) or self.overwrite:
            cursor.execute("SELECT id, title, text FROM articles")
            with open(corpus_path, "w", encoding="utf-8") as f:
                for item in tqdm(cursor):
                    wiki_id = item[0]
                    if wiki_id in self.corpus_ids:
                        # if id==0:
                        #     pass
                        title = item[1]
                        text = item[2]
                        chunked_text = self._chunk_text(text, 8)
                        for i, chunk in enumerate(chunked_text):
                            doc_obj = {
                                "_id": f"{wiki_id}-{i}",
                                "title": title,
                                "text": chunk
                            }
                            f.write(json.dumps(doc_obj) + "\n")

    def process_qrels(self):
        log_message(f"Processing qrels into {self.qrel_dir}.", print_message=True)

        # Get subset of documents in QRELs
        qrel_index = {}
        total_qrels = 0
        for subset in self.subsets:
            dataset = ir_datasets.load(f"natural-questions/{subset}")
            qrel_path = os.path.join(self.qrel_dir, f"{self.norm_subset_name[subset]}.tsv")
            for qrel in dataset.qrels_iter():
                nq_id, passage_idx = qrel.doc_id.split("-")
                if nq_id not in qrel_index:
                    qrel_index[nq_id] = {}
                qrel_index[nq_id][qrel.query_id] = int(passage_idx)
                total_qrels += 1
        
        dataset = ir_datasets.load(f"natural-questions")
        nq_keys = {}
        for doc in dataset.docs_iter():
            nq_id = doc.doc_id.split("-")[0]
            if nq_id not in nq_keys:
                nq_keys[nq_id] = -1
            nq_keys[nq_id] += 1

        # # Get the text from those relevant documents
        # original_corpus = {}
        # dataset = ir_datasets.load(f"natural-questions")
        # for doc in tqdm(dataset.docs_iter()):
        #     nq_id = doc.doc_id.split("-")[0]
        #     if nq_id in qrel_index.keys():
        #         if nq_id not in original_corpus:
        #             original_corpus[nq_id] = []
        #         original_corpus[nq_id].append(doc.text)

        # # Get the answer spans from the original text
        # original_spans = {}
        # for nq_id, doc in tqdm(original_corpus.items()):
        #     if nq_id not in original_spans:
        #         original_spans[nq_id] = {}

        #     start_idx = 0
        #     for qid in qrel_index[nq_id].keys():
        #         for idx, passage in enumerate(doc):
        #             end_idx = start_idx + passage.count(" ") + 1
        #             if idx == qrel_index[nq_id][qid]:
        #                 original_spans[nq_id][qid] = (start_idx, end_idx)
        #                 break
        #             start_idx = end_idx

        # Map the original answer spans to chunked text
        corpus_path = os.path.join(self.dataset_dir, "corpus.jsonl")
        corpus = load_jsonl(corpus_path)
        corpus_keys = set(corpus.keys())
        wiki_keys = {}
        for key in corpus_keys:
            wiki_id, passage_idx = key.split("-")
            if wiki_id not in wiki_keys:
                wiki_keys[wiki_id] = -1
            wiki_keys[wiki_id]+=1


        def get_span(nq_id, qid):
            start_idx = 0
            nb_passages = nq_keys[nq_id]
            for idx in range(nb_passages+1):
                nq_passage_id = f"{nq_id}-{idx}"
                passage = dataset.docs.lookup(nq_passage_id).text 
                end_idx = start_idx + passage.count(" ") + 1
                if idx == qrel_index[nq_id][qid]:
                    return start_idx, end_idx
                start_idx = end_idx

            return -1, -1 

        for subset in self.subsets:
            dataset = ir_datasets.load(f"natural-questions/{subset}")
            qrel_path = os.path.join(self.qrel_dir, f"{self.norm_subset_name[subset]}.tsv")

            if self.overwrite or not os.path.exists(qrel_path):
                with open(qrel_path, "w", encoding="utf-8") as f:
                    for qrel in dataset.qrels_iter():
                        nq_id, passage_idx = qrel.doc_id.split("-")
                        if nq_id in self.id_mappings:
                            wiki_id = self.id_mappings[nq_id]
                            start_span, end_span = get_span(nq_id, qrel.query_id)
                            # Now map to chunked text
                            chunk_length = wiki_keys.get(str(wiki_id), 0)
                            cumulative_length = 0
                            outputted = False
                            for idx in range(chunk_length):
                                mapped_start, mapped_end = -1, -1
                                chunk_text = corpus[f"{wiki_id}-{idx}"]["text"]
                                chunk_length = len(chunk_text.split(" "))
                                if cumulative_length <= start_span < cumulative_length + chunk_length:
                                    mapped_start = start_span - cumulative_length
                                if cumulative_length < end_span <= cumulative_length + chunk_length:
                                    mapped_end = end_span - cumulative_length
                                cumulative_length += chunk_length
                                if mapped_start != -1 or mapped_end != -1:
                                    f.write(f"{qrel.query_id}\t{wiki_id}-{idx}\t{qrel.relevance}\n")
                                    outputted = True
                            if not outputted: # Fallback to first chunk
                                f.write(f"{qrel.query_id}\t{wiki_id}-0\t{qrel.relevance}\n")

            else:
                log_message(f"Qrels for {subset} already exist at {qrel_path}. Skipping qrel processing.", print_message=True)


class NQTestProcessor(NQProcessor):
    def __init__(self, datapath, overwrite):
        super().__init__(datapath, overwrite, "_test") 
        id_mappings_path = os.path.join(self.dataset_dir.replace("_test", ""), "id_mappings.json")
        with open(id_mappings_path, "r", encoding="utf-8") as f:
                self.id_mappings = json.load(f)
    
    def process_corpus(self):
        corpus_path = os.path.join(self.dataset_dir, "corpus.jsonl")

        if self.overwrite or not os.path.exists(corpus_path):
            
            dataset = ir_datasets.load("natural-questions")
            nq_test = {}
            for doc in tqdm(dataset.docs_iter()):
                nq_id, passage_id = doc.doc_id.split("-")
                if nq_id not in nq_test:
                    nq_test[nq_id] = {
                        "title": doc.document_title,
                        "text": doc.text
                        }
                nq_test[nq_id]["text"] = nq_test[nq_id]["text"] + " " + doc.text

            with open(corpus_path, "w") as f:
                for docid, doc in tqdm(nq_test.items()):
                    f.write(json.dumps({"_id": docid, "text": doc["text"], "title": doc["title"]}) + '\n')

        else:
            log_message(f"Dataset already processed. Skipping.")
    

if __name__ == "__main__":
    args = parse_arguments()
    processor = NQProcessor(args.datapath, args.overwrite)
    
    processor.download()
    processor.process_corpus()
    processor.process_queries()
    processor.process_qrels()

    
    short_processor = NQShortProcessor(args.datapath, args.overwrite)
    
    short_processor.download()
    short_processor.process_corpus()
    short_processor.process_queries()
    short_processor.process_qrels()

    
    passage_processor = NQDocumentPassageProcessor(args.datapath, args.overwrite)
    
    passage_processor.process_corpus()
    passage_processor.process_queries()
    passage_processor.process_qrels()

    
    test_processor = NQTestProcessor(args.datapath, args.overwrite)
    
    test_processor.process_corpus()
    test_processor.process_queries()
    test_processor.process_qrels()
