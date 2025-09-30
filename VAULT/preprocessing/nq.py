from utils import *
from wikipedia import *

class NQProcessor(DatasetProcessor):
    def __init__(self, datapath, overwrite):
        super().__init__(datapath, "nq", overwrite)
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

    def process_corpus(self):
        # Check if Wikipedia files are there
        if len(os.listdir(self.download_dir))>1:
            self.wikipedia_processor.process_corpus()
        else:
            raise Exception(f"No extracted Wikipedia folders found at {self.download_dir}. Please run 'process_wikipedia.sh' first.")

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

    def _create_id_mappings(self):
        id_mappings_path = os.path.join(self.dataset_dir, "id_mappings.json")

        if os.path.exists(id_mappings_path) and not self.overwrite:
            with open(id_mappings_path, "r", encoding="utf-8") as f:
                return json.load(f)
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
            
            return id_mappings

    def process_qrels(self):
        id_mappings = self._create_id_mappings() # Dict
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
                        if nq_docid in id_mappings:
                            doc_id = id_mappings[nq_docid]
                            cursor.execute("SELECT id, title, text, url FROM articles WHERE id = ?", (doc_id,))
                            if cursor.fetchone() is not None:
                                f.write(f"{qrel.query_id}\t{doc_id}\t{qrel.relevance}\n")
            else:
                log_message(f"Qrels for {subset} already exist at {qrel_path}. Skipping qrel processing.", print_message=True)
        
        connection.close()

    def process_short_dataset(self):
        ids = self.wikipedia_processor._get_ids()
        id_mappings = json.load(open(os.path.join(self.dataset_dir, "id_mappings.json"), "r", encoding="utf-8"))
        dataset = ir_datasets.load(f"natural-questions")
        short_corpus_path = os.path.join(self.short_dataset_dir, "corpus.jsonl")

        if self.overwrite or not os.path.exists(short_corpus_path):
            with open(short_corpus_path, "w", encoding="utf-8") as f:
                for doc in tqdm(dataset.docs_iter()):
                    nq_docid = int(doc.doc_id.split("-")[0])
                    docid = id_mappings.get(str(nq_docid), None)
                    if docid in ids:
                        doc_obj = {
                            "_id": doc.doc_id,
                            "text": doc.text, 
                            "title": doc.document_title
                        }
                        f.write(json.dumps(doc_obj) + "\n")
                        
        
        short_queries_path = os.path.join(self.short_dataset_dir, "queries.jsonl")
        if self.overwrite or not os.path.exists(short_queries_path):
            with open(short_queries_path, "w", encoding="utf-8") as f:
                for subset in self.subsets:
                    dataset = ir_datasets.load(f"natural-questions/{subset}")

                    for query in dataset.queries_iter():
                        query_obj = {
                            "_id": query.query_id,
                            "text": query.text
                        }
                        f.write(json.dumps(query_obj) + "\n")

        for subset in self.subsets:
            dataset = ir_datasets.load(f"natural-questions/{subset}")
            subset = self.norm_subset_name[subset]
            
            short_qrel_dir = os.path.join(self.short_dataset_dir, "qrels")
            os.makedirs(short_qrel_dir, exist_ok=True)
            qrel_path = os.path.join(short_qrel_dir, f"{subset}.tsv")
            if self.overwrite or not os.path.exists(qrel_path):
                with open(qrel_path, "w", encoding="utf-8") as f:
                    for qrel in dataset.qrels_iter():
                        nq_docid = int(qrel.doc_id.split("-")[0])
                        docid = id_mappings.get(str(nq_docid), None)
                        if docid in ids:
                            f.write(f"{qrel.query_id}\t{qrel.doc_id}\t{qrel.relevance}\n")
            
if __name__ == "__main__":
    args = parse_arguments()
    processor = NQProcessor(args.datapath, args.overwrite)
    
    processor.download()
    processor.process_corpus()
    processor.process_queries()
    processor.process_qrels()
    processor.process_short_dataset()
