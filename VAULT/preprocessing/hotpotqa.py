from utils import *
from wikipedia import *

class HotPotQAProcessor(DatasetProcessor):
    def __init__(self, datapath, overwrite):
        super().__init__(datapath, "hotpotqa", overwrite)
        self.wikipedia_processor = WikipediaProcessor(datapath, overwrite)
        self.download_dir = self.wikipedia_processor.download_dir
        self.wikipedia_dir = self.wikipedia_processor.dataset_dir
        self.subsets = ["train", "dev", "test"]
     
    def download(self):
        ir_datasets.load("beir/hotpotqa")

    def process_corpus(self):
        if len(os.listdir(self.download_dir))>1:
            self.wikipedia_processor.process_corpus(self.download_dir, self.wikipedia_dir, self.overwrite)
        else:
            raise Exception(f"No extracted Wikipedia folders found at {self.download_dir}. Please run 'process_wikipedia.sh' first.")

    def process_queries(self):
        dataset = ir_datasets.load(f"beir/hotpotqa")
        queries_path = os.path.join(self.dataset_dir, "queries.jsonl")

        if self.overwrite or not os.path.exists(queries_path):
            log_message(f"Processing queries into {queries_path}.", print_message=True)
            with open(queries_path, "w", encoding="utf-8") as f:
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
            dataset = ir_datasets.load(f"beir/hotpotqa/{subset}")
            qrel_path = os.path.join(self.qrel_dir, f"{subset}.tsv")

            if self.overwrite or not os.path.exists(qrel_path):
                with open(qrel_path, "w", encoding="utf-8") as f:
                    for qrel in dataset.qrels_iter():
                        cursor.execute("SELECT id, title, text, url FROM articles WHERE id = ?", (qrel.doc_id,))
                        if cursor.fetchone() is not None:
                            f.write(f"{qrel.query_id}\t{qrel.doc_id}\t{qrel.relevance}\n")
            else:
                log_message(f"Qrels for {subset} already exist at {qrel_path}. Skipping qrel processing.", print_message=True)

        connection.close()

    def process_short_dataset(self):
        ids = self.wikipedia_processor._get_ids()
        dataset = ir_datasets.load(f"beir/hotpotqa")
        short_corpus_path = os.path.join(self.short_dataset_dir, "corpus.jsonl")

        if self.overwrite or not os.path.exists(short_corpus_path):
            with open(short_corpus_path, "w", encoding="utf-8") as f:
                for doc in tqdm(dataset.docs_iter()):
                    docid = int(doc.doc_id)
                    if docid in ids:
                        doc_obj = {
                            "_id": docid,
                            "text": doc.text, 
                            "title": doc.title
                        }
                        f.write(json.dumps(doc_obj) + "\n")
            
        short_queries_path = os.path.join(self.short_dataset_dir, "queries.jsonl")
        queries_path = os.path.join(self.dataset_dir, "queries.jsonl")
        if not os.path.exists(queries_path):
            raise Exception(f"Queries file not found at {queries_path}. Please run 'process_queries()' first.")
        shutil.copy(queries_path, short_queries_path)

        short_qrel_dir = os.path.join(self.short_dataset_dir, "qrels")
        os.makedirs(short_qrel_dir, exist_ok=True)
        for subset in self.subsets:           
            short_qrel_path = os.path.join(short_qrel_dir, f"{subset}.tsv")
            qrel_path = os.path.join(self.qrel_dir, f"{subset}.tsv")
            if not os.path.exists(qrel_path):
                raise Exception(f"Qrels file for {subset} not found at {qrel_path}. Please run 'process_qrels()' first.")
            shutil.copy(qrel_path, short_qrel_path)

if __name__ == "__main__":
    args = parse_arguments()
    processor = HotPotQAProcessor(args.datapath, args.overwrite)
    
    processor.download()
    processor.process_queries()
    processor.process_qrels()
    processor.process_short_dataset()