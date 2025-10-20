from utils import *
from wikipedia import *

class HotPotQAProcessor(DatasetProcessor):
    def __init__(self, datapath, overwrite, suffix=""):
        super().__init__(datapath, "hotpotqa"+suffix, overwrite)
        self.wikipedia_processor = WikipediaProcessor(datapath, overwrite)
        self.download_dir = self.wikipedia_processor.download_dir
        self.wikipedia_dir = self.wikipedia_processor.dataset_dir
        self.subsets = ["train", "dev", "test"]
     
    def download(self):
        ir_datasets.load("beir/hotpotqa")

    def process_corpus(self):
        if len(os.listdir(self.download_dir))>1:
            self.wikipedia_processor.process_corpus()
        else:
            raise Exception(f"No extracted Wikipedia folders found at {self.download_dir}. Please run 'process_wikipedia.sh' first.")

        corpus_id_file = os.path.join(self.dataset_dir, "corpus_ids.csv")
        if not os.path.exists(corpus_id_file) or self.overwrite:
            dataset = ir_datasets.load("beir/hotpotqa")
            hpqa_ids = set()
            with open(corpus_id_file, "w") as ids_out:
                for doc in tqdm(dataset.docs_iter()):
                    hpqa_id = doc.doc_id
                    if hpqa_id not in hpqa_ids:
                        ids_out.write(str(hpqa_id)+"\n")
                        hpqa_ids.add(hpqa_id)
        else:
            log_message(f"Corpus ID file already exists, skipping. ")

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

class HotPotQAShortProcessor(HotPotQAProcessor):
    def __init__(self, datapath, overwrite):
        super().__init__(datapath, overwrite, "_short")

    def process_corpus(self):
        corpus_path = os.path.join(self.dataset_dir, "corpus.jsonl")

        if self.overwrite or not os.path.exists(corpus_path):
            with open(corpus_path, "w", encoding="utf-8") as f:
                dataset = ir_datasets.load(f"beir/hotpotqa")
                for doc in tqdm(dataset.docs_iter()):
                    docid = int(doc.doc_id)
                    doc_obj = {
                        "_id": str(docid),
                        "text": doc.text, 
                        "title": doc.title
                    }
                    f.write(json.dumps(doc_obj) + "\n")
        else:
            log_message(f"Dataset already processed. Skipping.")
    
    def process_qrels(self):
        log_message(f"Processing qrels into {self.qrel_dir}.", print_message=True)

        for subset in self.subsets:
            dataset = ir_datasets.load(f"beir/hotpotqa/{subset}")
            qrel_path = os.path.join(self.qrel_dir, f"{subset}.tsv")

            if self.overwrite or not os.path.exists(qrel_path):
                with open(qrel_path, "w", encoding="utf-8") as f:
                    for qrel in dataset.qrels_iter():
                        f.write(f"{qrel.query_id}\t{qrel.doc_id}\t{qrel.relevance}\n")
            else:
                log_message(f"Qrels for {subset} already exist at {qrel_path}. Skipping qrel processing.", print_message=True)



if __name__ == "__main__":
    args = parse_arguments()
    processor = HotPotQAProcessor(args.datapath, args.overwrite)
    
    processor.download()
    processor.process_corpus()
    processor.process_queries()
    processor.process_qrels()

    short_processor = HotPotQAShortProcessor(args.datapath, args.overwrite)
    
    
    short_processor.download()
    short_processor.process_corpus()
    short_processor.process_queries()
    short_processor.process_qrels()
