from utils import *
from concurrent.futures import ThreadPoolExecutor, as_completed

class SciDocsProcessor(DatasetProcessor):
    def __init__(self, datapath, overwrite):
        super().__init__(datapath, "scidocs", overwrite)
     
    def download(self):
        ir_datasets.load("beir/scidocs")
        db_path = os.path.join(self.dataset_dir, "corpus.db")
        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()
        
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id TEXT PRIMARY KEY,
            corpusid INTEGER,
            text TEXT
        )
        """)
        self.connection.commit()
        
    def _stream_and_index(self, ssids):
        API_KEY = os.getenv("S2_API_KEY")
        if not API_KEY:
            raise ValueError("Please set the S2_API_KEY environment variable with your Semantic Scholar API key.")
        
        release_id = "2025-08-19"
        response = requests.get(
            f"https://api.semanticscholar.org/datasets/v1/release/{release_id}/dataset/s2orc/",
            headers={"x-api-key": API_KEY}
        ).json()

        def download_file(url):
            match = re.match(r"https://ai2-s2ag.s3.amazonaws.com/staging/(.*)/s2orc/(.*).gz(.*)", url)
            if match == None:
                return None
            assert match.group(1) == release_id
            if not match.group(2).startswith("shard"):
                log_message(f"Unexpected file {url}, skipping.", print_message=True)
            shard_id = match.group(2).split("_")[-1].split("?")[0]
            download_path = os.path.join(self.download_dir, f"{shard_id}.gz")

            if os.path.exists(download_path):
                return None  # already downloaded

            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(download_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            return download_path

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(download_file, url) for url in response["files"]]
            for future in tqdm(as_completed(futures), total=len(futures)):
                download_path = future.result()
                if download_path!=None: # Don't re-processed stuff that's already been processed
                    self._get_shard_articles(download_path, ssids)
                    os.remove(download_path)  # Clean up the downloaded shard file

        print("Downloaded all shards.")

    def _get_shard_articles(self, download_path, ssids):
        # 2. Walk through every file in every folder
        batch = [] 
        with gzip.open(download_path, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                article = json.loads(line)

                ssid = article["content"]["source"].get("pdfsha", None)

                if ssid in ssids:
                    text = article["content"]["text"]
                    corpusid = article["corpusid"]
                    
                    batch.append((ssid, text, corpusid))
        
            if len(batch)>0:
                self.cursor.executemany("INSERT OR IGNORE INTO articles VALUES (?, ?, ?)", batch)
                self.connection.commit()

    def process_corpus(self):
        dataset = ir_datasets.load(f"beir/scidocs")
        
        ssids = []
        for article in dataset.docs_iter():
            ssids.append(article.doc_id)

        self._stream_and_index(ssids)

    def process_queries(self):
        dataset = ir_datasets.load(f"beir/scidocs")
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
        
        db_path = os.path.join(self.download_dir, "corpus.db")
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        
        dataset = ir_datasets.load(f"beir/scidocs")
        qrel_path = os.path.join(self.qrel_dir, f"test.tsv")

        if self.overwrite or not os.path.exists(qrel_path):
            with open(qrel_path, "w", encoding="utf-8") as f:
                for qrel in dataset.qrels_iter():
                    cursor.execute("SELECT id, title, text, url FROM articles WHERE id = ?", (qrel.doc_id,))
                    if cursor.fetchone() is not None:
                        f.write(f"{qrel.query_id}\t{qrel.doc_id}\t{qrel.relevance}\n")
        else:
            log_message(f"Qrels for test already exist at {qrel_path}. Skipping qrel processing.", print_message=True)

        connection.close()

if __name__ == "__main__":
    args = parse_arguments()
    processor = SciDocsProcessor(args.datapath, args.overwrite)
    
    processor.download()
    processor.process_corpus()
    processor.process_queries()
    processor.process_qrels()
