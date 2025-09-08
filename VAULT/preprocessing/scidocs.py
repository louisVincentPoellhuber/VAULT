from utils import *

def download_s2orc(download_dir, overwrite=False):
    if len(os.listdir(download_dir)) ==0 or overwrite:
        API_KEY = os.getenv("S2_API_KEY")
        if not API_KEY:
            raise ValueError("Please set the S2_API_KEY environment variable with your Semantic Scholar API key.")

        # get latest release's ID
        response = requests.get("https://api.semanticscholar.org/datasets/v1/release/latest").json()
        RELEASE_ID = response["release_id"]
        print(f"Latest release ID: {RELEASE_ID}")

        # get the download links for the s2orc dataset; needs to pass API key through `x-api-key` header
        # download via wget. this can take a while...
        response = requests.get(f"https://api.semanticscholar.org/datasets/v1/release/{RELEASE_ID}/dataset/s2orc/", headers={"x-api-key": API_KEY}).json()
        for url in tqdm(response["files"]):
            match = re.match(r"https://ai2-s2ag.s3.amazonaws.com/staging/(.*)/s2orc/(.*).gz(.*)", url)
            assert match.group(1) == RELEASE_ID
            SHARD_ID = match.group(2)
            wget.download(url, out=os.path.join(download_dir, f"{SHARD_ID}.gz"))
        print("Downloaded all shards.")
    else:
        log_message("S2ORC shards already exist. Skipping download.", print_message=True)

def create_db(download_dir, db_dir, overwrite):
    db_path = os.path.join(db_dir, "corpus.db")
    shards_path = os.listdir(download_dir)

    if overwrite or not os.path.exists(db_path):
        # 1. Create database connection & table
        print("Creating database.")
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id TEXT PRIMARY KEY,
            corpusid INTEGER,
            text TEXT
        )
        """)
        conn.commit()

        # 2. Walk through every file in every folder
        batch = [] 
        total_processed_pages = 0
        for shard in tqdm(shards_path):
            shard = os.path.join('/Tmp/lvpoellhuber/datasets/vault/corpus/s2orc/downloads', shard)
            with gzip.open(shard, 'rt', encoding='utf-8') as f:
                for line in tqdm(f):
                    line = line.strip()
                    article = json.loads(line)

                    ssid = article["content"]["source"]["pdfsha"]
                    text = article["content"]["text"]
                    corpusid = article["corpusid"]
                    
                    batch.append((ssid, text, corpusid))
        
            total_processed_pages+=len(batch)
            cur.executemany("INSERT OR IGNORE INTO articles VALUES (?, ?, ?)", batch)
            conn.commit()
            batch.clear()

        # 4. Close connection
        conn.close()
    else:
        print("Database already exists. Skipping creation. ")

class SciDocsProcessor(DatasetProcessor):
    def __init__(self, datapath, overwrite):
        super().__init__(datapath, "scidocs", overwrite)
     
    def download(self):
        ir_datasets.load("beir/scidocs")
        download_s2orc(self.download_dir, self.overwrite)
        create_db(self.download_dir, self.dataset_dir, self.overwrite)

    def process_corpus(self):
        log_message("No need to process corpus for SciDocs. Using S2ORC database directly.", print_message=True)

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
        
        for subset in self.subsets:
            dataset = ir_datasets.load(f"beir/scidocs/{subset}")
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

if __name__ == "__main__":
    args = parse_arguments()
    processor = SciDocsProcessor(args.datapath, args.overwrite)
    
    processor.download()
    processor.process_corpus()
    processor.process_queries()
    processor.process_qrels()
