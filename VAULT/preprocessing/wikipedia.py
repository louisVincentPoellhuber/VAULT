import wikipediaapi
from utils import *
import bz2
from lxml import etree
import shutil
BASE_URL = "https://en.wikipedia.org/"

WIKI = wikipediaapi.Wikipedia(USER_AGENT, language='en')  # 'en' for English
import re
import json

    
class WikipediaProcessor(DatasetProcessor):
    def __init__(self, datapath, overwrite):
        super().__init__(datapath, "wikipedia", overwrite)
        self.datapath = datapath
        self.db_path = os.path.join(self.dataset_dir, "corpus.db")
        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()
        self.subsets = ["train", "test", "dev"]
        self.norm_subset_name = {
                "train": "train",
                "dev": "train", 
                "test": "test"
            }
        log_message("The HotPotQA dataset uses a 'dev' split which is unused for VAULT. For the purposes of this benchmark, it will be added to the 'train' split.", print_message=True)
        

    def download(self):
        zip_out_path = os.path.join(self.download_dir,"enwiki-latest-pages-articles.xml.bz2")

        if not os.path.exists(zip_out_path):
           raise Exception("XML Dump not found. Please run extract_wikipedia.sh.")
    
    def _get_ids(self):
        ids = set()
        for id in tqdm(self.cursor.execute("SELECT id FROM articles"), desc="Collecting all Wikipedia IDs from database."):
            ids.add(id[0])

        return ids
    
    def process_corpus(self):
        self.download()

        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY,
            title TEXT,
            text TEXT, 
            url TEXT
        )
        """)
        self.connection.commit()

        # Simple hack to check if there are some IDs in the database
        process = True
        nb_ids = 0
        for id in tqdm(self.cursor.execute("SELECT id FROM articles")):
            nb_ids +=1
            if nb_ids>10:
                process=False
                break

        if self.overwrite or process:

            # 2. Walk through every file in every folder
            batch = [] 
            total_processed_pages = 0
            for root, _, files in os.walk(self.download_dir):
                for filename in files:
                    if filename.startswith("wiki_"):
                        filepath = os.path.join(root, filename)

                        # 3. Read file line-by-line (streaming, not loading whole file in memory)
                        with open(filepath, "r", encoding="utf-8") as f:
                            for line in f:
                                article = json.loads(line)

                                batch.append((int(article["id"]), article["title"], article["text"], article["url"]))
                        
                        
                        if len(batch) >= 10000:
                            total_processed_pages+=len(batch)
                            print(f"{total_processed_pages} pages processed.")
                            self.cursor.executemany("INSERT OR IGNORE INTO articles VALUES (?, ?, ?, ?)", batch)
                            self.connection.commit()
                            batch.clear()

            total_processed_pages+=len(batch)
            print(f"{total_processed_pages} pages processed.")
            self.cursor.executemany("INSERT OR IGNORE INTO articles VALUES (?, ?, ?, ?)", batch)
            self.connection.commit()
            batch.clear()
        else:
            print("Database already exists. Skipping creation. ")

    def process_queries(self):
        hpqa_path = os.path.join(self.datapath, "hotpotqa")
        hpqa_queries = os.path.join(hpqa_path, "queries.jsonl")
        nq_path = os.path.join(self.datapath, "nq")
        nq_queries = os.path.join(nq_path, "queries.jsonl")

        if not os.path.exists(hpqa_queries):
            raise Exception("Dataset not found. Please process HotPotQA first, by running hotpotqa.py.")
        if not os.path.exists(nq_queries):
            raise Exception("Dataset not found. Please process NQ first, by running nq.py.")

        wikipedia_queries = os.path.join(self.dataset_dir, "queries.jsonl")
        with open(hpqa_queries, "r") as hpqa_in, open(nq_queries, "r") as nq_in, open(wikipedia_queries, "w") as wiki_out:
            for line in hpqa_in:
                wiki_out.write(line)
            for line in nq_in:
                wiki_out.write(line)

    def process_qrels(self):
        hpqa_qrel_dir = os.path.join(self.datapath, "hotpotqa", "qrels")
        nq_qrel_dir = os.path.join(self.datapath, "nq", "qrels")

        if not os.path.exists(hpqa_qrel_dir):
            raise Exception("Dataset not found. Please process HotPotQA first, by running hotpotqa.py.")
        if not os.path.exists(nq_qrel_dir):
            raise Exception("Dataset not found. Please process NQ first, by running nq.py.")

        qrel_dirs = [hpqa_qrel_dir, nq_qrel_dir]
        for qrel_dir in qrel_dirs:
            for subset in os.listdir(qrel_dir):
                subset_path = os.path.join(qrel_dir, subset)
                subset_name = self.norm_subset_name[subset.split(".")[0]]
                wikipedia_qrels_subset = os.path.join(self.qrel_dir, subset_name+".tsv")

                with open(subset_path, "r") as dataset_in, open(wikipedia_qrels_subset, "a") as wiki_out:
                    for line in dataset_in:
                        wiki_out.write(line)

if __name__ == "__main__":
    args = parse_arguments()
    processor = WikipediaProcessor(args.datapath, args.overwrite)
    
    processor.process_corpus()
    processor.process_queries()
    processor.process_qrels()
