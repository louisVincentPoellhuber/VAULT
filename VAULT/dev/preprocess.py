import os
import sqlite3
import json
import re
import ir_datasets
import nltk
from tqdm import tqdm
nltk.download('punkt')

BLACKLIST_HEADERS = {
    "References", "Bibliography", "Notes", "External links", "See also",
    "Citations", "Further reading", "Sources"
}

def is_header(line):
    """Heuristic: detect if a line is a section header."""
    if not line:
        return False
    
    if not line[0].isupper():
        return False
    if not line.endswith("."):
        return False
    
    words = line.split()
    if len(words) > 8:
        return False
    
    header_name = line.rstrip(".")
    if header_name in BLACKLIST_HEADERS:
        return False
    
    if len(words) <= 6 and not re.search(r"[!?]", line):
        return True
    
    return False


def chunk_text(text, max_words):
    """Split text into chunks of at most max_words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks


def split_sections(text, max_words=512):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    sections = []
    current_header = "Introduction"
    current_text = []

    for line in lines:
        if is_header(line):
            # Save the previous section if it has content
            if current_text:
                full_text = current_header + ". " + " ".join(current_text).strip()
                sections.extend(chunk_text(full_text, max_words))
                current_text = []
            current_header = line.rstrip(".")
        else:
            current_text.append(line)

    # Save the last section
    if current_text:
        full_text = current_header + ". " + " ".join(current_text).strip()
        sections.extend(chunk_text(full_text, max_words))

    return sections

def split_text(text):
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        sentences = ["."]

    return sentences

def create_db(download_dir, db_dir, overwrite, split_fct = split_text):
    db_path = os.path.join(db_dir, "corpus.db")
    # dataset = ir_datasets.load(f"beir/hotpotqa/train")
    # train_qrels = list(dataset.qrels_iter())
    # relevant_doc_ids = set(qrel.doc_id for qrel in train_qrels)

    if overwrite or not os.path.exists(db_path):
        # 1. Create database connection & table
        print("Creating database.")
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id TEXT PRIMARY KEY,
            title TEXT,
            text TEXT, 
            url TEXT
        )
        """)
        conn.commit()

        # 2. Walk through every file in every folder
        batch = [] 
        total_processed_pages = 0
        for root, _, files in tqdm(os.walk(download_dir), total=len(os.listdir(download_dir))):
            for filename in files:
                if filename.startswith("wiki_"):
                    filepath = os.path.join(root, filename)

                    # 3. Read file line-by-line (streaming, not loading whole file in memory)
                    with open(filepath, "r", encoding="utf-8") as f:
                        for line in f:
                            article = json.loads(line)
                            # if article["id"] in relevant_doc_ids:
                            text = split_fct(article["text"]) # So index everything
                            for i, part in enumerate(text):
                                id = f"{article['id']}-{i}"
                                batch.append((id, article["title"], part, article["url"]))
                    
                    
                    if len(batch) >= 10000:
                        total_processed_pages+=len(batch)
                        # print(f"{total_processed_pages} sentences processed.")
                        cur.executemany("INSERT OR IGNORE INTO articles VALUES (?, ?, ?, ?)", batch)
                        conn.commit()
                        batch.clear()

        total_processed_pages+=len(batch)
        # print(f"{total_processed_pages} pages processed.")
        cur.executemany("INSERT OR IGNORE INTO articles VALUES (?, ?, ?, ?)", batch)
        conn.commit()
        batch.clear()
        
        # 4. Close connection
        conn.close()
    else:
        print("Database already exists. Skipping creation. ")

def get_all_ids(cursor):
    cursor.execute("SELECT id FROM articles")
    ids = [row[0] for row in cursor.fetchall()]
    
    id_dict={}
    for id in ids:
        base_id, section_id = id.split("-")
        if base_id not in id_dict:
            id_dict[base_id] = []
        id_dict[base_id].append(id)

    return id_dict

def process_queries(dataset_dir, overwrite):
    dataset = ir_datasets.load(f"beir/hotpotqa")
    queries_path = os.path.join(dataset_dir, "queries.jsonl")

    if overwrite or not os.path.exists(queries_path):
        with open(queries_path, "w", encoding="utf-8") as f:
            for query in dataset.queries_iter():
                obj = {
                    "_id": query.query_id,
                    "text": query.text
                }
                f.write(json.dumps(obj) + "\n")

def process_qrels(wikipedia_dir, qrel_dir, subsets, overwrite):
   
    db_path = os.path.join(wikipedia_dir, "corpus.db")
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    
    id_dict = get_all_ids(cursor)

    for subset in subsets:
        dataset = ir_datasets.load(f"beir/hotpotqa/{subset}")
        qrel_path = os.path.join(qrel_dir, f"{subset}.tsv")

        if overwrite or not os.path.exists(qrel_path):
            with open(qrel_path, "w", encoding="utf-8") as f:
                for qrel in dataset.qrels_iter():
                    if qrel.doc_id in id_dict:
                        for full_doc_id in id_dict[qrel.doc_id]:
                            f.write(f"{qrel.query_id}\t{full_doc_id}\t{qrel.relevance}\n")
    connection.close()

if __name__ == "__main__":
    db_dir = "/Tmp/lvpoellhuber/datasets/vault/hotpotqa_sentence"
    download_dir = "/Tmp/lvpoellhuber/datasets/vault/wikipedia/downloads"
    qrel_dir = db_dir+"/qrels"

    os.makedirs(db_dir, exist_ok=True)
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(qrel_dir, exist_ok=True)


    create_db(download_dir, db_dir, overwrite=False)
    process_queries(db_dir, overwrite=False)
    process_qrels(db_dir, qrel_dir, subsets=["train", "test"], overwrite=False)