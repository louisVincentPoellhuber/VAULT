import wikipediaapi
from utils import *
import bz2
from lxml import etree
import shutil
BASE_URL = "https://en.wikipedia.org/"

WIKI = wikipediaapi.Wikipedia(USER_AGENT, language='en')  # 'en' for English

def make_wikipedia_folders(vault_dir):
    wikipedia_dir = os.path.join(vault_dir, "wikipedia")
    os.makedirs(wikipedia_dir, exist_ok=True)

    wikipedia_download_dir = os.path.join(wikipedia_dir, "downloads")
    os.makedirs(wikipedia_download_dir, exist_ok=True)

    return wikipedia_dir, wikipedia_download_dir

def download_wikipedia(output_dir, overwrite):
    url = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"
    zip_out_path = os.path.join(output_dir,"enwiki-latest-pages-articles.xml.bz2")

    if overwrite or not os.path.exists(zip_out_path):
        print("Downloading compressed file.")
        headers = {
            "User-Agent": USER_AGENT
        }
        
        response = requests.get(url, headers=headers, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1MB
        tqdm_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

        with open(zip_out_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    tqdm_bar.update(len(chunk))
        tqdm_bar.close()


    xml_out_path = os.path.join(output_dir, "enwiki-latest-pages-articles.xml")
    if overwrite or not os.path.exists(xml_out_path):
        print("Extracting compressed file into XML.")
        with bz2.open(zip_out_path, "rb") as source, \
            open(xml_out_path, "wb") as dest:
                shutil.copyfileobj(source, dest)

    return xml_out_path

def create_db(download_dir, db_dir, overwrite):
    db_path = os.path.join(db_dir, "corpus.db")

    if overwrite or not os.path.exists(db_path):
        # 1. Create database connection & table
        print("Creating database.")
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY,
            title TEXT,
            text TEXT, 
            url TEXT
        )
        """)
        conn.commit()

        # 2. Walk through every file in every folder
        batch = [] 
        total_processed_pages = 0
        for root, _, files in os.walk(download_dir):
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
                        cur.executemany("INSERT OR IGNORE INTO articles VALUES (?, ?, ?, ?)", batch)
                        conn.commit()
                        batch.clear()

        total_processed_pages+=len(batch)
        print(f"{total_processed_pages} pages processed.")
        cur.executemany("INSERT OR IGNORE INTO articles VALUES (?, ?, ?, ?)", batch)
        conn.commit()
        batch.clear()
        
        # 4. Close connection
        conn.close()
    else:
        print("Database already exists. Skipping creation. ")