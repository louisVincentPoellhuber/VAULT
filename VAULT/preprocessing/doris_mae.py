from utils import *
import kagglehub

def get_annotations(dataset):
    annotations = pd.DataFrame(columns=["aspect_id", "abstract_id", "score"])
    for annotation in tqdm(dataset["Annotation"]):
        aspect_id = annotation["aspect_id"]
        abstract_id = annotation["abstract_id"]
        score = annotation["score"]
        
        if score>0:
            row = pd.DataFrame({
                "aspect_id": [aspect_id],
                "abstract_id": [abstract_id],
                "score": [score]})
            
            annotations = pd.concat([annotations, row], ignore_index=True)
    return annotations

def get_qrels(annotations, aspect_ids, qid):
    
    # Get the QRELs for all the listed aspects 
    aspects_qrels = annotations[annotations["aspect_id"].isin(aspect_ids)]
    # Sum over the abstract IDs
    grouped_qrels = aspects_qrels[["abstract_id", "score"]].groupby("abstract_id").sum()
    # Average score
    avg_score = grouped_qrels / len(aspect_ids)
    # Extract QRELs with an average score larger than 1
    qrel_list = avg_score[avg_score["score"]>=1].index.to_list()
    # If no QREL passing the thresshold exist, take the most relevant one
    if len(qrel_list)==0:
        qrel_list = avg_score[avg_score["score"]==avg_score["score"].max()].index.to_list()
        log_message(f"No sufficiently relevant document found for question {qid}: taking the next most relevant document(s).", level=logging.WARNING)

    return qrel_list

def create_test_train_split(qrels, queries, train_ratio=0.75):
    all_qrel_ids = list(qrels.keys())
    full_qrels_ids = []
    sub_qrels_ids = []
    # Sort qrels into full and sub-queries
    for qid in qrels.keys():
        if queries[qid]["type"]=="full_query":
            full_qrels_ids.append(qid)
        elif queries[qid]["type"]=="sub_query":
            sub_qrels_ids.append(qid)

    # Sample full and sub queries equally
    nb_train_full = round(len(full_qrels_ids)*train_ratio)
    nb_train_sub = round(len(sub_qrels_ids)*train_ratio)

    train_qrels = random.sample(full_qrels_ids, nb_train_full)
    train_qrels += random.sample(sub_qrels_ids, nb_train_sub)
    train_qrels = set(train_qrels)

    # Pick the remaining IDs for testing
    test_qrels = set(all_qrel_ids).symmetric_difference(set(train_qrels))

    return train_qrels, test_qrels

class DorisMAEProcessor(DatasetProcessor):
    def __init__(self, datapath, overwrite):
        super().__init__(datapath, "doris-mae", overwrite)
        self.subsets = ["train", "test"]
        self.download_batch_size = 2500

     
    def download(self):    
        file_url = "https://zenodo.org/records/8299749/files/DORIS-MAE_dataset_v1.json?download=1"
        local_path = os.path.join(self.download_dir, "DORIS-MAE_dataset_v1.json")

        # Download DORIS-MAE dataset
        if not os.path.exists(local_path) or self.overwrite:
            with requests.get(file_url, stream=True) as r:
                r.raise_for_status()
                with open(local_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024*1024):
                        f.write(chunk) 
        
        with open(local_path, 'r') as file:
            dataset = json.load(file)

        self.dataset = dataset
        doris_corpus = dataset["Corpus"]

        # Extract Arxiv IDs from DORIS-MAE corpus
        doris_ids = set()
        doris_dict = {}
        pattern = "[+-]?([0-9]*[.])?[0-9]+"
        for sample in doris_corpus:
            id = sample["url"].split("/")[-1]
            if len(id)>2:
                if id[-2]=="v":
                    id = id[:-2]
            if re.match(pattern, id): # Filter out IDs that are not Arxiv IDs
                doris_ids.add(str(id))
                doris_dict[id] = sample["abstract_id"]

        # Download Arxiv metadatas
        path = kagglehub.dataset_download("Cornell-University/arxiv")
        dataset_path = os.path.join(path, "arxiv-metadata-oai-snapshot.json")

        # Create database for IDs and Text
        db_path = os.path.join(self.dataset_dir, "corpus.db")
        connection= sqlite3.connect(db_path)
        cursor = connection.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id TEXT PRIMARY KEY,
            title TEXT, 
            abstract_id INTEGER
        )
        """)
        connection.commit()

        # Extract download URLs from Arxiv papers existing in DORIS-MAE
        if len(os.listdir(self.download_dir))<=2 or self.overwrite:
            db_batch = []
            url_batch = []
            batch_nb = 0
            data = pd.read_json(dataset_path, lines=True)
            for row in tqdm(data.iterrows()):
                example = row[1]
                if example["id"] in doris_ids:
                    id = str(example["id"])
                    is_old_id = "/" in id
                    if is_old_id:
                        category = example["id"].split("/")[0]
                        id = id.split("/")[1]
                        month_id = id[:4]
                    else:
                        category = "arxiv"
                        month_id = id.split(".")[0]

                        # Zero pad
                        if len(month_id) == 3:
                            month_id = "0" + month_id

                    most_recent_version = example["versions"][-1]["version"]
                    download_url = f"gs://arxiv-dataset/arxiv/{category}/pdf/{month_id}/{id}{most_recent_version}.pdf"
                    db_batch.append((id+most_recent_version, example["title"], doris_dict[example["id"]]))
                    url_batch.append(download_url + "\n")

                if len(url_batch) >= self.download_batch_size:
                    url_path = os.path.join(self.download_dir, f"urls_{batch_nb}.txt")
                    with open(url_path, "w", encoding="utf-8") as f:
                        f.writelines(url_batch)
                    batch_nb+=1
                    url_batch.clear()

                if len(db_batch)>= 10000:
                    cursor.executemany("INSERT OR IGNORE INTO articles VALUES (?, ?, ?)", db_batch)
                    connection.commit()
                    db_batch.clear()
        

    def process_corpus(self):
        db_path = os.path.join(self.dataset_dir, "corpus.db")
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database {db_path} does not exist. Please run the download step first.")
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        extract_texts = True
        try:
            cursor.execute("ALTER TABLE articles ADD COLUMN text TEXT;")
        except sqlite3.OperationalError:
            log_message("Texts already processed. Skipping text extraction.", print_message=True)
            extract_texts = False


        if extract_texts or self.overwrite:
            log_message(f"Processing corpus.", print_message=True)
            batch = []
            for file in tqdm(os.listdir(self.download_dir), desc="URL files to process"):
                # Find url_*.txt files
                if file.startswith("url"):
                    # Re-make the PDF directory that's deleted every iteration
                    pdf_dir = os.path.join(self.download_dir, "pdf")
                    os.makedirs(pdf_dir, exist_ok=True)

                    # Use gsutil to download sample of 1000 PDFs
                    url_path = os.path.join(self.download_dir, file)
            
                    cmd = [
                        "gsutil", "-m", "cp", "-I", pdf_dir
                    ]

                    # Download to pdf_dir
                    with open(url_path, "r") as infile:
                        subprocess.run(cmd, stdin=infile, check=False)

                    # Wait for the PDFs to finish downloading
                    time.sleep(1)

                    # Read the 100 PDFs and extract their text
                    for pdf in tqdm(os.listdir(pdf_dir), desc="PDFs to extract"):
                        if pdf.endswith(".pdf"):
                            text = pdf_to_text(os.path.join(pdf_dir, pdf))
                            id = pdf.replace(".pdf", "")
                            batch.append((text, id))
                    
                    # Clear the PDF directory to save storage space
                    shutil.rmtree(pdf_dir)
                        
                    # Add text to DB every now and then
                    cursor.executemany("UPDATE articles SET text=? WHERE id=?", batch)
                    connection.commit()
                    batch.clear()

            connection.close()
        else:
            log_message(f"Texts already extracted. Skipping extraction.")

    def process_queries(self):
        queries_path = os.path.join(self.dataset_dir, "queries.jsonl")

        if self.overwrite or not os.path.exists(queries_path):
            log_message(f"Processing queries into {queries_path}.", print_message=True)
            with open(queries_path, "w", encoding="utf-8") as f:
                for qid, query in enumerate(self.dataset["Query"]):
                    obj = {
                        "_id": str(qid),
                        "text": query["query_text"], 
                        "type": "full_query"
                    }
                    f.write(json.dumps(obj) + "\n")

                    for sub_qid, subquery in enumerate(query["sent2aspect_id"]):
                        obj = {
                            "_id": str(qid)+"-"+str(sub_qid),
                            "text": subquery, 
                            "type": "sub_query"
                        }
                        f.write(json.dumps(obj) + "\n")
        else:
            log_message(f"Queries already exist at {queries_path}. Skipping query processing.", print_message=True)

    def _get_arxiv2abstract(self):
        arxiv_db_path = os.path.join(self.dataset_dir, "corpus.db")
        connection = sqlite3.connect(arxiv_db_path)
        cursor = connection.cursor()

        abs2arxiv = {}
        cursor.execute("SELECT id, title FROM articles")
        for arxiv_id, abs_id in tqdm(cursor.execute("SELECT id, abstract_id FROM articles"), desc="Collecting all Arxiv ID - Abstract ID pairs from Arxiv database."):
            abs2arxiv[abs_id] = arxiv_id

        connection.close()

        return abs2arxiv

    def process_qrels(self):
        queries_path = os.path.join(self.dataset_dir, "queries.jsonl")
        queries = load_jsonl(queries_path)

        annotations = get_annotations(self.dataset)
        abs2arxiv = self._get_arxiv2abstract()

        qrels = {}
        for qid, query in enumerate(self.dataset["Query"]):    
            # List out all aspects and sub-aspects 
            aspect_ids = []
            sub_qid = 0
            for text, aspects in query['sent2aspect_id'].items():
                subaspects = []
                for subaspect in aspects:
                    subaspects += query['aspects'][subaspect] + [subaspect]

                aspect_ids += subaspects
                if len(subaspects)>0:
                    sqid = str(qid)+"-"+str(sub_qid)
                    qrels[sqid] = get_qrels(annotations, subaspects, sqid)
                    sub_qid+=1
                
            aspect_ids = list(set(aspect_ids))
            qrels[str(qid)] = get_qrels(annotations, aspect_ids, qid)

        train_list, test_list = create_test_train_split(qrels, queries, 0.75)

        for i, qrel_list in enumerate([train_list, test_list]):
            qrel_path = os.path.join(self.qrel_dir, self.subsets[i]+".tsv")

            if self.overwrite or not os.path.exists(qrel_path):
                with open(qrel_path, "w", encoding="utf-8") as f:
                    for qid in qrel_list:
                        qrel = qrels[qid]
                        for abs_id in qrel:
                            if abs_id in abs2arxiv.keys():
                                arxiv_id = abs2arxiv[abs_id]
                                f.write(f"{qid}\t{arxiv_id}\t{1}\n")
            else:
                log_message(f" Qrels for {self.subsets[i]} already exist at {qrel_path}. Skipping qrel processing.", print_message=True)
        

if __name__ == "__main__":
    args = parse_arguments()
    processor = DorisMAEProcessor(args.datapath, args.overwrite)
    
    processor.download()
    processor.process_corpus()
    processor.process_queries()
    processor.process_qrels()
