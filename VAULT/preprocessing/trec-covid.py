from utils import *

class TrecCovidProcessor(DatasetProcessor):
    def __init__(self, datapath, overwrite):
        super().__init__(datapath, "trec-covid",  overwrite)
        self.id_dict = {}
        self.download_batch_size = 2500
        self.subsets = ["train", "test"]
     
    def download(self):
        trec_url = "https://ir.nist.gov/covidSubmit/data/"
        files = ["topics-rnd5.xml", "qrels-covid_d5_j0.5-5.txt"]
        
        for file in files:
            file_url = trec_url+file
            local_path = os.path.join(self.download_dir, file)

            with requests.get(file_url, stream=True) as r:
                r.raise_for_status()
                with open(local_path, 'w') as f:
                    f.write(r.text) 

        cord_url = "https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/cord-19_2020-07-16.tar.gz"

        local_path = os.path.join(self.download_dir, "cord-19_2020-07-16.tar.gz")

        if not os.path.exists(local_path) or self.overwrite:
            with requests.get(cord_url, stream=True) as r:
                r.raise_for_status()
                with open(local_path, 'wb') as f:
                    for chunk in tqdm(r.iter_content(chunk_size=1024*1024)):
                        f.write(chunk) 
        else:
            log_message(f"Data already downloaded. ")


    def process_corpus(self):
        corpus_path = os.path.join(self.dataset_dir, "corpus.jsonl")

        if not os.path.exists(corpus_path) or self.overwrite:
            # Iterate over the downloaded files, looking for the cord-19 tars
            with open(corpus_path, "w") as corpus_file:
                for filename in os.listdir(self.download_dir):
                    if filename.startswith("cord-19"):
                        tar_path = os.path.join(self.download_dir, filename)
                        date = filename.split("_")[-1].split(".")[0]

                        # Open the cord-19 tar archive
                        with tarfile.open(tar_path, "r") as outer_tar:
                            # Get metadata information, and append it to the ID Table
                            metadata_member = outer_tar.getmember(f"{date}/metadata.csv")
                            f = outer_tar.extractfile(metadata_member)
                            df = pd.read_csv(f)[["sha", "cord_uid"]].dropna()
                            id_dict = dict(zip(df["sha"], df["cord_uid"]))
                            self.id_dict.update(id_dict)

                            for member in tqdm(outer_tar.getmembers()):                                                
                                if member.name.endswith(".tar.gz"):
                                    # Extract the inner JSON archives
                                    f = outer_tar.extractfile(member)
                                    inner_bytes = f.read()
                                    inner_fileobj = io.BytesIO(inner_bytes)

                                    # Open the inner tar.gz
                                    with tarfile.open(fileobj=inner_fileobj, mode="r") as inner_tar:
                                        for inner_member in inner_tar.getmembers():
                                            if inner_member.name.endswith(".json"):
                                                with inner_tar.extractfile(inner_member) as inner_f:
                                                    article = json.load(inner_f)
                                                    cord_id = self.id_dict.get(article["paper_id"], None)
                                                    if cord_id is not None:
                                                        title = article["metadata"]["title"]
                                                        abstract = " ".join([text["text"] for text in article["abstract"]])
                                                        body_text = " ".join([text["text"] for text in article["body_text"]])
                                                        full_text = abstract+" "+body_text

                                                        obj = {
                                                            "_id": cord_id,
                                                            "text": full_text, 
                                                            "title": title
                                                        }
                                                        corpus_file.write(json.dumps(obj) + "\n")

    def _get_cord_ids(self):
        corpus_path = os.path.join(self.dataset_dir, "corpus.jsonl")
        corpus = load_jsonl(corpus_path)

        ids = []
        for item in corpus:
            ids.append(item)

        return ids

                                        
    def process_queries(self):
        queries_path = os.path.join(self.dataset_dir, "queries.jsonl")
        queries_in_file = os.path.join(self.download_dir, "topics-rnd5.xml")

        if self.overwrite or not os.path.exists(queries_path):
            log_message(f"Processing queries into {queries_path}.", print_message=True)
            with open(queries_path, "a", encoding="utf-8") as f:
                tree = ET.parse(queries_in_file)
                root = tree.getroot()
                
                for topic in root.findall("topic"):
                    id = topic.get("number")
                    query = topic.find("question").text.strip()
                    obj = {
                        "_id": id,
                        "text": query
                    }
                    f.write(json.dumps(obj) + "\n")
        else:
            log_message(f"Queries already exist at {queries_path}. Skipping query processing.", print_message=True)

    def process_qrels(self):
        log_message(f"Processing qrels into {self.qrel_dir}.", print_message=True)
        qrel_in_file = os.path.join(self.download_dir, "qrels-covid_d5_j0.5-5.txt")
        qrel_path = os.path.join(self.qrel_dir, "test.tsv")
        cord_ids = self._get_cord_ids()

        all_qrels = []
        if self.overwrite or not os.path.exists(qrel_path): 
            with open(qrel_in_file, "r") as qrel_in:
                for line in qrel_in.readlines():
                    qid, _, cord_id, score = line.strip().split(" ")

                    if (cord_id in cord_ids) and (int(score)>0):
                        all_qrels.append((qid, cord_id, score))

            ratio = 0.75
            nb_train = round(ratio * len(all_qrels))
            train_qrels = random.sample(all_qrels, k=nb_train)
            test_qrels = list(set(all_qrels).symmetric_difference(set(train_qrels)))

            qrels = [train_qrels, test_qrels]

            for i, subset_qrel in enumerate(qrels):
                subset = self.subsets[i]
                qrel_path = os.path.join(self.qrel_dir, subset+".tsv")

                with open(qrel_path, "w", encoding="utf-8") as qrel_file:
                    for qrel in subset_qrel:
                        qrel_file.write(f"{qrel[0]}\t{qrel[1]}\t{qrel[2]}\n")
        else:
            log_message(f"Qrels for test already exist at {qrel_path}. Skipping qrel processing.", print_message=True)

    def process_short_dataset(self):
        dataset = ir_datasets.load("cord19/trec-covid")
        corpus_ids = self._get_cord_ids()

        short_corpus_path = os.path.join(self.short_dataset_dir, "corpus.jsonl")

        with open(short_corpus_path, "w", encoding="utf-8") as f:
            for doc in tqdm(dataset.docs_iter()):
                docid = doc.doc_id
                if docid in corpus_ids:
                    doc_obj = {
                        "_id": docid,
                        "text": doc.abstract, 
                        "title": doc.title
                    }
                    f.write(json.dumps(doc_obj) + "\n")

        short_queries_path = os.path.join(self.short_dataset_dir, "queries.jsonl")
        queries_path = os.path.join(self.dataset_dir, "queries.jsonl")
        if not os.path.exists(queries_path):
            raise Exception(f"Queries not found at {queries_path}. Please run full dataset processing first.")
        shutil.copy(queries_path, short_queries_path)

        short_qrel_dir = os.path.join(self.short_dataset_dir, "qrels")
        os.makedirs(short_qrel_dir, exist_ok=True)
        for subset in self.subsets:
            short_qrel_path = os.path.join(short_qrel_dir, subset+".tsv")
            qrel_path = os.path.join(self.qrel_dir, subset+".tsv")
            if not os.path.exists(qrel_path):
                raise Exception(f"Qrels not found at {qrel_path}. Please run full dataset processing first.")
        shutil.copy(qrel_path, short_qrel_path)

if __name__ == "__main__":
    args = parse_arguments()
    processor = TrecCovidProcessor(args.datapath, args.overwrite)
    
    processor.download()
    processor.process_corpus()
    processor.process_queries()
    processor.process_qrels()
    processor.process_short_dataset()