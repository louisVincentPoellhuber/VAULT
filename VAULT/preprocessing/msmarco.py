from utils import *

class MSMarcoProcessor(DatasetProcessor):
    def __init__(self, datapath, overwrite, suffix=""):
        super().__init__(datapath, "msmarco"+suffix, overwrite)
        self.subsets = ["train", "dev"]
        self.norm_subset_name = {
                "train": "train",
                "dev": "test"
            }
        log_message("The MS MARCO dataset uses 'train' and 'dev' split. These will be renamed to 'train' and 'test' respectively for consistency.", print_message=True)
     
     
    def download(self):
        files = ["msmarco-docs.tsv.gz", "msmarco-doctrain-queries.tsv.gz", "msmarco-doctrain-qrels.tsv.gz", "msmarco-docdev-queries.tsv.gz", "msmarco-docdev-qrels.tsv.gz"]
        url = "https://msmarco.z22.web.core.windows.net/msmarcoranking/"

        filenames = "\t\n| ".join(files)
        log_message(f"Downloading files: {filenames}")
        for download_file in files:
            file_url = url + download_file
            download_path = os.path.join(self.download_dir, download_file)
            
            if not os.path.exists(download_path) or self.overwrite:
                r = requests.get(file_url, stream=True)
                total = int(r.headers.get("Content-Length", 0))
                with (
                    open(download_path, "wb") as fd,
                    tqdm(
                        desc=download_path,
                        total=total,
                        unit="iB",
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as bar,
                ):
                    for data in r.iter_content(chunk_size=1024):
                        size = fd.write(data)
                        bar.update(size)
            else:
                log_message(f"File already exists at {download_path}. Skipping download.", print_message=True)

    def process_corpus(self):
        corpus_path = os.path.join(self.dataset_dir, "corpus.jsonl")

        if not os.path.exists(corpus_path) or self.overwrite:
            log_message("Writing corpus to disk")
            with open(corpus_path, "w") as corpus_out, \
                gzip.open(os.path.join(self.download_dir, "msmarco-docs.tsv.gz"), 'rt', encoding='utf8') as corpus_in:
                print("Reading corpus all at once. This will take about two minutes.")
                corpus_in_content = corpus_in.readlines()
                
                skipped_docs = ["D2070532", "D705499", "D3246147", "D3414215", "D3336081", "D3259427"] # Known problematic documentss
                for doc_line in tqdm(corpus_in_content):
                    doc = doc_line.rstrip().split("\t")
                    if len(doc) == 4:
                        json.dump({"_id": doc[0], "title": doc[2], "text": doc[3]}, corpus_out)
                        corpus_out.write("\n")
                    else:
                        skipped_docs.append(doc[0])

            skipped_doc_path = os.path.join(self.download_dir, "skipped_docs.csv")
            pd.Series(skipped_docs).to_csv(skipped_doc_path, index=False, header=None)
        else:
            log_message(f"Corpus already exists at {corpus_path}. Skipping corpus processing.", print_message=True)

    def process_queries(self):
        queries_path = os.path.join(self.dataset_dir, "queries.jsonl")
    
        if not os.path.exists(queries_path) or self.overwrite:
            querystring = {}
            with gzip.open(os.path.join(self.download_dir, "msmarco-doctrain-queries.tsv.gz"), 'rt', encoding='utf8') as f:
                tsvreader = csv.reader(f, delimiter="\t")
                for [topicid, querystring_of_topicid] in tsvreader:
                    querystring[topicid] = querystring_of_topicid
            with gzip.open(os.path.join(self.download_dir, "msmarco-docdev-queries.tsv.gz"), 'rt', encoding='utf8') as f:
                tsvreader = csv.reader(f, delimiter="\t")
                for [topicid, querystring_of_topicid] in tsvreader:
                    querystring[topicid] = querystring_of_topicid      

            log_message("Writing queries to disk")
            queries_filepah = os.path.join(self.download_dir, "queries.jsonl")

            with open(queries_filepah, "w") as f:
                for qid, query in tqdm(querystring.items()):
                    json.dump({"_id": qid, "text": query, "metadata": {}}, f)
                    f.write("\n")

        else:
            log_message(f"Queries already exist at {queries_path}. Skipping query processing.", print_message=True)

    def process_qrels(self):
        log_message(f"Processing qrels into {self.qrel_dir}.", print_message=True)

        skipped_doc_path = os.path.join(self.download_dir, "skipped_docs.csv")
        skipped_docs = pd.read_csv(skipped_doc_path, header=None)[0].to_list()
        
        for subset in self.subsets:
            qrel_filepath = os.path.join(self.qrel_dir, self.norm_subset_name[subset]+".tsv")

            if not os.path.exists(qrel_filepath) or self.overwrite:
                subset_qrel = {}
                with gzip.open(os.path.join(self.download_dir, f"msmarco-doc{subset}-qrels.tsv.gz"), 'rt', encoding='utf8') as f:
                    tsvreader = csv.reader(f, delimiter="\t")
                    for item in tsvreader:
                        topicid, _, docid, rel = item[0].split(" ")
                        assert rel == "1"
                        if docid not in skipped_docs: # Skip  documents without text
                            subset_qrel[topicid] = docid

                log_message("Writing qrels to disk")

                with open(qrel_filepath, "w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f, delimiter="\t", lineterminator='\n')
                    
                    writer.writerow(["query-id", "corpus-id", "score"])
                    
                    for qid, docid in subset_qrel.items():
                        writer.writerow([qid, docid, 1])
            else:
                log_message(f"Qrels for {subset} already exist at {qrel_filepath}. Skipping qrel processing.", print_message=True)

class MSMarcoShortProcessor(MSMarcoProcessor):
    def __init__(self, datapath, overwrite):
        super().__init__(datapath, overwrite, "_short")
        self.subsets = ["train", "dev"]
        self.norm_subset_name = {
                "train": "train",
                "dev": "test"
            }
        log_message("The MS MARCO Short dataset uses 'train' and 'dev' split. These will be renamed to 'train' and 'test' respectively for consistency.", print_message=True)
     
     
    def download(self):
        files = ["collection.tar.gz", "queries.tar.gz", "qrels.dev.tsv", "qrels.train.tsv"]
        url = "https://msmarco.z22.web.core.windows.net/msmarcoranking/"

        filenames = "\t\n| ".join(files)
        log_message(f"Downloading files: {filenames}")
        for download_file in files:
            file_url = url + download_file
            download_path = os.path.join(self.download_dir, download_file)
            
            if not os.path.exists(download_path) or self.overwrite:
                r = requests.get(file_url, stream=True)
                total = int(r.headers.get("Content-Length", 0))
                with (
                    open(download_path, "wb") as fd,
                    tqdm(
                        desc=download_path,
                        total=total,
                        unit="iB",
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as bar,
                ):
                    for data in r.iter_content(chunk_size=1024):
                        size = fd.write(data)
                        bar.update(size)
            else:
                log_message(f"File already exists at {download_path}. Skipping download.", print_message=True)

    def process_corpus(self):
        corpus_path = os.path.join(self.dataset_dir, "corpus.jsonl")

        if not os.path.exists(corpus_path) or self.overwrite:
            log_message("Writing corpus to disk")
            with open(corpus_path, "w") as corpus_out, \
            tarfile.open(os.path.join(self.download_dir, "collection.tar.gz"), 'r:gz') as tar:
                for member in tar.getmembers():
                    if member.isfile():
                        f = tar.extractfile(member)
                        if f is not None:
                            print("Reading corpus all at once.")
                            corpus_in_content = f.readlines()
                            
                            for i, doc_line in enumerate(tqdm(corpus_in_content)):
                                doc = doc_line.decode("utf-8").rstrip().split("\t")
                                json.dump({"_id": doc[0],"text": doc[1]}, corpus_out)
                                corpus_out.write("\n")
        else:
            log_message(f"Corpus already exists at {corpus_path}. Skipping corpus processing.", print_message=True)

    def process_queries(self):
        queries_path = os.path.join(self.dataset_dir, "queries.jsonl")
    
        if not os.path.exists(queries_path) or self.overwrite:
            querystrings = {}
            with tarfile.open(os.path.join(self.download_dir, "queries.tar.gz"), 'r:gz') as tar:
                for member in tar.getmembers():
                    if member.isfile():
                        f = tar.extractfile(member)
                        if f is not None:
                            queries_in_content = f.readlines()
                            for i, line in enumerate(tqdm(queries_in_content)):
                                query = line.decode("utf-8").rstrip().split("\t")
                                qid = query[0]
                                querystr = query[1]
                                querystrings[qid] = querystr
            
            log_message("Writing queries to disk")
            queries_filepah = os.path.join(self.download_dir, "queries.jsonl")

            with open(queries_filepah, "w") as f:
                for qid, query in tqdm(querystrings.items()):
                    json.dump({"_id": qid, "text": query, "metadata": {}}, f)
                    f.write("\n")

        else:
            log_message(f"Queries already exist at {queries_path}. Skipping query processing.", print_message=True)

    def process_qrels(self):
        log_message(f"Processing qrels into {self.qrel_dir}.", print_message=True)

        skipped_doc_path = os.path.join(self.download_dir, "skipped_docs.csv")
        skipped_docs = pd.read_csv(skipped_doc_path, header=None)[0].to_list()
        
        for subset in self.subsets:
            qrel_filepath = os.path.join(self.qrel_dir, self.norm_subset_name[subset]+".tsv")

            if not os.path.exists(qrel_filepath) or self.overwrite:
                subset_qrel = {}
                with gzip.open(os.path.join(self.download_dir, f"qrels.{subset}.tsv"), 'rt', encoding='utf8') as f:
                    tsvreader = csv.reader(f, delimiter="\t")
                    for item in tsvreader:
                        topicid, _, docid, rel = item[0].split(" ")
                        assert rel == "1"
                        if docid not in skipped_docs: # Skip  documents without text
                            subset_qrel[topicid] = docid

                log_message("Writing qrels to disk")

                with open(qrel_filepath, "w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f, delimiter="\t", lineterminator='\n')
                    
                    writer.writerow(["query-id", "corpus-id", "score"])
                    
                    for qid, docid in subset_qrel.items():
                        writer.writerow([qid, docid, 1])
            else:
                log_message(f"Qrels for {subset} already exist at {qrel_filepath}. Skipping qrel processing.", print_message=True)
               
if __name__ == "__main__":
    args = parse_arguments()
    processor = MSMarcoProcessor(args.datapath, args.overwrite)
    
    processor.download()
    processor.process_corpus()
    processor.process_queries()
    processor.process_qrels()


    short_processor = MSMarcoShortProcessor(args.datapath, args.overwrite)
    
    short_processor.download()
    short_processor.process_corpus()
    short_processor.process_queries()
    short_processor.process_qrels()
