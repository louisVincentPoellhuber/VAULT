from utils import *

import sys
csv.field_size_limit(sys.maxsize)
class WikIRProcessor(DatasetProcessor):
    def __init__(self, datapath, overwrite):
        super().__init__(datapath, "wikir", overwrite)
        self.subsets = ["training", "test", "validation"]
        self.norm_subset_name = {
            "training": "train",
            "test": "test",
            "validation": "val"
        }
        log_message("This dataset uses a 'training', 'test' and 'validation' split. These will be renamed to 'train', 'test' and 'val' respectively for consistency.", print_message=True)
     
    def download(self):
        file_url = "https://zenodo.org/records/3707238/files/enwikIRS.zip?download=1"

        download_path = os.path.join(self.download_dir, "DORIS-enwikIRS.zip")

        if not os.path.exists(download_path) or self.overwrite:
            with requests.get(file_url, stream=True) as r:
                r.raise_for_status()
                with open(download_path, 'wb') as f:
                    for chunk in tqdm(r.iter_content(chunk_size=1024*1024)):
                        f.write(chunk) 
        else:
            log_message(f"File already exists at {download_path}. Skipping download.", print_message=True)

    def process_corpus(self):
        download_path = os.path.join(self.download_dir, "DORIS-enwikIRS.zip")
        corpus_path = os.path.join(self.dataset_dir, "corpus.jsonl")

        if not os.path.exists(corpus_path) or self.overwrite:
            with zipfile.ZipFile(download_path, 'r') as zf, \
                open(corpus_path, 'w', encoding='utf-8') as corpus_file:
                
                with zf.open('enwikIRS/documents.csv') as f:
                    reader = csv.reader(io.TextIOWrapper(f, 'utf-8'))
                    # Skip header
                    next(reader)
                    for row in tqdm(reader, desc="Processing corpus documents"):
                        obj = {
                            "_id": row[0],
                            "text": row[1], 
                            "title": "" # There are no titles in this corpus
                        }
                        corpus_file.write(json.dumps(obj) + "\n")
        else:
            log_message(f"Corpus already exists at {corpus_path}. Skipping corpus processing.", print_message=True)

    def process_queries(self):
        download_path = os.path.join(self.download_dir, "DORIS-enwikIRS.zip")
        queries_path = os.path.join(self.dataset_dir, "queries.jsonl")
    
        if not os.path.exists(queries_path) or self.overwrite:
            log_message(f"Processing queries into {queries_path}.", print_message=True)
            with zipfile.ZipFile(download_path, 'r') as zf, \
                open(queries_path, 'w', encoding='utf-8') as queries_file:
                
                for subset in self.subsets:
                    with zf.open(f'enwikIRS/{subset}/queries.csv') as f:
                        reader = csv.reader(io.TextIOWrapper(f, 'utf-8'))
                        # Skip header
                        next(reader)
                        for row in reader:
                            obj = {
                                "_id": row[0],
                                "text": row[1]
                            }
                            queries_file.write(json.dumps(obj) + "\n")
        else:
            log_message(f"Queries already exist at {queries_path}. Skipping query processing.", print_message=True)

    def process_qrels(self):
        download_path = os.path.join(self.download_dir, "DORIS-enwikIRS.zip")
        log_message(f"Processing qrels into {self.qrel_dir}.", print_message=True)

        with zipfile.ZipFile(download_path, 'r') as zf:
            for subset in self.subsets:
                qrel_path = os.path.join(self.qrel_dir, f"{self.norm_subset_name[subset]}.tsv")

                if not os.path.exists(qrel_path) or self.overwrite:
                    
                    with zf.open(f'enwikIRS/{subset}/qrels') as f, \
                     open(qrel_path, 'w', encoding='utf-8') as qrel_file:
                        reader = csv.reader(io.TextIOWrapper(f, 'utf-8'))

                        for row in reader:
                            row = row[0].strip().split('\t')
                            if row[3] == "1":
                                qrel_file.write(f"{row[0]}\t{row[2]}\t{row[3]}\n")
                else:
                    log_message(f"Qrels for {subset} already exist at {qrel_path}. Skipping qrel processing.", print_message=True)


if __name__ == "__main__":
    args = parse_arguments()
    processor = WikIRProcessor(args.datapath, args.overwrite)
    
    processor.download()
    processor.process_corpus()
    processor.process_queries()
    processor.process_qrels()
