from beir.retrieval.search.dense import FlatIPFaissSearch
import importlib.util
import logging
import os
import numpy as np
from tqdm import tqdm

if importlib.util.find_spec("faiss") is not None:
    import faiss

from beir.retrieval.search.dense.faiss_index import FaissBinaryIndex, FaissHNSWIndex, FaissIndex, FaissTrainIndex
logger = logging.getLogger(__name__)


class StreamedFlatIPFaissSearch(FlatIPFaissSearch):
    
    def _create_mapping_ids(self, corpus_ids):
        for idx in range(len(corpus_ids)):
            self.mapping[corpus_ids[idx]] = idx
            self.rev_mapping[idx] = corpus_ids[idx]

    
    def index(self, corpus: dict[str, dict[str, str]], score_function: str = None, **kwargs):
        faiss_ids, corpus_embeddings = self._index(corpus, score_function, **kwargs)
        base_index = faiss.IndexFlatIP(self.dim_size)
        if self.use_gpu:
            logger.info("Moving Faiss Index from CPU to GPU...")
            gpu_base_index = faiss.index_cpu_to_gpu(self.single_gpu, 0, base_index)
            self.faiss_index = FaissIndex.build(faiss_ids, corpus_embeddings, gpu_base_index)
        else:
            self.faiss_index = FaissIndex.build(faiss_ids, corpus_embeddings, base_index)
            

    def _index(self, corpus: dict[str, dict[str, str]], score_function: str = None):
        corpus_ids = corpus.ids
        self._create_mapping_ids(corpus_ids)
        normalize_embeddings = True if score_function == "cos_sim" else False

        logger.info("Encoding Corpus in batches... Warning: This might take a while!")

        itr = range(0, len(corpus), self.corpus_chunk_size)

        for batch_num, corpus_start_idx in enumerate(itr):
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))
            batch_ids = corpus_ids[corpus_start_idx:corpus_end_idx]
            batch = [corpus[corpus_id] for corpus_id in batch_ids]
            logger.info(f"Encoding Batch {batch_num + 1}/{len(itr)}. Normalize: {normalize_embeddings}...")

            # Encode chunk of corpus
            sub_corpus_embeddings = self.model.encode_corpus(
                batch,
                batch_size=self.batch_size,
                show_progress_bar=True,
                normalize_embeddings=normalize_embeddings,
            )

            if not batch_num:
                corpus_embeddings = sub_corpus_embeddings
            else:
                corpus_embeddings = np.vstack([corpus_embeddings, sub_corpus_embeddings])

        # Index chunk of corpus into faiss index
        logger.info("Indexing Passages into Faiss...")

        faiss_ids = [self.mapping.get(corpus_id) for corpus_id in corpus_ids]
        self.dim_size = corpus_embeddings.shape[1]

        del sub_corpus_embeddings

        return faiss_ids, corpus_embeddings
