
import logging
import os
from accelerate import Accelerator, DistributedDataParallelKwargs
import time
from beir.retrieval.evaluation import EvaluateRetrieval
import pytrec_eval
import torch
from typing import List


JOBID = os.getenv("SLURM_JOB_ID")
if JOBID == None: # No slurm? Try to get the experiment name exported in a bash file
    JOBID = os.getenv("EXP_NAME")
    if JOBID == None: # No experiment name? Use local
        JOBID = "local"
    os.makedirs(f"src/longtriever/logs", exist_ok=True)
    log_path = f"src/longtriever/logs/{JOBID}.log"
else:
    log_path = f"slurm-{JOBID}.log"

# if JOBID == None: JOBID = "debug"
logging.basicConfig( 
    encoding="utf-8", 
    filename=log_path, 
    filemode="a", 
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level = logging.INFO
    )

MAIN_PROCESS = Accelerator().is_main_process
STORAGE_DIR = os.getenv("STORAGE_DIR")

def log_message(message, level=logging.WARNING, force_message = False, print_message = False):
    if force_message or MAIN_PROCESS:
        logging.log(msg=message, level=level)
        if print_message:
            print(message)

def tensorize_batch(sequences: List[torch.Tensor], padding_value, align_right=False) -> torch.Tensor:
    if len(sequences[0].size()) == 1:
        max_len_1 = max([s.size(0) for s in sequences])
        out_dims = (len(sequences), max_len_1)
        out_tensor = sequences[0].new_full(out_dims, padding_value)
        for i, tensor in enumerate(sequences):
            length_1 = tensor.size(0)
            if align_right:
                out_tensor[i, -length_1:] = tensor
            else:
                out_tensor[i, :length_1] = tensor
        return out_tensor
    elif len(sequences[0].size()) == 2:
        max_len_1 = max([s.size(0) for s in sequences])
        max_len_2 = max([s.size(1) for s in sequences])
        out_dims = (len(sequences), max_len_1, max_len_2)
        out_tensor = sequences[0].new_full(out_dims, padding_value)
        for i, tensor in enumerate(sequences):
            length_1 = tensor.size(0)
            length_2 = tensor.size(1)
            if align_right:
                out_tensor[i, -length_1:, -length_2:] = tensor
            else:
                out_tensor[i, :length_1, :length_2] = tensor
        return out_tensor
    elif len(sequences[0].size()) == 3:
        max_len_1 = max([s.size(0) for s in sequences])
        max_len_2 = max([s.size(1) for s in sequences])
        max_len_3 = max([s.size(2) for s in sequences])
        out_dims = (len(sequences), max_len_1, max_len_2, max_len_3)
        out_tensor = sequences[0].new_full(out_dims, padding_value)
        for i, tensor in enumerate(sequences):
            length_1 = tensor.size(0)
            length_2 = tensor.size(1)
            length_3 = tensor.size(2)
            if align_right:
                out_tensor[i, -length_1:, -length_2:, -length_3:] = tensor
            else:
                out_tensor[i, :length_1, :length_2, :length_3] = tensor
        return out_tensor
    else:
        raise

class CustomEvaluateRetrieval(EvaluateRetrieval):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    
    @staticmethod
    def evaluate(
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        k_values: list[int],
        ignore_identical_ids: bool = True,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
        if ignore_identical_ids:
            logging.info(
                "For evaluation, we ignore identical query and document ids (default), please explicitly set ``ignore_identical_ids=False`` to ignore this."
            )
            popped = []
            for qid, rels in results.items():
                for pid in list(rels):
                    if qid == pid:
                        results[qid].pop(pid)
                        popped.append(pid)

        ndcg = {}
        _map = {}
        recall = {}
        precision = {}
        mrr = {}

        mrr[f"MRR"] = 0.0
        for k in k_values:
            ndcg[f"NDCG@{k}"] = 0.0
            _map[f"MAP@{k}"] = 0.0
            recall[f"Recall@{k}"] = 0.0
            precision[f"P@{k}"] = 0.0

        map_string = "map_cut." + ",".join([str(k) for k in k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
        recall_string = "recall." + ",".join([str(k) for k in k_values])
        precision_string = "P." + ",".join([str(k) for k in k_values])
        mrr_string = "recip_rank." + ",".join([str(k) for k in k_values])
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string, mrr_string})
        scores = evaluator.evaluate(results)

        for query_id in scores.keys():
            mrr[f"MRR"] += scores[query_id]["recip_rank"]
            for k in k_values:
                ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
                _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
                recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
                precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]

        mrr[f"MRR"] = round(mrr[f"MRR"] / len(scores), 5)
        for k in k_values:
            ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
            _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
            recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
            precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)

        for eval in [ndcg, _map, recall, precision, mrr]:
            logging.info("\n")
            for k in eval.keys():
                huh = f"{k}: {eval[k]:.4f}"
                logging.info(f"{k}: {eval[k]:.4f}")

        return ndcg, _map, recall, precision, mrr
