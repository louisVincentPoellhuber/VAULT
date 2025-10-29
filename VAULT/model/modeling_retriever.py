from .modeling_utils import *
import torch
from torch import Tensor, nn
import torch.distributed as dist
import torch.nn.functional as F
from beir.retrieval.models.util import extract_corpus_sentences
from tqdm import trange
from .data_handler import DataCollatorForEvaluatingLongtriever, DataCollatorForEvaluatingBert

def compute_multi_passage_loss(loss_fct, co_query_embeddings, co_corpus_embeddings, **kwargs):
    co_query_embeddings = co_query_embeddings.squeeze(1)
    total_loss = 0.0
    num_passages = co_corpus_embeddings.size(1)
    
    for i in range(num_passages):
        passage_embeddings = co_corpus_embeddings[:, i, :]
        loss = loss_fct(co_query_embeddings, passage_embeddings, **kwargs)
        total_loss += loss
    
    avg_loss = total_loss / num_passages
    return avg_loss


def compute_contrastive_loss(co_query_embeddings, co_corpus_embeddings, **kwargs):
        
    similarities_1 = torch.matmul(co_query_embeddings, co_corpus_embeddings.transpose(0, 1))
    similarities_2 = torch.matmul(co_query_embeddings, co_query_embeddings.transpose(0, 1))
    similarities_2.fill_diagonal_(float('-inf'))

    # # If there are negative embeddings, compute their similarities and append them to the queries's similarities
    # co_neg_embeddings = kwargs.get("co_neg_embeddings", None)
    # if co_neg_embeddings is not None:
    #     similarities_3 = torch.matmul(co_query_embeddings, co_neg_embeddings.transpose(0, 1))
    #     similarities_2 = torch.cat([similarities_2, similarities_3], dim=1)

    similarities=torch.cat([similarities_1,similarities_2],dim=1)
    labels=torch.arange(similarities.shape[0],dtype=torch.long,device=similarities.device)
    co_loss = F.cross_entropy(similarities, labels) * dist.get_world_size()
    return co_loss

def compute_cross_entropy_loss(co_query_embeddings, co_corpus_embeddings, **kwargs):
    similarities = torch.matmul(co_query_embeddings, co_corpus_embeddings.transpose(0, 1))
    labels=torch.arange(similarities.shape[0],dtype=torch.long,device=similarities.device)

    # co_neg_embeddings = kwargs.get("co_neg_embeddings", None)
    # if co_neg_embeddings is not None:
    #     similarities_2 = torch.matmul(co_query_embeddings, co_neg_embeddings.transpose(0, 1))
    #     similarities = torch.cat([similarities, similarities_2], dim=1)

    co_loss = F.cross_entropy(similarities, labels) * dist.get_world_size()
    return co_loss


LOSS_FUNCTIONS = {
    "contrastive": compute_contrastive_loss,
    "cross_entropy": compute_cross_entropy_loss,
}


class BertRetriever(nn.Module):
    def __init__(self,
                 model,
                 data_collator=DataCollatorForEvaluatingBert("bert-base-uncased", 512, 512, 8), 
                 normalize=False, 
                 loss_function="contrastive", 
                 **kwargs):
        super().__init__()
        self.encoder=model
        # self.batch_size=batch_size
        self.sep=" [SEP] "
        self.data_collator = data_collator
        self.normalize = normalize
        self.loss_fct = LOSS_FUNCTIONS[loss_function]
        self.output_passage_embeddings = kwargs.get("output_passage_embeddings", False)

    def save_pretrained(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)

    def _gather_tensor(self, t: Tensor):
        all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(all_tensors, t)
        all_tensors[dist.get_rank()] = t
        return all_tensors

    def forward(self, query_input_ids, query_attention_mask, corpus_input_ids, corpus_attention_mask):
        query_embeddings=self.encoder(query_input_ids,query_attention_mask,return_dict=True).last_hidden_state[:, 0]
        corpus_embeddings=self.encoder(corpus_input_ids,corpus_attention_mask,return_dict=True).last_hidden_state[:, 0]
        co_query_embeddings = torch.cat(self._gather_tensor(query_embeddings.contiguous()))
        co_corpus_embeddings = torch.cat(self._gather_tensor(corpus_embeddings.contiguous()))
        co_loss = self.loss_fct(co_query_embeddings, co_corpus_embeddings)
        return (co_loss,)

    
    def encode_queries(self, queries, batch_size,**kwargs):
        query_embeddings = []
        verbose = kwargs.get("verbose", True)
        range_fct = trange if verbose else range
        with torch.no_grad():
            for start_idx in range_fct(0, len(queries), batch_size):
                sub_queries = queries[start_idx:start_idx + batch_size]
                if self.output_passage_embeddings:
                    query_outputs, query_index = self.tokenize(sub_queries)
                else:
                    query_outputs = self.tokenize(sub_queries)
                query_embeddings.append(query_outputs)

        co_query_embeddings = torch.cat(query_embeddings)

        if kwargs.get("normalize", self.normalize):
            query_embeddings = F.normalize(co_query_embeddings, p=2, dim=1)

        return co_query_embeddings.cpu()
    

    def encode_corpus(self, corpus, batch_size, **kwargs):
        corpus_embeddings = []
        corpus_index = []
        sentences = extract_corpus_sentences(corpus=corpus, sep=self.sep)

        verbose = kwargs.get("verbose", True)
        range_fct = trange if verbose else range
        with torch.no_grad():
            for start_idx in range_fct(0, len(sentences), batch_size):

                sub_sentences = sentences[start_idx : start_idx + batch_size]

                if self.output_passage_embeddings:
                    sub_corpus = corpus[start_idx : start_idx + batch_size]
                    ctx_outputs, ctx_index = self.tokenize(sub_sentences)
                    sub_corpus_index = [item["_id"] for item in sub_corpus]

                    # Dynamically detect example boundaries using index value 0
                    current_doc_idx = 0
                    combined_index = []
                    for i, block_idx in enumerate(ctx_index):
                        # When we see 0, we're starting a new document
                        if block_idx == 0 and i > 0:
                            current_doc_idx += 1
                        combined_index.append(f"{sub_corpus_index[current_doc_idx]}-{block_idx}")

                    corpus_index.extend(combined_index)
                    corpus_embeddings.append(ctx_outputs)
                else:
                    corpus_embeddings.append(self.tokenize(sub_sentences))

        co_corpus_embeddings = torch.cat(corpus_embeddings)
        
        # NOTE: It's actually called normalize_embeddings, but I want self.normalize to take priority
        if kwargs.get("normalize", self.normalize):
            corpus_embeddings = F.normalize(co_corpus_embeddings, p=2, dim=1)

        if self.output_passage_embeddings:
            return co_corpus_embeddings.cpu(), corpus_index
        
        return co_corpus_embeddings.cpu()
    
    
    def tokenize(self, sub_input):
        ctx_input = self.data_collator(sub_input)
        ctx_input_ids = ctx_input["input_ids"].to(self.encoder.device)
        ctx_attention_mask = ctx_input["attention_mask"].to(self.encoder.device)
        ctx_outputs = self.encoder(ctx_input_ids, ctx_attention_mask).last_hidden_state[:, 0]

        
        if self.output_passage_embeddings:
            ctx_index = ctx_input["index"]
            return ctx_outputs, ctx_index

        return ctx_outputs
    
    def rerank(self, query, corpus, batch_size=64, top_k=100, **kwargs):
        query_embedding = self.encode_queries([query], batch_size=batch_size, **kwargs)
        
        if self.output_passage_embeddings:
            corpus_embeddings, corpus_index = self.encode_corpus(corpus, batch_size=batch_size, rerank=True, **kwargs)

            scores = torch.matmul(query_embedding, corpus_embeddings.transpose(0, 1)).squeeze(0)

            # Combine corpus_index and scores in a dictionary
            docid_score_dict = {docid: scores[idx].item() for idx, docid in enumerate(corpus_index)}

            # Split docid to get the full docid (before the "-" separator)
            # Keep only top-k unique docids with highest scores
            seen_docids = {}
            for docid, score in sorted(docid_score_dict.items(), key=lambda x: x[1], reverse=True):
                full_docid = docid.split("-")[0]
                if full_docid not in seen_docids:
                    seen_docids[full_docid] = score
                    if len(seen_docids) >= top_k:
                        break
            ranked_docids = seen_docids
        else:
            corpus_embeddings = self.encode_corpus(corpus, batch_size=batch_size, rerank=True, **kwargs)
            scores = torch.matmul(query_embedding, corpus_embeddings.transpose(0, 1)).squeeze(0)
            # Sort by score (descending) and filter by top_k
            corpus_index = [item["_id"] for item in corpus]

            docid_score_dict = {docid: scores[idx].item() for idx, docid in enumerate(corpus_index)}
            ranked_docids = dict(sorted(docid_score_dict.items(), key=lambda x: x[1], reverse=True)[:top_k])

        return ranked_docids

    def eval(self):
        self.encoder.eval()
        self.encoder.to("cuda:0")
        self.training = False

class LongtrieverRetriever(BertRetriever):
    def forward(self, query_input_ids, query_attention_mask, corpus_input_ids, corpus_attention_mask, **kwargs):

        corpus_embeddings = self.encoder(corpus_input_ids, corpus_attention_mask)
        query_embeddings = self.encoder(query_input_ids, query_attention_mask)
        co_query_embeddings = torch.cat(self._gather_tensor(query_embeddings.contiguous()))
        co_corpus_embeddings = torch.cat(self._gather_tensor(corpus_embeddings.contiguous()))
    
        # TODO: Create custom loss function to handle multiple CLS tokens. It would most likely be a weighted sum of all the losses
        if len(co_corpus_embeddings.size())>2:
            co_loss = compute_multi_passage_loss(self.loss_fct, co_query_embeddings, co_corpus_embeddings)
        else:
            co_loss = self.loss_fct(co_query_embeddings, co_corpus_embeddings)
        return (co_loss,)
    
    def tokenize(self, sub_input):
        ctx_input = self.data_collator(sub_input)
        ctx_input_ids = ctx_input["input_ids"].to(self.encoder.device)
        ctx_attention_mask = ctx_input["attention_mask"].to(self.encoder.device)
        ctx_outputs = self.encoder(ctx_input_ids, ctx_attention_mask) # No last_hidden_state

        if self.output_passage_embeddings:
            nb_examples, nb_blocks, hidden_size = ctx_outputs.size()
            ctx_outputs = ctx_outputs.view(nb_examples * nb_blocks, hidden_size)
            ctx_index = torch.arange(nb_blocks).repeat(nb_examples)

            return ctx_outputs, ctx_index

        return ctx_outputs
    
    
class SiameseRetriever(BertRetriever):
    def __init__(self, 
                 ctx_encoder,
                 q_encoder,
                 data_collator=DataCollatorForEvaluatingBert("bert-base-uncased", 512, 512, 8), 
                 normalize=False, 
                 loss_function="contrastive"):
        super().__init__(None, data_collator, normalize, loss_function)

        self.encoder = None
        self.ctx_encoder = ctx_encoder
        self.q_encoder = q_encoder

    
    def tokenize(self, sub_input, encoder):
        input = self.data_collator(sub_input)
        input_ids = input["input_ids"].to(encoder.device)
        attention_mask = input["attention_mask"].to(encoder.device)
        outputs = encoder(input_ids, attention_mask).pooler_output

        return outputs
    
    def encode_corpus(self, corpus, batch_size, **kwargs):
        corpus_embeddings = []
        sentences = extract_corpus_sentences(corpus=corpus, sep=self.sep)

        with torch.no_grad():
            for start_idx in trange(0, len(sentences), batch_size):

                sub_sentences = sentences[start_idx : start_idx + batch_size]
            
                corpus_embeddings.append(self.tokenize(sub_sentences, self.ctx_encoder))

        co_corpus_embeddings = torch.cat(corpus_embeddings)
        
        # NOTE: It's actually called normalize_embeddings, but I want self.normalize to take priority
        if kwargs.get("normalize", self.normalize):
            corpus_embeddings = F.normalize(co_corpus_embeddings, p=2, dim=1)

        return co_corpus_embeddings.cpu()
    
    
    def encode_queries(self, queries, batch_size,**kwargs):
        query_embeddings = []
        with torch.no_grad():
            for start_idx in trange(0, len(queries), batch_size):
                sub_queries = queries[start_idx:start_idx + batch_size]
                query_embeddings.append(self.tokenize(sub_queries, self.q_encoder))

        co_query_embeddings = torch.cat(query_embeddings)

        if kwargs.get("normalize", self.normalize):
            query_embeddings = F.normalize(co_query_embeddings, p=2, dim=1)

        return co_query_embeddings.cpu()
    
    
    def eval(self):
        self.ctx_encoder.eval() 
        self.q_encoder.eval() 
        self.ctx_encoder.to("cuda:0") 
        self.q_encoder.to("cuda:0") 
        self.training = False