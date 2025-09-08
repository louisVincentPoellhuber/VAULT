import torch
from transformers import BertModel
from transformers.models.bert.modeling_bert import BertEmbeddings,BertLayer,BertOnlyMLMHead,BertPreTrainedModel
from torch import Tensor, nn
from enhancedDecoder import BertLayerForDecoder

# Mean pooling useful for encoder based models with Mean Pooling usually used with cosine similarity and normalization.
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# CLS pooling useful for encoder based models with CLS Pooling usually used with dot-product similarity.
def cls_pooling(model_output, attention_mask=None):
    return model_output.last_hidden_state[:, 0, :]  # CLS token is first token


class BlockLevelContextawareEncoder(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.text_encoding_layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.information_exchanging_layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.ablation_config = kwargs.get("ablation_config", {"inter_block_encoder": True, "doc_token": True})

    def forward(
        self,
        hidden_states,
        attention_mask,
        reduce_hidden_states,
        node_mask,
    ):
        _, L_, D = hidden_states.shape
        B, _, _, N_ = node_mask.shape
        N=N_-1
        for i, layer_module in enumerate(self.text_encoding_layer):
            if (i>0) or (not self.ablation_config["inter_block_encoder"]):
                layer_outputs = layer_module(hidden_states, attention_mask, output_attentions=self.config.output_attentions)
            else:
                temp_attention_mask = attention_mask.clone()
                temp_attention_mask[:,:,:,0] = -10000.0
                layer_outputs = layer_module(hidden_states, temp_attention_mask, output_attentions=self.config.output_attentions)
                reduce_hidden_states=reduce_hidden_states[None,:,:].repeat(B,1,1)

            hidden_states = layer_outputs[0]

            if self.ablation_config["inter_block_encoder"]:
                hidden_states = hidden_states.view(B, N, L_, D)
                cls_hidden_states = hidden_states[:, :, 1, :].clone()

                reduce_cls_hidden_states=torch.cat([reduce_hidden_states,cls_hidden_states],dim=1) #[B,N+1,D]
                station_hidden_states = self.information_exchanging_layer[i](reduce_cls_hidden_states, node_mask, output_attentions=self.config.output_attentions)[0]
                reduce_hidden_states = station_hidden_states[:,:1,:]
                hidden_states[:, :, 0, :] = station_hidden_states[:,1:,:]
                hidden_states = hidden_states.view(B * N, L_, D)

        return (reduce_hidden_states, hidden_states, )

class Longtriever(BertModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, add_pooling_layer=False)
        self.config = config
        self.ablation_config = kwargs.get("ablation_config", {"inter_block_encoder": True, "doc_token": True})
        self.embeddings = BertEmbeddings(config)
        self.encoder = BlockLevelContextawareEncoder(config, **kwargs)
        
        self.doc_token_init = kwargs.get("doc_token_init", "default")
        if self.doc_token_init == "cls":
            self.doc_embeddings = None
        else:
            self.doc_embeddings = nn.Embedding(1,config.hidden_size).weight #[1,D]
            if self.doc_token_init == "default":
                self.doc_embeddings.data.uniform_(-1e-20, 1e-20) # Initialize near zero
            elif self.doc_token_init == "zero":
                self.doc_embeddings.data.zero_()

        # Initialize weights and apply final processing
        self.post_init()

    def get_extended_attention_mask(self, attention_mask: Tensor) -> Tensor:
        #station_mask==0? for attention_mask==0
        if self.ablation_config["doc_token"]:
            station_mask = torch.ones((attention_mask.shape[0],1),dtype=attention_mask.dtype,device=attention_mask.device) # [B*N,1]
            attention_mask = torch.cat([station_mask,attention_mask],dim=1) # [B*N,1+L]
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(
        self,
        input_ids,
        attention_mask,
        sentence_mask=None,
        return_last_hiddens=False
    ):
        if sentence_mask is None:
            sentence_mask = (torch.sum(attention_mask,dim=-1)>0).to(dtype=attention_mask.dtype)

        sentence_mask = torch.cat([torch.ones_like(sentence_mask[:,:1]),sentence_mask],dim=1)

        input_shape = input_ids.size()
        batch_size, sent_num, seq_length = input_shape

        input_ids = input_ids.view(batch_size*sent_num,seq_length)
        attention_mask = attention_mask.view(batch_size*sent_num,seq_length)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask) #[B*N,1,1,1+L]
        extended_sentence_mask = (1.0 - sentence_mask[:, None, None, :]) * -10000.0

        embedding_output = self.embeddings(input_ids=input_ids)
        
        if self.ablation_config["doc_token"]:
            station_placeholder = torch.zeros((embedding_output.shape[0], 1, embedding_output.shape[-1]),dtype=embedding_output.dtype,device=embedding_output.device)
            embedding_output = torch.cat([station_placeholder, embedding_output], dim=1)  # [B*N,1+L,D]
            
            encoder_outputs = self.encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                reduce_hidden_states=self.doc_embeddings,
                node_mask=extended_sentence_mask,
            )
        else:            
            encoder_outputs = self.encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                reduce_hidden_states=self.doc_embeddings,
                node_mask=extended_sentence_mask,
            )

        # Pooling depending on context
        if self.ablation_config["inter_block_encoder"]:
            text_vec = encoder_outputs[0].squeeze(1)
        elif not self.ablation_config["inter_block_encoder"] and self.ablation_config["doc_token"]:
            text_vec = encoder_outputs[1][:,1,:] 
        elif not self.ablation_config["inter_block_encoder"] and not self.ablation_config["doc_token"]:
            text_vec = encoder_outputs[1][:,0,:] 

        if not return_last_hiddens:
            return text_vec
        else:
            last_hiddens=encoder_outputs[1].view(batch_size,sent_num,seq_length+1,self.config.hidden_size)
            return (text_vec, last_hiddens) #[B,D] and #[B,N,L+1,D]


class LongtrieverForPretraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = Longtriever(config)
        self.cls = BertOnlyMLMHead(config)

        self.decoder_embeddings = self.bert.embeddings
        self.c_head = BertLayerForDecoder(config)
        self.c_head.apply(self._init_weights)

        self.cross_entropy = nn.CrossEntropyLoss()

        self.post_init()

    def forward(self, encoder_input_ids_batch, encoder_attention_mask_batch, encoder_labels_batch, decoder_input_ids_batch,
                decoder_matrix_attention_mask_batch, decoder_labels_batch,):

        batch_size, sent_num, seq_length = encoder_input_ids_batch.shape

        text_vec, last_hiddens=self.bert(encoder_input_ids_batch,encoder_attention_mask_batch,return_last_hiddens=True)
        last_hiddens=last_hiddens[:,:,1:,:]
        _, encoder_mlm_loss = self.mlm_loss(last_hiddens,encoder_labels_batch)

        cls_hiddens = last_hiddens[:, :, :1, :] # [B,N,1,D]
        doc_embeddings = text_vec[:,None,None,:].repeat(1,sent_num,1,1) #[B,N,1,D]

        decoder_embedding_output = self.decoder_embeddings(input_ids=decoder_input_ids_batch.view(batch_size*sent_num,seq_length)).view(batch_size,sent_num,seq_length,self.config.hidden_size) # [B,N,L,D]
        hiddens = torch.cat([doc_embeddings,cls_hiddens, decoder_embedding_output[:, :, 1:, :]], dim=2).view(batch_size*sent_num,seq_length+1,self.config.hidden_size) #[B*N,L+1,D]

        decoder_position_ids = self.bert.embeddings.position_ids[:, :seq_length]
        decoder_position_embeddings = self.bert.embeddings.position_embeddings(decoder_position_ids) # [1,L,D]
        query = (decoder_position_embeddings[:,None,:,:] + doc_embeddings).view(batch_size*sent_num,seq_length,self.config.hidden_size) #[B*N,L,D]

        decoder_matrix_attention_mask_batch=decoder_matrix_attention_mask_batch.view(batch_size*sent_num,seq_length,seq_length) #[B*N,L,L]
        decoder_matrix_attention_mask_batch=torch.cat([decoder_matrix_attention_mask_batch.new_full((batch_size*sent_num,seq_length,1),1),decoder_matrix_attention_mask_batch],dim=2) #[B*N,L,L+1]
        matrix_attention_mask = self.get_extended_attention_mask(
            decoder_matrix_attention_mask_batch,
            decoder_matrix_attention_mask_batch.shape,
            decoder_matrix_attention_mask_batch.device
        )

        hiddens = self.c_head(query=query,
                              key=hiddens,
                              value=hiddens,
                              attention_mask=matrix_attention_mask)[0]
        _, decoder_mlm_loss = self.mlm_loss(hiddens, decoder_labels_batch)

        return (encoder_mlm_loss+decoder_mlm_loss, )

    def mlm_loss(self, hiddens, labels):
        pred_scores = self.cls(hiddens)
        masked_lm_loss = self.cross_entropy(
            pred_scores.view(-1, self.config.vocab_size),
            labels.view(-1)
        )
        return pred_scores, masked_lm_loss
