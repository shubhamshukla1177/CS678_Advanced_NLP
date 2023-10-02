from typing import Dict, List, Optional, Union, Tuple, Callable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_bert import BertPreTrainedModel
from utils import *


class BertSelfAttention(nn.Module):
  # This function is already provided for you. No change is required.
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # initialize the linear transformation layers for key, value, query
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # this dropout is applied after calculating the attention score following the original implementation of transformer
    # although it is a bit unusual, we empirically observe that it yields better performance
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  # This function is already provided for you. No change is required.
  def transform(self, x, linear_layer):
    # the corresponding linear_layer of k, v, q are used to project the hidden_state (x)
    bs, seq_len = x.shape[:2]
    proj = linear_layer(x)
    # next, we need to produce multiple heads for the proj
    # this is done by spliting the hidden state to self.num_attention_heads, each of size self.attention_head_size
    proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)
    # by proper transpose, we have proj of [bs, num_attention_heads, seq_len, attention_head_size]
    proj = proj.transpose(1, 2)
    return proj

  # TODO : Complete this function step by step.
  def attention(self, key, query, value, attention_mask):
    # General instructions :
    ######################################################################################
    # each attention is calculated following eq (1) of https://arxiv.org/pdf/1706.03762.pdf
    # attention scores are calculated by multiply query and key
    # and get back a score matrix S of [bs, num_attention_heads, seq_len, seq_len]
    # S[*, i, j, k] represents the (unnormalized)attention score between the j-th and k-th token, given by i-th attention head
    # before normalizing the scores, use the attention mask to mask out the padding token scores
    # Note again: in the attention_mask non-padding tokens with 0 and padding tokens with a large negative number
    ######################################################################################

    # Step 1: Calculate S = QK^T
    Q = query
    K = key
    V = value
    S = torch.matmul(Q, K.transpose(-1,-2))
    S = S/math.sqrt(self.attention_head_size)
    # Step 2: Apply the mask to S
    
    # Step 3: Normalize the scores
    S = S + attention_mask
    
    # Step 4: Apply softmax to get attention probabilities
    att_prob = F.softmax(S, dim=-1)
    
    # Step 5: Apply dropout to the attention to get the final attention scores
    att_prob = self.dropout(att_prob)
    
    # Step 6: Multiply the attention scores to the value and get back V'
    att_val = torch.matmul(att_prob, V)
    
    # Step 7: Concat the multi-heads and recover the original shape [bs, seq_len, num_attention_heads * attention_head_size = hidden_size]
    att_val = att_val.transpose(1,2).contiguous()
    
    # Step 8: Return V' in the shape expected by the caller, i.e [bs, seq_len, hidden_size]
    att_val = att_val.view(att_val.shape[0],att_val.shape[1],self.all_head_size)
    
    return att_val
    #raise NotImplementedError # remove this line when the function is implemented

  # This function is already provided for you. No change is required.
  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # first, we have to generate the key, value, query for each token for multi-head attention w/ transform (more details inside the function)
    # of *_layers are of [bs, num_attention_heads, seq_len, attention_head_size]
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    # calculate the multi-head attention
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value


class BertLayer(nn.Module):
  # This function is already provided for you. No change is required.
  def __init__(self, config):
    super().__init__()
    # self attention
    self.self_attention = BertSelfAttention(config)
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # feed forward
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # layer out
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)


  # TODO : Complete this function step by step.
  def add_norm(self, input, output, dense_layer, dropout, ln_layer):
    """
    input: the input
    output: the input that requires the sublayer to transform
    dense_layer, dropput: the sublayer
    ln_layer: layer norm that takes input+sublayer(output)
    """
    # Step 1: Pass output to dense layer
    dense_op = dense_layer(output)
    # Step 2: Apply dropout to output of dense layer
    dense_op = dropout(dense_op)
    # Step 3: Add output of dense layer to the input
    dense_op_ip = input + dense_op
    # Step 4: Apply layer norm to the output of the add-norm layer
    dense_op_ip = ln_layer(dense_op_ip)
    # Step 5: Return the output of the layer norm
    return dense_op_ip
    #raise NotImplementedError # remove this line when the function is implemented
    

  # TODO : Complete this function step by step.
  def forward(self, hidden_states, attention_mask):
    # General description of different inputs you would need at different steps
    # of this function.
    """
    hidden_states: either from the embedding layer (first bert layer) or from the previous bert layer
    as shown in the left of Figure 1 of https://arxiv.org/pdf/1706.03762.pdf
    each block consists of
    1. a multi-head attention layer (BertSelfAttention)
    2. a add-norm that takes the output of BertSelfAttention and the input of BertSelfAttention
    3. a feed forward layer
    4. a add-norm that takes the output of feed forward layer and the input of feed forward layer
    """

    # Step 1: Get the output of the multi-head attention layer using self.self_attention
    att_op = self.self_attention(hidden_states, attention_mask)
   
    # Step 2: Apply add-norm layer using self.add_norm
    att_op = self.add_norm(hidden_states, att_op, self.attention_dense, self.attention_dropout, self.attention_layer_norm)
    
    # Step 3: Get the output of feed forward layer using self.interm_dense
    feedfwd_op = self.interm_dense(att_op)
    
    # Step 4: Apply activation function to the output of feed forward layer using self.interm_af
    feedfwd_op = self.interm_af(feedfwd_op)
    
    # Step 5: Apply another add-norm layer using self.add_norm
    feedfwd_op = self.add_norm(att_op, feedfwd_op, self.out_dense, self.out_dropout, self.out_layer_norm)
    
    # Step 6: Return the output of this add-norm layer
    #raise NotImplementedError # remove this line when the function is implemented
    return feedfwd_op

class BertModel(BertPreTrainedModel):
  """
  the bert model returns the final embeddings for each token in a sentence
  it consists
  1. embedding (used in self.embed)
  2. a stack of n bert layers (used in self.encode)
  3. a linear transformation layer for [CLS] token (used in self.forward, as given)
  """
  # This function is already provided for you. No change is required.
  def __init__(self, config):
    super().__init__(config)
    self.config = config

    # embedding
    self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
    self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    self.tk_type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
    self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
    # position_ids (1, len position emb) is a constant, register to buffer
    position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
    self.register_buffer('position_ids', position_ids)

    # bert encoder
    self.bert_layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    # for [CLS] token
    self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.pooler_af = nn.Tanh()

    self.init_weights()

  # TODO : Complete this function step by step.
  def embed(self, input_ids):
    input_shape = input_ids.size()
    seq_length = input_shape[1]

    # Step 1 : Get word embedding using self.word_embedding
    inputs_embeds = self.word_embedding(input_ids)

    # get position index and position embedding from self.pos_embedding
    pos_ids = self.position_ids[:, :seq_length]

    # Step 2 : Get position embedding using self.pos_embedding
    pos_embeds = self.pos_embedding(pos_ids)

    # get token type ids, since we are not consider token type, just a placeholder
    tk_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
    tk_type_embeds = self.tk_type_embedding(tk_type_ids)

    # add three embeddings together
    embeds = inputs_embeds + tk_type_embeds + pos_embeds

    # layer norm and dropout
    embeds = self.embed_layer_norm(embeds)
    embeds = self.embed_dropout(embeds)

    # Step 3 : Return the embeddings
    #raise NotImplementedError # remove this line when the function is implemented
    return embeds

  # This function is already provided for you. No change is required.
  def encode(self, hidden_states, attention_mask):
    """
    hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
    attention_mask: [batch_size, seq_len]
    """
    # get the extended attention mask for self attention
    # returns extended_attention_mask of [batch_size, 1, 1, seq_len]
    # non-padding tokens with 0 and padding tokens with a large negative number
    extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)

    # pass the hidden states through the encoder layers
    for i, layer_module in enumerate(self.bert_layers):
      # feed the encoding from the last bert_layer to the next
      hidden_states = layer_module(hidden_states, extended_attention_mask)

    return hidden_states

  def forward(self, input_ids, attention_mask):
    """
    input_ids: [batch_size, seq_len], seq_len is the max length of the batch
    attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
    """
    # get the embedding for each input token
    embedding_output = self.embed(input_ids=input_ids)

    # feed to a transformer (a stack of BertLayers)
    sequence_output = self.encode(embedding_output, attention_mask=attention_mask)

    # get cls token hidden state
    first_tk = sequence_output[:, 0]
    first_tk = self.pooler_dense(first_tk)
    first_tk = self.pooler_af(first_tk)

    return {'last_hidden_state': sequence_output, 'pooler_output': first_tk}
