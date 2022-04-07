"""
Music Transformer
=================

A Transformer model based on the MusicTransformer to create an 
aritifical intelligent drummer.

References
----------
Music Transformer 
https://arxiv.org/abs/1809.04281

transformer architecture 
https://arxiv.org/abs/1706.03762

relative position representations  
https://arxiv.org/abs/1803.02155

reference for training implementations
https://nlp.seas.harvard.edu/2018/04/03/attention.html
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__copyright__ = 'Copyright © 2022 by Christoph Kirst'

import math
import copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn.parameter import Parameter


def clone(module, n):
    """Clone modules n times."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def average(model, models):
    "Average models into model"
    for ps in zip(*[m.params() for m in [model] + models]):
        ps[0].copy_(torch.sum(*ps[1:]) / len(ps[1:]))


class Embedding(nn.Module):
    """Embedding of discrete tokens into a vector space."""
    def __init__(self, n_tokens, d_model):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(n_tokens, d_model)
        self.factor = np.sqrt(d_model);

    def forward(self, x):
        return self.embedding(x) * self.factor


# class LayerNorm(nn.Module):
#     "Layernorm module"
#     def __init__(self, d_model, eps=1e-6):
#         super(LayerNorm, self).__init__()
#         self.a_2 = nn.Parameter(torch.ones(d_model))
#         self.b_2 = nn.Parameter(torch.zeros(d_model))
#         self.eps = eps

#     def forward(self, x):
#         mean = x.mean(-1, keepdim=True)
#         std = x.std(-1, keepdim=True)
#         return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
 
LayerNorm = nn.LayerNorm   
  
  
class MultiHeadedAttention(nn.Module):
    """Multi head attention network with optional relative position representation."""
    def __init__(self, d_model, n_heads,
                 max_relative_position = None, 
                 add_relative_position_to_value = False, 
                 dropout = 0.1, 
                 mask_future = True, 
                 save_attention = False):
        super(MultiHeadedAttention, self).__init__()
        if (d_model % n_heads != 0):
            raise ValueError('Model dimension d_model=%d must be divisible by the number of attention heads n_heads=%d!' % (d_model, n_heads))

        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_query = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)
        
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None;

        if max_relative_position is not None:
            self.max_relative_position = max_relative_position
            if self.mask_future:
               self.relative_position = Parameter(torch.rand((max_relative_position, self.d_head), dtype=torch.float32))               
               #self.mask_relative_position = self.future_mask(max_relative_position)
            else:
               self.relative_position = Parameter(torch.rand((2*(max_relative_position - 1) + 1, self.d_head), dtype=torch.float32))
                
            if add_relative_position_to_value:
                if self.mask_future:
                   self.value_relative_position = Parameter(torch.rand((max_relative_position, self.d_head), dtype=torch.float32))
                else:
                   self.value_relative_position = Parameter(torch.rand((2*(max_relative_position -1) + 1, self.d_head), dtype=torch.float32)) 
            else:
               self.value_relative_position = None
            
        else:
            self.relative_position = None  
            self.value_relative_position = None

        #storage for visualization etc.
        self.save_attention = save_attention;
        #self.save_srel = save_srel;
        self.attention = None
        #self.srel = None;
    
    
    def forward(self, query, key, value, mask = None):
        "Mulithead attention."
        # querry.shape = (n_batch, seq_len, d_model)
        n_batch = query.size(0)
        d_head  = self.d_head
        n_heads = self.n_heads
        seq_len = query.size(1)
        
        # masking
        if self.mask_future is True:
            #if self.relative_position is not None and seq_len <= self.max_relative_position:
            #    mask_future = self.mask_relative_position[:,:,:seq_len, :seq_len]; 
            #else:
            mask_future = self.future_mask(seq_len);
        else:
            mask_future = None;
        
        if mask is not None: #and mask_future is not None:
            #mask.shape = (n_batch, seq_len)
            #mask_future.shape = (1,1,seq_len,seq_len)
            
            # alternative 1: precalculate mask
            #mask = mask.unsqueeze(1).unsqueeze(1);
            ##mask.shape = (n_batch, 1, 1, seq_len)
            #mask = torch.logical_or(mask,  mask_future);
            ##mask.shape = (n_batch, 1, seq_len, seq_len); 
            
            #alternative 2: separate smaller masks
            mask = mask.unsqueeze(1).unsqueeze(1);
            ##mask.shape = (n_batch, 1, 1, seq_len)
        else:
            #alternative 1:
            #mask = mask_future;
            
            #alternative 2:
            mask = None;
        
        # linear projection + distribution over heads: d_model => h x d_head 
        shape = (n_batch, seq_len, n_heads, d_head)
        query = self.linear_query(query).reshape(*shape).transpose(1,2)
        value = self.linear_value(value).reshape(*shape).transpose(1,2)
        #query.shape = value.shape = (n_batch, n_heads, seq_len, d_head)        
        key   = self.linear_key(  key  ).reshape(*shape).permute(0, 2, 3, 1)
        #key.shape = (n_batch, n_heads, d_head, seq_len)

        # q-k attention
        #print(query.shape, key.shape)
        attn = torch.matmul(query, key) 
        #scroes.shape = (n_batch, n_heads, seq_len, seq_len)

        # relative position representation 
        if self.relative_position is not None:
            er = self.relative_position_representation(seq_len, self.relative_position);
            #er.shape = (seq_len, d_head)  or (2*seq_delta+1, d_head)
            er = er.transpose(0,1);
            #er.shape = (d_head, seq_len) or  (d_head, 2*seq_delta+1)
        
            # Q E^T
            qe = torch.matmul(query, er);
            # qe.shape = (n_batch, n_head, seq_len, seq_len) or (n_batch, n_head, seq_len, 2*seq_delta+1)
            
            if self.mask_future:
                # reslice to get srel (assumes row mayor ordering)
                # skew procdure: https://arxiv.org/abs/1809.04281 
                qe = qe.flip(-1)
                qe = F.pad(qe, (1,0)) 
                qe = torch.reshape(qe.contiguous(), (n_batch, n_heads, seq_len+1, seq_len))
                #srel = qe[:, :, 1:, :]
                #srel.shape = (n_batch, n_heads, seq_len, seq_len)
                attn += qe[:, :, 1:, :];
            else:
                # reslice to get srel (assumes row mayor ordering)
                qe = qe.contiguous().reshape(n_batch, n_heads, -1)
                qe = qe[:,:,seq_len-1:seq_len*(2*seq_len-2)+seq_len-1];
                qe = qe.reshape(n_batch, n_heads,seq_len,2*seq_len-2);
                #srel = qe[...,:seq_len];
                #srel.shape = (n_batch, n_heads, seq_len, seq_len)
                attn += qe[...,:seq_len]

            #save srel
            #if self.save_srel:
            #    self.srel = srel;
    
            #add relative position encoding to attention score
            #attn += srel
            #srel = None;  # free some memory
                
        # attention
        attn /= math.sqrt(d_head)     
        if mask is not None:
            attn.masked_fill_(mask, float("-inf"));
        # alternative 2 for masking:
        if mask_future is not None:
            attn.masked_fill_(mask_future, float("-inf"));
        
        attn = F.softmax(attn, dim = -1)
        #att.shape = (n_batch, n_heads, seq_len, seq_len)
        # (seq_len, seq_lem ) = (query index = output sequence position, key index = attended positoin in input sequence))
        
        #save attention
        if self.save_attention:
            self.attention = attn;
        
        # value
        z = torch.matmul(attn, value) 
        # z.shape = (n_batch, n_heads, seq_len, d_head)
        
        # add relative position representation to z
        if self.value_relative_position is not None:
            vr = self.relative_position_representation(seq_len, self.value_relative_position);
            if self.mask_future:
               vr = torch.cat([vr.flip(0), torch.zeros((seq_len-1,self.d_head), device=self.device())], dim=0);
            #vr.shape = (d_head, 2*seq_len-1)
 
            # alternative 1: n matrix multiplications 
            #seq_delta = seq_len - 1;
            #for l in range(seq_len):
            #    z[...,l,:] += torch.matmul(attn[...,l,:], vr[(seq_delta - l):(seq_delta - l + seq_len),:])
            
            # alternative 2: circulant view
            vr = vr.flip(0);
            vr = vr.unfold(dimension=0, size=seq_len, step=1);
            #vr.shape = (seq_len (q), d_head, seq_len (k))
            vr = vr.transpose(-1,-2).flip(-2);
            #vr.shape = (seq_len (q), 1, seq_len (k), d_head)   
            zr = torch.matmul(attn.unsqueeze(-2), vr);   #attn.unsqueeze(-2).shape = (n_batch, n_heads, seq_len (q), 1, seq_len (k)) -> to ensure broadcasting of dot product
            #zr.shape = (n_batch, n_heads, d_heads, seq_len (q), 1, d_head)
            zr = zr.squeeze(-2);
            #zr.shape = (n_batch, n_heads, d_heads, seq_len, d_head)
            z += zr;
               
        z = z.transpose(1, 2).contiguous()
        z = z.reshape(n_batch, seq_len, -1);
        #z.shape = (n_batch, seq_len, d_model)
        #z = z.view(n_batch, -1, self.n_heads * self.d_k)
        
        if self.dropout:
            z = self.dropout(z);
        
        return z;
    
    def device(self):
        device = next(self.parameters()).device;
        return device;
    
    def future_mask(self, size):
        """Mask to not attend to future positions in sequence."""
        return torch.triu(torch.ones(size, size, dtype=torch.bool, device=self.device(), requires_grad = False), diagonal=1).unsqueeze(0).unsqueeze(0);

    def relative_position_representation(self, seq_len, relative_position):
        """Clip/pad the relative positional embedding to fit sequence."""
        max_rel = self.max_relative_position;
        if self.mask_future:
            if seq_len <= max_rel:
                er = relative_position[:seq_len]
            else:
                er = torch.zeros((seq_len, self.d_head), device=self.device());
                er[:max_rel,:] = relative_position[:, :]
                er[max_rel:,:] = relative_position[-1,:]        
        else: # past and future attention
             max_delta = max_rel - 1;
             seq_delta = seq_len - 1;
             if seq_delta <= max_delta:
                er = relative_position[(max_delta-seq_delta) : (max_delta+seq_delta+1)]
             else:
                er = torch.zeros((2*seq_delta+1, self.d_head), device=self.device());
                er[(seq_delta-max_delta):(seq_delta+max_delta+1),:] = relative_position[:, :]
                er[:(seq_delta-max_delta),:] = relative_position[0, :]   
                er[(seq_delta+max_delta):,:] = relative_position[-1, :] 
                
        return er;
        

class FeedForward(nn.Module):
    """Feed forward network for a Transfomer layer."""
    def __init__(self, d_model, d_feedforward, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_feedforward)
        self.linear2 = nn.Linear(d_feedforward, d_model)

        
        if dropout is not None:
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
        else:
            self.dropout1 = None;
            self.dropout2 = None;
    
    def forward(self, x):
        x = F.relu(self.linear1(x));
        if self.dropout1:
            x = self.dropout1(x);
        x = self.linear2(x);
        if self.dropout2:
            x = self.dropout2(x);
        return x;


class PositionalEncoding(nn.Module):
    "Absolute positional encoding."
    def __init__(self, d_model, max_sequence, dropout = None):
        super(PositionalEncoding, self).__init__()
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None;
        
        pe = torch.zeros(max_sequence, d_model, requires_grad = False)
        position = torch.arange(0, max_sequence).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        if self.dropout:
            x = self.dropout(x)
        return x;


class EncoderLayer(nn.Module):
    """EncoderLayer for the transformer with self-attention and feed-forward network."""
    
    def __init__(self, d_model, attention, feedforward, norm_first = False):
        super(EncoderLayer, self).__init__()
        self.attention = attention
        self.feedforward = feedforward
        self.d_model = d_model
        self.norm_first = norm_first
        self.norm_attention = LayerNorm(d_model);
        self.norm_feedforward = LayerNorm(d_model);

    def forward(self, x, mask = None):
        """Single multihead attention encoding layer."""
        if self.norm_first:
            x = self.norm_attention(x);
            x = x + self.attention(x, x, x, mask)
            x = self.norm_feedforward(x)
            x = x + self.feedforward(x)
        else:
            x = x + self.attention(x, x, x, mask)            
            x = self.norm_attention(x)
            x = x + self.feedforward(x)
            x = self.norm_feedforward(x)
            
        return x


class Encoder(nn.Module):
    """"Encoder for Transformer via n layers with multihead attention."""
    def __init__(self, layer, n_layers, norm = True):
        super(Encoder, self).__init__()
        self.layers = clone(layer, n_layers)
        if norm:
            self.norm = LayerNorm(layer.d_model)
        else:
            self.norm = None;
        
    def forward(self, x, mask = None):
        "Forward input through layers."
        for layer in self.layers:
            x = layer(x, mask=mask)
        if self.norm:
            x = self.norm(x);
        return x


class Generator(nn.Module):
    def __init__(self, d_model, n_tokens, softmax = True, logits = True):
        super(Generator, self).__init__()
        self.linear = nn.Linear(d_model, n_tokens);
        self.logits = logits;
        if self.logits and softmax is False:
            softmax = True;
        if softmax is True:
            if self.logits:
                self.softmax = F.log_softmax;
            else:
                self.softmax = F.softmax;         
        else:
            self.softmax = False;
    
    def forward(self, x):
        x = self.linear(x);
        if self.softmax:
            x = self.softmax(x, dim=-1);
        return x;
    

class Transformer(nn.Module):
    """Transformer network to transform a sequence of tokens into a future sequence of tokens."""
    def __init__(self, encoder, embedding, generator, criterion = None):
        super(Transformer, self).__init__()
        self.encoder   = encoder
        self.embedding = embedding
        self.generator = generator;
        if criterion is not None:
            self.criterion = criterion;
      
    @property
    def n_tokens(self):
        return self.embedding.embedding.weight.shape[0];
    
    @property
    def n_src_tokens(self):
        return self.embedding.embedding.weight.shape[0];   
    
    @property
    def n_tgt_tokens(self):
        return self.generator.linear.weight.shape[0]; 
      
    @property
    def d_model(self):
        return self.embedding.embedding.weight.shape[1];
      
    def forward(self, x, mask = None):
        """Transform sequence into scores for future sequence."""
        return self.generator(self.encode(x, mask))
     
    def encode(self, x, mask = None):
        return self.encoder(self.embedding(x), mask)
    
    def decode(self, y, method = 'max'):
        """Convert output of forward into a token sequence."""
        if method == 'max':
            _, sequence = torch.max(y, dim = -1)       
        elif method == 'random':
            if self.generator.logits:
                distrib = torch.distributions.categorical.Categorical(logits=y)
            else:
                if not self.generator.softmax:
                    y = F.softmax(y, dim=-1);
                distrib = torch.distributions.categorical.Categorical(probs=y)
            sequence = distrib.sample()
        return sequence;
     
    def loss(self, src, tgt, mask = None):
        return self.criterion(self.forward(src, mask=mask), tgt);
    
    def generate(self, primer = None, sequence_length = 1024, model_sequence_length = None, method = 'random', verbose = True, 
                       start_token = 1, end_token = None, ignore_token = None, n_tokens = None,
                       dtype = torch.long, softmax = True, beam_search = 4, return_probabilities = False):
        """Generate sequence from optional primer.
        
        Arguments
        ---------
        method : random | beam | max
           Generate output symbols by drawing form the random output distribution of the model, using beam search via greedy maximum probability.
        """
        assert (not self.training), "not in training mode."
        
        if verbose is True:
            verbose = 10;
        
        if verbose:
            print("generating sequence via method=%s of max_sequence_length=%d" % (method, sequence_length))

        device = self.device();
        
        if n_tokens is None:
            n_tokens = self.n_tokens;

        sequence = torch.full((1, sequence_length), ignore_token, dtype=dtype, device=device)

        if model_sequence_length is None:
            model_sequence_length = sequence_length;

        if primer is not None:
           n_primer = len(primer)
           sequence[..., :n_primer] = primer.type(dtype).to(device)
        else:
           n_primer = 1;   
           sequence[..., :1] = start_token;
           
        if return_probabilities:
            probs = torch.full((1, sequence_length, n_tokens), 0, dtype=torch.float, device=device)
            probs[0,torch.arange(n_primer),sequence[..., :n_primer]] = 1;  
        
        if softmax is True:
            softmax = nn.Softmax(dim=-1)
        
        # generation loop
        s = n_primer
        while(s < sequence_length):
            s0 = max(0, s - model_sequence_length);
            y = self.forward(sequence[..., s0:s]);
            
            if n_tokens is not None:
                y = y[..., :n_tokens]
            if softmax:
                y = softmax(y);
            prob_tokens = y[:, -1, :]
            
            if return_probabilities: # shifted by one as forward shifts everyhting by one
                probs[:,s,:] = prob_tokens;

            if method == 'max':
                _, next_token = torch.max(prob_tokens, dim = -1)
                sequence[:, s] = next_token;
                
            elif method == 'random':
                distrib = torch.distributions.categorical.Categorical(probs=prob_tokens)
                next_token = distrib.sample()
                sequence[:, s] = next_token;
                
            elif method == 'beam':
                n_tokens = prob_tokens.size(-1);
                prob_tokens = prob_tokens.flatten()
                top_res, top_idx = torch.topk(prob_tokens, beam_search)

                beam_rows = top_idx // n_tokens
                beam_cols = top_idx % n_tokens

                sequence = sequence[beam_rows, :]
                sequence[..., s] = beam_cols
                
                if return_probabilities:
                    probs = probs[beam_rows, :];
                
                next_token = None

            else:
                raise ValueError("method = %r not in ['random', 'beam', 'max']" % method);

            if end_token is not None and (next_token == end_token):
                if verbose:
                    print("Model called end of sequence at:", s, "/", sequence_length)
                break

            s += 1
            if verbose and (s % verbose == 0):
                print(s, "/", sequence_length)

        sequence = sequence[:, :s];
        if return_probabilities:
            probs = probs[:,:s];
        if (primer is not None and primer.ndim == 1) and method != 'beam':
            sequence = sequence.squeeze(0);
            if return_probabilities:
                probs = probs.squeeze(0); 
        
        result = (sequence,)
        if return_probabilities:
            result += (probs,);
        if len(result) == 1:
            result = result[0];
        return result;

    @classmethod
    def create(cls,
               n_tokens     = None,
               n_src_tokens = None, 
               n_tgt_tokens = None,
               n_layers = 6, 
               d_model = 512, 
               d_feedforward = 2048, 
               n_heads = 8, 
               max_sequence_length = 2048, 
               dropout=0.1, 
               dropout_attention = None, 
               max_relative_position = True, 
               add_relative_position_to_value = False, 
               mask_future = True,
               save_attention = False,
               save_srel = False):
        """Construct Transformer model from hyperparameters."""
        if n_tokens is not None:
            n_src_tokens = n_tgt_tokens = n_tokens;
        if n_tgt_tokens is None:
            n_tgt_tokens = n_src_tokens;
        
        if max_relative_position is not None:
            max_relative_position = max_sequence_length;
            embedding = Embedding(n_src_tokens, d_model)
        else:
            position  = PositionalEncoding(d_model, dropout)
            embedding = nn.Sequential(Embedding(n_src_tokens, d_model), position)
        
        attention   = MultiHeadedAttention(d_model=d_model, n_heads=n_heads, 
                                           max_relative_position=max_relative_position, 
                                           add_relative_position_to_value=add_relative_position_to_value, 
                                           dropout=dropout_attention, 
                                           mask_future=mask_future, 
                                           save_attention=save_attention, 
                                           save_srel=save_srel)
        
        feedforward = FeedForward(d_model, d_feedforward, dropout=dropout)
        
        encoder = Encoder(EncoderLayer(d_model, attention, feedforward, dropout), n_layers)
        
        generator = Generator(d_model, n_tgt_tokens)
        
        model = cls(encoder, embedding, generator)

        model.reset_parameter()
        
        return model;
    
    def reset_parameter(self):
        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def sequence_mask(size: int) -> Tensor:
        """Square mask for a sequence to not attend to the future. 
        
        masked   = float('-inf').
        unmasked = float(0.0).
        """
        return torch.triu(torch.full((size, size), float('-inf')), diagonal=1)

    def device(self):
        device = next(self.parameters()).device;
        return device;



class Optimizer:
    """Optimer wrapper with learning rate dynamics used for Transformer training."""
    def __init__(self, d_model, factor, warmup, optimizer, epoch = 0):
        self.optimizer = optimizer
        self.warmup = warmup
        self.factor = factor
        self.d_model = d_model
        self.epoch = epoch
        
    def step(self):
        """Update learning rate and parameters."""
        self.set_rate();
        self.optimizer.step()
        
    def zero_grad(self):
        self.optimizer.zero_grad();
        
    def set_epoch(self, epoch):
        """Start at specified step."""
        self.epoch = epoch;
        self.set_rate();
        
    def step_epoch(self):
        self.epoch += 1;
        self.set_rate();
    
    def set_rate(self):
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
             
    def rate(self, epoch = None):
        """Learning rate dynamics.
        
        lr = factor / (d_model)**0.5  * min(step**-0.5, step * warump**(-1.5))
        
        Reference
        ---------
        https://arxiv.org/abs/1706.03762
        """
        if epoch is None:
            epoch = self.epoch
        if epoch <= 0:
            if epoch < 0:
                print('Warning: epoch = %r' % epoch);
            epoch = 1;
        return self.factor *  (self.d_model ** (-0.5) * min(epoch ** (-0.5), epoch * self.warmup ** (-1.5)))
 
    @classmethod
    def create(cls, model, factor = 2, warmup = 4000, optimizer = None, betas = (0.9, 0.98), eps = 1e-9):
        d_model = model.d_model
        if optimizer is None:
            optimizer =  torch.optim.Adam(model.parameters(), lr=0, betas=betas, eps=eps)
        return cls(d_model=d_model, factor=factor, warmup=warmup, optimizer=optimizer);



class SmoothLabelLoss(nn.Module):
    """Smooth labele loss"""
    
    __constants__ = ['smoothing', 'n_vocab', 'ignore_token', 'reduction', 'input_is_logits']

    def __init__(self, n_tokens, smoothing = 0.1, ignore_token = None, reduction = 'mean', log_source = True, log_target = False, save_target_smoothed = False):
        super(SmoothLabelLoss, self).__init__()
        
        assert 0.0 <= smoothing <= 1.0
         
        self.smoothing = smoothing;
        self.confidence = 1 - smoothing;
        self.n_tokens = n_tokens
        
        self.ignore_token = ignore_token
        self.log_source = log_source
        self.log_target = log_target
        self.reduction=reduction
        
        self.save_target_smoothed = save_target_smoothed
        self.target_smoothed = None
        
        if self.ignore_token is not None:
            self.uniform = 1/ (self.n_tokens - 1);
        else:
            self.uniform = 1/ self.n_tokens;


    def forward(self, src, tgt):
        if self.smoothing:
           p = F.one_hot(tgt.long(), self.n_tokens).type(torch.float32)
           q = self.uniform;
           tgt_smoothed = self.confidence * p + self.smoothing * q;
        else:
           tgt_smoothed = tgt;

        if self.save_target_smoothed:
            self.target_smoothed = tgt_smoothed;

        # if self.ignore_token is not None:
        #     if self.smoothing:
        #         mask = (tgt == self.ignore_token).unsqueeze(-2);
        #         tgt_smoothed = tgt_smoothed.masked_fill(mask, 0)
        #     else:
        #         keep = tgt != self.ignore_token;
        #         src = src[keep];
        #         tgt_smoothed = tgt_smoothed[keep];
    
        crit = self.criterion(src, tgt_smoothed)
        if self.reduction == 'mean':
            if self.ignore_token is not None:
                n = torch.sum(tgt != self.ignore_token)
            else:
                n = tgt.numel()
            loss = crit.sum() / n
        elif self.reduction == 'sum':
            loss = crit.sum()
        elif self.reduction == 'none' or self.reduction is None:
            loss = crit;
        else:
            raise NotImplementedError('reduction = %r' % self.reduction);

        return loss;
      
    def cirterion(self, src, tgt):
        raise NotImplementedError();

     
class SmoothKLDivLoss(SmoothLabelLoss):
    """Smooth KL divergence loss to use with Transformer training."""
 
    def __init__(self, n_tokens, ignore_token = None, smoothing = 0.0, reduction = 'mean', log_source = True, log_target = False, save_target_smoothed = False): 
        super(SmoothKLDivLoss, self).__init__(n_tokens=n_tokens, ignore_token=ignore_token, smoothing=smoothing, log_source=log_source, log_target=log_target, save_target_smoothed=save_target_smoothed);
        self.loss = nn.KLDivLoss(reduction='none', log_target=log_target);

    def criterion(self, src, tgt):
        if not self.log_source:
            src = torch.log(src); 
        if self.log_target:
            tgt = torch.exp(tgt);
        return self.loss(src, tgt)
  
    
class SmoothCrossEntropyLoss(SmoothLabelLoss):
    """Smooth cross entropy loss to use with Transformer training."""
 
    def __init__(self, n_tokens, ignore_token = None, smoothing = 0.0, reduction = 'mean', log_source = True, log_target = False, save_target_smoothed = False): 
        super(SmoothCrossEntropyLoss, self).__init__(n_tokens=n_tokens, ignore_token=ignore_token, smoothing=smoothing, log_source=log_source, log_target=log_target, save_target_smoothed=save_target_smoothed);
        self.loss = nn.CrossEntropyLoss(reduction = 'none', ignore_index=ignore_token);

    def criterion(self, src, tgt):
        if not self.log_source:
            src = torch.log(src);
        if self.log_target:
            tgt = torch.exp(tgt);
        return self.loss(src, tgt);

           
Loss = SmoothCrossEntropyLoss
  

class Batch:
    "Batch of data with mask for training."
    def __init__(self, src, tgt=None, ignore_token = None):
        self.src = src
        self.tgt = tgt;
        self.ignore_token = ignore_token;    
        self.n_batch = src.size(0);

    def src_mask(self, src = None):
        if src is None:
            src = self.src;
        return self.mask(src, self.ignore_token);
    
    def tgt_mask(self, tgt = None):
        if tgt is None:
            tgt = self.tgt;
        return self.mask(tgt, self.ignore_token);
    
    def src_attention_mask(self, src = None):
        if src is None:
            src = self.src; 
        return self.attention_mask(src, self.ignore_token);
        
    def tgt_attention_mask(self, tgt = None):
        if tgt is None:
            tgt = self.tgt;
        return self.attention_mask(tgt, self.ignore_token);
    
    def n_src_tokens(self, src = None):
        if src is None:
            src = self.src; 
        return self.n_tokens(src, self.ignore_token)
    
    def n_tgt_tokens(self, tgt = None):
        if tgt is None:
            tgt = self.tgt;
        return self.n_tokens(tgt, self.ignore_token) 
    
    def batch_size(self):
        return self.src.shape[0];
        
    @staticmethod
    def n_tokens(matrix, ignore_token = None):
        if matrix is None:
            return 0;
        if ignore_token is not None:
            return (matrix != ignore_token).data.sum()
        else:
            return matrix.numel(); 
    
    @staticmethod
    def mask(matrix, ignore_token = None):
        if ignore_token is not None:
            mask = (matrix == ignore_token).data
        else:
            mask = None;
            #mask = torch.zeros((1,1), dtype=torch.bool);  
        return mask

    @staticmethod
    def attention_mask(matrix, ignore_token = None, with_attention_heads = True):
        # attention mask is (..., ref/query position, attend to position)
        if ignore_token is not None:
            mask = (matrix == ignore_token).data.unsqueeze(-2) # shape=(batch_size, 1, seq_len)
        else:
            mask = None;
            #mask = torch.zeros((1,1), dtype=torch.bool).unsqueeze(0);
        if with_attention_heads:
            mask = mask.unsqueeze(1) 
            #mask shape = (batch_size, 1, 1, seq_len)
        return mask
  
    def __repr__(self):
        r = "Batch(%d, %d)" % (self.src.shape[0], self.src.shape[1]);
        if self.ignore_token:
            r += '[%d]' % self.ignore_token;
        return r;



