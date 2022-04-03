#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to test the attention layer in the transformer model
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__copyright__ = 'Copyright © 2022 by Christoph Kirst'


import torch

import aid.model.transformer as tf

import matplotlib.pyplot as plt

# notes: 
# attention (reference poistion, attneeiton to other elements in seuqnece)
# mask: True at places to mask out 

#%% Set up attention layer

b = 3;   # batch size
h = 2;   # heads
d = 4;   # dimension for each head
d_model = d * h;

a = tf.MultiHeadedAttention(d_model=d_model, n_heads=h, max_relative_position = None, dropout = None, mask_future = True)
a0 =  tf.MultiHeadedAttention(d_model=d_model, n_heads=h, max_relative_position = None, dropout = None, mask_future = False)

m = a.future_mask(5);

plt.figure(1); plt.clf();
plt.imshow(m.data.numpy()[0,0].T, origin='lower')


#%% set linear encodings to identity 

def to_identity(a):
    a.linear_key.weight.data[:] = torch.diag(torch.ones(d*h))
    a.linear_value.weight.data[:] = torch.diag(torch.ones(d*h))
    a.linear_query.weight.data[:] = torch.diag(torch.ones(d*h))
    
    a.linear_key.bias.data[:] = torch.zeros(d*h)
    a.linear_value.bias.data[:] = torch.zeros(d*h)
    a.linear_query.bias.data[:] = torch.zeros(d*h)
    
to_identity(a)
to_identity(a0)

#%% generate simple input sequence

l = 4
x = torch.zeros((b,l,d_model))
x[0,:,0] = torch.arange(1,l+1)
xbh = x.reshape(b,l,h,d)  # batched / separated into heads
print(xbh[0,:,0,:])
print((b,l,h,d))

#%% pass through attention layer - maksed and unmasked

y = a.forward(x, x, x)

ybh = y.reshape((b,l,h,d))
print(ybh)

attn = a.attention;
print(attn.shape)
print(torch.allclose(attn.sum(dim=-1),torch.ones(attn.sum(dim=-1).shape)))
# dime -1 is the summation index along sequence to obtain value, dime -2 is the poisiton index in the sequence (self-attnetion index)

ym = a0.forward(x, x, x,)
ymbh = ym.reshape((b,l,h,d))


attn_masked = a0.attention;
print(torch.allclose(attn_masked.sum(dim=-1),torch.ones(attn_masked.sum(dim=-1).shape)))
# dime -1 is the summation index along sequence to obtain value, dime -2 is the poisiton index in the sequence (self-attnetion index)

m = a.future_mask(l)

plt.figure(2); plt.clf();

plt.subplot(2,3,1)
plt.imshow(xbh[0,:,0,:].data.numpy().T, origin = 'lower')
plt.xlabel('sequence'); plt.ylabel('head dimension')
plt.title('input seuence')

plt.subplot(2,3,2)
plt.imshow(ybh[0,:,0,:].data.numpy().T, origin = 'lower')
plt.title('output sequence masked')

plt.subplot(2,3,3)
plt.imshow(attn[0,0,:,:].data.numpy().T, origin = 'lower')
plt.xlabel('sequence position (self-attention)'); plt.ylabel('attention to other symbols')
plt.title('attention masked')

plt.subplot(2,3,4);
plt.imshow(m.data.numpy()[0,0].T, origin='lower')
plt.title('mask')

plt.subplot(2,3,5)
plt.imshow(ymbh[0,:,0,:].data.numpy().T, origin = 'lower')
plt.title('output sequence unmasked')

plt.subplot(2,3,6)
plt.imshow(attn_masked[0,0,:,:].data.numpy().T, origin = 'lower')
plt.xlabel('sequence position'); plt.ylabel('attention to other symbols')
plt.title('attention unmasked')

plt.tight_layout()


#%% attention masking and ignored symbol (padding)

# attention is (seuqnece position of query, attention to other symbols)
# thus to blank out ignored symbols maks should be set as:
    
code = torch.tensor([1,2,3,4,2,6])
ignore_code = 2;
l = len(code)

mask = torch.zeros((l,l), dtype=torch.bool)
mask[:,code == ignore_code] = True
#mask[code==ignore_code,:]   = True

plt.figure(3); plt.clf();
plt.imshow(mask.data.numpy().T, origin='lower')


#%% Relative Positional Encdoing

h = 2;
L = 10;
d = 3;

#Er = torch.zeros((h,L,d))
Er = torch.zeros((L,d))

for r in range(L):
    Er[...,r,:] = -r 

b = 3;
l = 7;
q = torch.zeros((b,h,l,d))


#%% 
def relative_positions(seq_len):
    result = []
    for i in range(seq_len):
        front = list(range(-i, 0))
        end = list(range(seq_len - i))
        result.append(front + end)
    return result

print(relative_positions(5))


#%% test relative poistoin encoding -> put index in the embedding

import aid.model.transformer as tf

b = 3;
h = 2;

d = 5;
L = 11;

Er = torch.zeros((L,d))
for r in range(L):
    Er[...,r,:] = -r 

Er0 = torch.zeros((2*L-1,d))
for i,r in enumerate(range(-L+1,L)):
    Er0[...,i,:] = r 

#Er[...,L-1,:] = +1  # just to mark relative 0 index with a non zero value

a = tf.MultiHeadedAttention(d_model = h*d, n_heads = h, max_relative_position = L, dropout = None, mask_future = True)
a0 = tf.MultiHeadedAttention(d_model = h*d, n_heads = h, max_relative_position = L, dropout = None, mask_future = False)

def set_weights(a, Er):
    a.relative_position.data[:] = Er[:]
    
    a.linear_key.weight.data[:] = torch.diag(torch.ones(d*h))
    a.linear_value.weight.data[:] = torch.diag(torch.ones(d*h))
    a.linear_query.weight.data[:] = torch.diag(torch.ones(d*h))
    
    a.linear_key.bias.data[:] = torch.zeros(d*h)
    a.linear_value.bias.data[:] = torch.zeros(d*h)
    a.linear_query.bias.data[:] = torch.zeros(d*h)
    
set_weights(a, Er);
set_weights(a0, Er0);


#%%

plt.figure(10); plt.clf();
plt.imshow(Er.detach().numpy().T, origin='lower')

l = 7;
er = a.relative_position_representation(l, a.relative_position)
er0 = a0.relative_position_representation(l, a0.relative_position)

plt.figure(11); plt.clf();
plt.subplot(1,2,1)
plt.imshow(er.detach().numpy().T, origin='lower')
plt.subplot(1,2,2)
plt.imshow(er0.detach().numpy().T, origin='lower')

#%% relative poisiton represenation for attention

l = 4
q = torch.zeros((b,l,h*d))
q[:,:,1] = 1
k = torch.zeros((b,l,h*d))
#print(q.reshape(b,l,h,d))

z = a(q, q, q)

m  = a.future_mask(l)
print(a.srel.masked_fill(m, 0)[0])

z0 = a0(q, q, q)
print(a0.srel[0])

# is ok

#%% relative position representation for value

L= 6;

a = tf.MultiHeadedAttention(d_model = h*d, n_heads = h, value_relative_position = True, max_relative_position = L, dropout = None, mask_future = True)
a0 = tf.MultiHeadedAttention(d_model = h*d, n_heads = h, value_relative_position = True, max_relative_position = L, dropout = None, mask_future = False)


Er = torch.zeros((L,d))
for r in range(L):
    Er[r,:] = -r 
for i in range(L):
    Er[i,:] += -(10**(i+1)) * torch.arange(d)    


Er0 = torch.zeros((2*L-1,d))
for i,r in enumerate(range(-L+1,L)):
    Er0[i,:] = r   
for i in range(2*L-1):
    Er0[i,:] += -(10**(i+1)) * torch.arange(d)   

def set_weights(a, Er):
    #a.relative_position.data[:] = Er[:]
    a.relative_position.data[:] = 0;
    a.value_relative_position.data[:] = Er[:]
    
    a.linear_key.weight.data[:] = torch.diag(torch.ones(d*h))
    a.linear_value.weight.data[:] = 0
    a.linear_query.weight.data[:] = torch.diag(torch.ones(d*h))
    
    a.linear_key.bias.data[:] = torch.zeros(d*h)
    a.linear_value.bias.data[:] = torch.zeros(d*h)
    a.linear_query.bias.data[:] = torch.zeros(d*h)
    
set_weights(a, Er);
set_weights(a0, Er0);



#%%
l = 4
q = torch.zeros((b,l,h*d))
q[:,:,1] = 1
k = torch.zeros((b,l,h*d))
#print(q.reshape(b,l,h,d))

z = a(q, q, q)

m  = a.future_mask(l)
#print(a.srel.masked_fill(m, 0)[0])

z0 = a0(q, q, q)
#print(a0.srel[0])

print(z.reshape(b,l,h,d).transpose(1,2)[0,0])

#%%

seq_len = 4;
vr = a.relative_position_representation(seq_len, a.value_relative_position);
if a.mask_future:
    vr = torch.cat([vr.flip(0), torch.zeros((seq_len-1,a.d_head))], dim=0);

attn = a.attention

z = torch.zeros(b,h,seq_len, d)
for l in range(seq_len):
    z[...,l,:] += torch.matmul(attn[...,l,:], vr[(seq_len -1 - l):(seq_len -1 - l + seq_len),:])

print(z[0,0])


#%%
vr = a.relative_position_representation(seq_len, a.value_relative_position);
if a.mask_future:
    vr = torch.cat([vr.flip(0), torch.zeros((seq_len-1,a.d_head))], dim=0);
for i in range(l):
    vr[i,:] += -(10**(i+1)) * torch.arange(d) 

vvr = torch.stack([vr[(seq_len -1 - l):(seq_len -1 - l + seq_len),:] for l in range(seq_len)])

cvr = vr.flip(0);
cvr = cvr.unfold(dimension=0, size=seq_len, step=1)
cvr = cvr.transpose(-1,-2).flip(-2)

print(torch.all(vvr == cvr))


#%% Attention and masking

b = 2;
l = 8;
h = 1;
d = 4;

a = tf.MultiHeadedAttention(d_model = h*d, n_heads = h, value_relative_position = False, max_relative_position = None, dropout = None, mask_future = True)

def set_weights(a):
    a.linear_key.weight.data[:] = torch.diag(torch.ones(d*h))
    a.linear_value.weight.data[:] = torch.diag(torch.ones(d*h))
    a.linear_query.weight.data[:] = torch.diag(torch.ones(d*h))
    
    a.linear_key.bias.data[:] = torch.zeros(d*h)
    a.linear_value.bias.data[:] = torch.zeros(d*h)
    a.linear_query.bias.data[:] = torch.zeros(d*h)
    
set_weights(a);

# data
q = torch.zeros((b,l,h*d))
q[:,:,1] = 1
k = torch.zeros((b,l,h*d))

#mask
ignore_code = 3;
mask = (torch.arange(l).unsqueeze(0) == ignore_code).data.unsqueeze(-2).unsqueeze(1) 
                                                        
                                                        
z = a.forward(q,q,q, mask=mask)                                                        
print(z[0,0])                                                     

plt.figure(6); plt.clf();
plt.subplot(1,3,1); 
plt.imshow(mask[0,0].data.numpy().T, origin='lower')
plt.subplot(1,3,2);
plt.imshow(a.future_mask(l)[0,0].data.numpy().T, origin='lower')
plt.subplot(1,3,3)
plt.imshow(a.attention[0,0].data.numpy().T, origin='lower')


#%% test masking - not as the q sequence position will be dropped via ignore index in the loss function only the attened to (k) index is masked out

test = torch.randn(a.attention.shape)
plt.figure(7); plt.clf();
plt.subplot(1,2,1)
plt.imshow(test.data.numpy()[0,0].T, origin='lower')
plt.subplot(1,2,2)

att = test.masked_fill(mask, float('-inf'));
import torch.nn.functional as F
att = F.softmax(att, dim=-1)

plt.imshow(att[0,0].data.numpy().T, origin='lower')







#%% clipboard 


# n = 4;
# delta = torch.cat([torch.arange(-n+1,n) * i for i in range(1,n+1)]).reshape(n,2*n-1)

# absl = delta.reshape(-1)[n-1:n*(n+n-2)+n-1].reshape(n, n+n-2)[:,:n]
# print(absl)


#     @staticmethod
#     def circulant(matrix, dim = -1, rotate_right = True):
#         n = matrix.size(dim);
#         if rotate_right:
#             matrix = matrix.flip(dim);
#         circulant = torch.cat([matrix, torch.narrow(matrix, dim=dim, start=0, length=n-1)], dim=dim);
#         circulant = circulant.unfold(dim, n, 1);
#         if rotate_right:
#             circulant = circulant.flip(dim);
#         return circulant;
