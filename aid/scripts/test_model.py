#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer Tests
"""
__author__ = "Christoph Kirst"

import numpy as np
import matplotlib.pyplot as plt

#% Test - move forward task

import torch
from torch.utils.data import Dataset, DataLoader

from aid.dataset.groove import generate_source_target, collate_batch

CODE_SIZE = 128;
CODE_PAD = CODE_SIZE-1;

class CopyDataset(Dataset):

    def __init__(self, max_length = 32, code_size = CODE_SIZE, ignore_code = CODE_PAD, n_samples = 500, data_length = 128, random_sequence = True):
        super(CopyDataset, self).__init__()
        
        self.max_length      = max_length;
        self.code_siZe       = code_size;
        self.random_sequence = random_sequence;
        self.ignore_code     = ignore_code;
        
        if data_length is None:
            data_length = max_length;
        self.data = torch.randint(0, code_size, (n_samples, data_length));
        
        #make sure we do have pad symbols
        if self.ignore_code:
            idx = self.data == self.ignore_code;
            if self.ignore_code == 0:
                replace = 1;
            else:
                replace = 0;     
            self.data[idx] = replace;
        
    def __len__(self):
        """number of data files"""
        return self.data.shape[0];


    def __getitem__(self, idx):
        """Returns input and target sequence"""
        
        code = self.data[idx]
        src, _ = generate_source_target(code, self.max_length, self.random_sequence, ignore_code = self.ignore_code)

        return src, src


class ForwardDataset(CopyDataset):

    def __init__(self, **kwargs):
        super(ForwardDataset, self).__init__(**kwargs);

    def __getitem__(self, idx):
        """Returns input and target sequence"""
        
        code = self.data[idx]
        #reverse of standard src,tgt
        tgt, src = generate_source_target(code, self.max_length + 1, self.random_sequence, ignore_code = self.ignore_code)
        src = src[:-1];
        tgt = tgt[:-1];
        tgt[0] = src[0];

        return src, tgt


class SeriesDataset(Dataset):

    def __init__(self, max_length = 32, code_size = CODE_SIZE, ignore_code = CODE_PAD, n_samples = 500, data_length = 128, random_sequence = True):
        super(CopyDataset, self).__init__()
        
        self.max_length      = max_length;
        self.code_siZe       = code_size;
        self.random_sequence = random_sequence;
        self.ignore_code     = ignore_code;
        
        if data_length is None:
            data_length = max_length;
        self.data = torch.randint(0, code_size, (n_samples, data_length));
        
        #make sure we do have pad symbols
        if self.ignore_code:
            idx = self.data == self.ignore_code;
            if self.ignore_code == 0:
                replace = 1;
            else:
                replace = 0;     
            self.data[idx] = replace;
        
    def __len__(self):
        """number of data files"""
        return self.data.shape[0];


    def __getitem__(self, idx):
        """Returns input and target sequence"""
        
        code = self.data[idx]
        src, _ = generate_source_target(code, self.max_length, self.random_sequence, ignore_code = self.ignore_code)

        return src, src

#%

import functools as ft

collate_fn = ft.partial(collate_batch, ignore_code = CODE_PAD)
dataset = CopyDataset()
data = DataLoader(dataset, batch_size = 10, shuffle = True, collate_fn = collate_fn);

batch = next(iter(data));

plt.figure(1); plt.clf();
plt.subplot(2,1,1)
plt.imshow(batch.src.data.numpy(), origin='lower')
plt.title('source')
plt.subplot(2,1,2)
plt.imshow(batch.tgt.data.numpy(), origin='lower')
plt.title('target')


#% create model

from aid.model.run import create_model

model = create_model(n_src_vocab=CODE_SIZE, dropout = None)

#% generate untraiined sequence

src = batch.src[:1];

model.eval();

y = model(src)
_, tgt = torch.max(y, dim = -1)     

plt.figure(2); plt.clf();
plt.imshow([src[0].data.numpy(), tgt[0].data.numpy()], origin = 'lower')
plt.title('src vs max likely tgt')


#%% train

from aid.model.run import train

data_eval = DataLoader(dataset, batch_size = 32, shuffle = False, collate_fn = collate_fn);

model, optimizer, loss = train( epochs = 25,
                                model_parameter = dict(n_layers = 1),
                                loss_parameter = dict(n_src_vocab = CODE_SIZE, ignore_code = CODE_PAD),
                                data_train=data, n_train_batches = 32,
                                data_eval_train=data_eval, data_eval_test=data_eval, n_evaluate_test_batches = 5, n_evaluate_train_batches = 0,
                                use_tensorboard = True,
                                base_directory = '~/Desktop/test'
                              );


#%% test on sample

src = batch.src[:1];

model.eval();

y = model(src)
_, tgt = torch.max(y, dim = -1)     


plt.figure(2); plt.clf();
plt.imshow([src[0].data.numpy(), tgt[0].data.numpy()], origin = 'lower')



#%% train - forard 

from aid.model.run import train

dataset = ForwardDataset();
data = DataLoader(dataset, batch_size = 10, shuffle = True, collate_fn = collate_fn);
data_eval = DataLoader(dataset, batch_size = 32, shuffle = False, collate_fn = collate_fn);

batch = next(iter(data));

plt.figure(3); plt.clf();
plt.subplot(2,1,1)
plt.imshow(batch.src.data, origin='lower')
plt.title('src')
plt.subplot(2,1,2)
plt.imshow(batch.tgt.data, origin='lower')
plt.title('tgt')

#%% train forward model
model, optimizer, loss = train( epochs = 50,
                                model_parameter = dict(n_layers = 1),
                                loss_parameter = dict(n_src_vocab = CODE_SIZE, ignore_code = CODE_PAD),
                                data_train=data, n_train_batches = 32,
                                data_eval_train=data_eval, data_eval_test=data_eval, n_evaluate_test_batches = 5, n_evaluate_train_batches = 0,
                                use_tensorboard = True,
                                base_directory = '~/Desktop/test'
                              );

#%% check forward

import torch.nn.functional as F

src = batch.src[:1];

model.eval();

y = model(src)
_, tgt = torch.max(y, dim = -1)   


l0 = model.encoder.layers[0]
attn = l0.attention.attention[0]
seq_len = attn.shape[0]

plt.figure(2); plt.clf();
plt.subplot(2,seq_len,(1,seq_len//2));
p = F.softmax(y[0][:,:CODE_SIZE], dim=-1);
plt.imshow(p.data.T.numpy(), origin = 'lower')
plt.title('probabilities')
plt.subplot(2,seq_len,(seq_len//2+1,seq_len))
plt.imshow([src[0].data.numpy(), tgt[0].data.numpy(), batch.tgt[0].data.numpy()], origin = 'lower')
plt.title('src, tgt, y')

for i in range(seq_len):
    plt.subplot(2,seq_len, (seq_len + i + 1));
    plt.imshow(attn[i].data.numpy().T, origin='lower')
    plt.xlabel('q'); plt.ylabel('k');
    plt.title('attention head: %d' % i)

plt.tight_layout()


#%% 

from aid.model.utils import plot_attention

src = batch.src[:1];

plt.figure(5); plt.clf();
plot_attention(model, src, label = np.arange(src.shape[-1]))
plt.tight_layout()

#%% generate sequence - nothing to expect here, just test the routines

primer = batch.src[0][:10];

model.eval();
sequence_trained, probs_trained = model.generate(primer = primer, max_tgt_length = 32, method = 'max', verbose = True, ignore_code = CODE_PAD, max_code = CODE_SIZE, return_probabilities = True)

plt.figure(4); plt.clf();
plt.subplot(2,1,1)
plt.imshow([batch.src[0].data.numpy(), sequence_trained.data.numpy()], origin = 'lower')
plt.subplot(2,1,2)
p = probs_trained.data.numpy().T;
plt.imshow(p, origin='lower', aspect = 'auto', clim= (0, np.percentile(p, 99.5)))



#%% train epoch - debug

from aid.model.run import create_model, create_optimizer, create_loss, train_epoch

model = create_model(max_relative_position = True, value_relative_position = None)

optimizer = create_optimizer(model=model)

loss = create_loss()


#%%

ll = train_epoch(0, model, data, loss, optimizer, n_batches = 25, verbose = 1)

print(ll)

#%%

src = batch.src
src_mask = batch.src_mask();
tgt = batch.tgt
nrm = batch.n_tgt_codes();

y = model(src, src_mask)

#%% detect nans -> nans appear only if the sequence starts with a ignored symbol ! -> make sure this does not happen!

for b,yy in enumerate(y):
    if torch.any(torch.isnan(yy)):
        print(b); break;





#%% ##### Test real data ###########################################################

import aid.dataset.groove as groove

#%% directories

directory_data = '/home/ckirst/Media/Music/AImedia/MLMusic/Data/groove_encoded'
directory_base = '/home/ckirst/Media/Music/AImedia/MLMusic/Results/aid'

#%% data

data_train, data_validate, data_test = groove.train_validate_test_datasets(directory=directory_data)

print(len(data_train), len(data_validate), len(data_test))

#%% midi /code length


plt.figure(1); plt.clf()
dataset_names = { 0 : 'train', 1: 'validate', 2 : 'test'}
for i,dataset in enumerate(groove.train_validate_test_datasets(directory_data)):
    lengths = [len(d[0]) for d in dataset]

    plt.subplot(3,2,1 + 2 * i);
    plt.plot(lengths);
    plt.plot(sorted(lengths)[::-1])
    plt.title(dataset_names[i])
    plt.subplot(3,2,2 + 2 * i);
    plt.hist(lengths, bins=1024)
    plt.title(dataset_names[i])


#%%

data_train, data_validate, data_test = groove.train_validate_test_dataloaders(directory=directory_data, max_length = 2048, batch_size = 10)

#%%

for b, batch in enumerate(data_train):
    print(b, batch.src.shape)


#%%

from aid.dataset.midi_utils import plot
for b, batch in enumerate(data_train):
    print(b)
    plot(batch.src[0], ptype = 'bar')
    if b > 1:
        break;


#%%

from aid.dataset.groove import CODE_SIZE
from aid.model.run import create_model

model = create_model()

import itertools

batch = next(itertools.islice(iter(data_train), 15, None))

idx = [8];
src = batch.src[idx]
src_mask = batch.src_mask()[idx]

y = model(src, src_mask)

src_code = src[0].data.numpy()
code = model.decode(y, method='max', max_code = CODE_SIZE)[0].data.numpy()

layer = 0;
head  = 0;
attn = model.encoder.layers[layer].attention.attention[0,head].data.numpy()

#%%

from aid.dataset.midi_encoder import decode_midi
from aid.dataset.midi_utils import plot

plot(src_code)

events = decode_midi(src_code, return_events=True)
print(events)

#%%

import matplotlib.pyplot as plt
from aid.dataset.midi_utils import plot_midi_bars

plt.figure(10); plt.clf();
plot_midi_bars(src_code, attention=attn[-1])





#%% ##### Train real data ###########################################################

import aid.dataset.groove as groove

directory_data = '/home/ckirst/Media/Music/AImedia/MLMusic/Data/groove_encoded'
directory_base = '/home/ckirst/Media/Music/AImedia/MLMusic/Results/aid'

from aid.model.run import train


#%%
train(epochs=100,
      n_train_batches = 5,
      n_evaluate_train_batches  = 0,
      n_evaluate_test_batches   = 1,
      
      data_parameter = dict(batch_size=20),
      
      base_directory=directory_base,
      data_directory=directory_data,
      
      save = 1
     )


#%% continue traninig from last saved model


train(epochs=100,
      n_train_batches = 5,
      n_evaluate_train_batches  = 0,
      n_evaluate_test_batches   = 1,
      
      data_parameter = dict(batch_size=20),
      
      base_directory=directory_base,
      data_directory=directory_data,
      
      continue_epoch = -1
     )

















#%% model

import aid.model.music_transformer as mt
from aid.model.run import create_model, CODE_SIZE

model = mt.Transformer.create(n_src_vocab=CODE_SIZE)
model = create_model()

#%% evluation and maksing

data_train, data_validate, data_test = groove.train_validate_test_dataloaders(directory=directory_data, max_length = 2048, batch_size = 10)
batch = next(iter(data_train))

tgt = batch.tgt;
src = batch.src
src_mask = batch.src_mask();
tgt_mask = batch.tgt_mask();

plt.figure(2); plt.clf();
plt.subplot(3,2,1);
plt.imshow(src_mask.detach().numpy())
plt.subplot(3,2,2)
plt.imshow(tgt_mask.detach().numpy())
plt.subplot(3,2,3)
plt.imshow(src.detach().numpy())
plt.subplot(3,2,4)
plt.imshow(tgt.detach().numpy())



#%%
model.eval();

emb = model.embedding(src)
print(emb.shape)

encode_layer = model.encoder.layers[0]
encl = encode_layer(emb, src_mask)

enc = model.encoder(src, src_mask)



out = model(src, src_mask)


#%% learn simple copy forward task

n_max = model.CODE_SIZE;















