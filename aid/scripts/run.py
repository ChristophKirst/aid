#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Drummer Project

Project to create an AI drummer that performs with Pianist Jenny Q Chai
"""

#%% Train music transformer for AI drummer / test speed and quality

from aid.model.training import train

train(epochs=2000, n_eval_train_samples=20, n_eval_test_samples=20, n_train_batches = None, 
      directory_input   = '/global/home/users/ckirst/Media/Music/AImedia/MLMusic/Data/groove_encode',
      directory_results = '/global/home/users/ckirst/Media/Music/AImedia/MLMusic/Code/aid/results')

train(epochs=10, n_eval_train_samples=2, n_eval_test_samples=2, n_train_batches = 10)


#%%

continue_epoch = True;

train(epochs=100, n_eval_train_samples=2, n_eval_test_samples=2, n_train_batches = 100,
      continue_epoch = continue_epoch, continue_weights = -1);

#%%

from aid.model.generation import generate

midi, code = generate(primer=50, n_primer = 200,
                      plot_primer = True, target_sequence_length=300, beam_search=None, file_model=-1, return_code = True)


#%% 

import aid.utils.utils as utils


utils.plot_midi(midi)

utils.play_midi(midi);


#%%

import utils.midi_encoder as encoder

[encoder.CODE_TO_EVENT[c] for c in code]


    
#%%

import aid.dataset.groove as groove
dt, dv, dtst = groove.initialize_datasets(directory_base   = '/home/ckirst/Media/Music/AImedia/MLMusic/Data/groove_encoded', max_seq=2048)

x,t = dt[0]

import aid.utils.utils as utils

utils.play_code(x);


#%% get idea of length of data sets

import numpy as np
ll = np.array([np.where(dt[i][0].detach().numpy() == encoder.ENCODE_TOKEN_PAD)[0][0] if encoder.ENCODE_TOKEN_PAD in dt[i][0] else len(dt[i][0]) for i in range(0,500, 1)])

import matplotlib.pyplot as plt
plt.figure(1); plt.clf()
#plt.plot(ll)
plt.hist(ll, bins=124)


small = np.where(ll <= 100 )[0]

for s in small[:10]:
   utils.plot(dt[s][0])


utils.play(dt[small[9]][0])





