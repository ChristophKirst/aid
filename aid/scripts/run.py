#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Drummer Project

Project to create an AI drummer that performs with Pianist Jenny Q Chai
"""

#%% Train music transformer for AI drummer / test speed and quality

from aid.model.training import train

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
