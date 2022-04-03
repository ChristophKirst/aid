#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Drummer Project

Project to create an AI drummer that performs with Pianist Jenny Q Chai

Translate midi data to data for tranformer network
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__copyright__ = 'Copyright © 2022 by Christoph Kirst'


#%%

import os
import aid.dataset.groove as groove
import aid.dataset.midi_encoder as encoder

prefix = '/global'
prefix = '';

#%% download data set

directory_midi = groove.download(directory= prefix + '/home/ckirst/Media/Music/AImedia/MLMusic/Data');


#%%
directory_midi   =  prefix + '/home/ckirst/Media/Music/AImedia/MLMusic/Data/groove'
directory_encode =  prefix + '/home/ckirst/Media/Music/AImedia/MLMusic/Data/groove_encoded'

#%% Test encoding

info = groove.info(directory_midi)

example = info['midi_filename'][0];

import aid.utils.utils as au

au.plot_midi(example)

au.play_midi(example)

code = encoder.encode_midi(example)
print(code[:20])

#%% Encode

groove.encode_dataset(directory_midi, directory_encode)


#%% 

directory_train, directory_validate, directory_test = groove.train_validate_test_directories(directory_encode)

dataset = groove.GrooveDataset(directory_train, random_sequence = None, max_length = None)

#%% lengths of the midi data -> uniform batch sizes are better for efficient traiing (varialbe sequence length)
import matplotlib.pyplot as plt
plt.figure(1); plt.clf()
dataset_names = { 0 : 'train', 1: 'validate', 2 : 'test'}
for i,dataset in enumerate(groove.train_validate_test_datasets(directory_encode)):
    
    lengths = [len(d[0]) for d in dataset]
    
    plt.subplot(3,2,1 + 2 * i);
    plt.plot(lengths);
    plt.plot(sorted(lengths)[::-1])
    plt.title(dataset_names[i])
    plt.subplot(3,2,2 + 2 * i);
    plt.hist(lengths, bins=1024)
    plt.title(dataset_names[i])



#%% listen to a long data set

dataset = groove.GrooveDataset(directory_train, random_sequence = None, max_length = None)
lengths = [len(d[0]) for d in dataset]

import numpy as np
sorted_idx = np.argsort(lengths)[::-1]

import aid.utils.utils as utils

i = sorted_idx[10];
code = dataset[i][0];
print(lengths[i])

utils.play(code)
