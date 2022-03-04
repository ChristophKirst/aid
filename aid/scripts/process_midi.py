#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Drummer Project

Project to create an AI drummer that performs with Pianist Jenny Q Chai

Translate midi data to data for tranformer network
"""

import os
import aid.dataset.groove as groove
import aid.utils.midi_encoder as encoder

#%%
directory_midi   =  '/home/ckirst/Media/Music/AImedia/MLMusic/Data/groove'
directory_encode =  '/home/ckirst/Media/Music/AImedia/MLMusic/Data/groove_encoded'

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

dataset = groove.GrooveDataset(directory_train)
