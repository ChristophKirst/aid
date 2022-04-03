#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Drummer Project

Project to create an AI drummer that performs with Pianist Jenny Q Chai
"""


#improeve tihs: loss functoin: not just coress entropy : if the event is a time shift wth differnt time intervall should be better than other ptich n=ro note event etc...
# hierarhical classes  -> maybe have a transformer that results in class + value events -> 

#todos: 
    # 1) attention visualization (done))
    # 2) accompaniemant transformer
    # 3) choose best transfoerm model XL , reformer, time compressoin 
    # 4) better time embedding via VSAs ?
    # 5) multi-scale transformer / transfoemr attention to understand compelx dynamical systems ?
    # 6) generalized granger causality: tranforer causality: how much does predictability improve with additional brain area dynamics


#%% ##### Train real data ###########################################################

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



