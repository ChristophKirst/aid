#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Drummer Project

Project to create an AI drummer that performs with Pianist Jenny Q Chai
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__copyright__ = 'Copyright © 2022 by Christoph Kirst'


"""
### Remote training

### Screen
screen -S aid
# ctrl-a ctrl-d to detach
screen -r aid 

### ssh
ssh ckirst@exalearn.lbl.gov
ssh ckirst@exalearn2.lbl.gov
sshfs ckirst@exalearn2.lbl.gov:/global/home/users/ckirst/Media/Music/AImedia/MLMusic/ /home/ckirst/exalearn/


### tensorboard
tensorboard --logdir /home/ckirst/exalearn/Results/aid/tensorboard/


### Notes
#improe: loss functoin: not just coress entropy : if the event is a time shift wth differnt time intervall should be better than other ptich n=ro note event etc...
# hierarhical classes  -> maybe have a transformer that results in class + value events -> 

#todos: 
    # 1) attention visualization (done))
    # 2) accompaniemant transformer
    # 3) choose best transfoerm model XL , reformer, time compressoin 
    # 4) better time embedding via VSAs ?
    # 5) multi-scale transformer / transfoemr attention to understand compelx dynamical systems ?
    # 6) generalized granger causality: tranforer causality: how much does predictability improve with additional brain area dynamics
    
    
# multi gpu training

    1.) replace DataParallel with DistributedDataParallel
    https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
    https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51  
"""

#%% ##### Train real data ###########################################################

from aid.model.run import train
directory_data = '/home/ckirst/Media/Music/AImedia/MLMusic/Data/groove_encoded'
directory_base = '/home/ckirst/Media/Music/AImedia/MLMusic/Results/aid'

#%%
from aid.model.run import train;

directory_data = '/global/home/users/ckirst/Media/Music/AImedia/MLMusic/Data/groove_encoded';
directory_base = '/global/home/users/ckirst/Media/Music/AImedia/MLMusic/Results/aid';

model, optimizer, loss = train(
      epochs=2,
      n_train_batches = 2,
      n_evaluate_train_batches  = 0,
      n_evaluate_test_batches   = 1,
      
      model_parameter = dict(multi_gpu=True),
      
      data_parameter = dict(batch_size=4, max_sequence_length=1024),
      
      optimizer_parameter = dict(factor = 0.75, warmup = 4000),
      
      base_directory=directory_base,
      data_directory=directory_data,
      
      save = 1,
      
      clean_directories = True
     )


#%% full run
model, optimizer, loss = train(
      epochs=10000,
      n_train_batches = None,
      n_evaluate_train_batches  = 0,
      n_evaluate_test_batches   = None,
      
      data_parameter = dict(batch_size=5, max_sequence_length=512),
      
      optimizer_parameter = dict(factor = 0.75, warmup = 4000),
      
      base_directory=directory_base,
      data_directory=directory_data,
      
      save = 20,
      
      clean_directories = False
     )

#%%

model, optimizer, loss = train(
      epochs=2,
      n_train_batches = 2,
      n_evaluate_train_batches  = 0,
      n_evaluate_test_batches   = 1,
      
      model_parameter = dict(multi_gpu=True),
      
      data_parameter = dict(batch_size=8, max_sequence_length=32),
      
      optimizer_parameter = dict(factor = 0.75, warmup = 4000),
      
      base_directory=directory_base,
      data_directory=directory_data,
      
      save = 1,
      
      clean_directories = True
     )




#%% simple tests

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




#%% generate beats

from aid.model.run import generate

data_directory = '/home/ckirst/Media/Music/AImedia/MLMusic/Data/groove_encoded'
base_directory = '/home/ckirst/Media/Music/AImedia/MLMusic/Results/aid'


midi, tokens, primer, full_primer, probs, model = \
       generate(continue_model = 'best_acc',
                primer = 501, 
                max_primer_tokens = 50,  
                sequence_length = 1024,
                model_sequence_length = 200,
                
                model_parameter = dict(save_attention = False, mask_future = True),
                
                return_midi = True,
                return_tokens = True,
                return_primer = True,
                return_full_primer = True,
                return_probabilities = True,
                return_model = True,
                
                base_directory = base_directory,
                data_directory = data_directory,
                 
                save_midi = True,
                save_primer = True,
                
                plot_primer = False,
                plot = False,
                
                verbose = True
                );


#%%

from aid.dataset.midi_utils import plot, play
play(midi)


#%%

play(full_primer)
#%%
plot(primer)
plot(midi)
plot(full_primer)


#%%

play(midi)

#%%

play(primer)

#%% plot the probabilities 

#%%

from aid.model.run import load_model
base_directory = '/home/ckirst/Media/Music/AImedia/MLMusic/Results/aid'
model = load_model(source='best_acc', base_directory=base_directory)

#%%

import gc
gc.get_objects()


#%%
import numpy as np
import torch
import gc

def check_memory(verbose = False):
    n_objects = 0;
    total_size = 0;
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if verbose:
                    if hasattr(obj, 'name') and obj.name is not None:
                        print(obj.name)
                    print(type(obj), obj.size())
                n_objects += 1;
                total_size += np.prod(obj.size())
        except:
            pass
    
    print('number of objects: %d of total size %d' % (n_objects, total_size))


check_memory()
#%%

from aid.model.run import create_data

from aid.model.run import create_model


from aid.model.run import load_model, create_data
data_directory = '/home/ckirst/Media/Music/AImedia/MLMusic/Data/groove_encoded'
base_directory = '/home/ckirst/Media/Music/AImedia/MLMusic/Results/aid'

model = load_model(base_directory=base_directory, source='best_acc')

data, _ = create_data(directory=data_directory)

primer = data[500].src[:,:20];


#%%

y = model(primer)



#%% convert state dicts to newer versions

import torch

from aid.model.run import file_name_model
fn = file_name_model(base_directory = '/home/ckirst/Media/Music/AImedia/MLMusic/Results/aid', source='best_acc')
state_dict = torch.load(fn, map_location=torch.device('cpu'))

from collections import OrderedDict as odict
def filter_state_dict(state_dict, remove = None, keep = None):
    if remove is None: 
        remove = [];
    if isinstance(remove,str):
        remove = [remove];  
    
    if isinstance(keep,str):
        keep = [keep];  
        
    new_state_dict = odict();
        
    
    for k,v in state_dict.items():
        skip = False;
        for r in remove:
            if r in k:
                skip = True; 
                break;
        if skip:
            continue;
        
        if keep is None:
            new_state_dict[k] = v;
        else:
            for l in keep:
                if l in k:
                    new_state_dict[k] = v;
                    break;

    return new_state_dict;
    

new_state_dict = filter_state_dict(state_dict, remove='mask_relative_position')

#%%

from aid.model.run import create_model
model = create_model();
model.load_state_dict(new_state_dict)

#%% 

from aid.model.run import save_model
save_model(model, base_directory = '/home/ckirst/Media/Music/AImedia/MLMusic/Results/aid', sink='best_acc' )
