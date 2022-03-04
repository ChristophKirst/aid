#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Drummer Project

Project to create an AI drummer that performs with Pianist Jenny Q Chai
"""


# Train music transformer for AI drummer / test speed and quality


import os
import numpy as np

import torch

from aid.utils.midi_encoder import encode_midi, decode_midi
from aid.model.music_transformer import MusicTransformer
from aid.dataset.groove import initialize_datasets


if torch.cuda.device_count() > 0:
  device = torch.device("cuda")
else:
  #print('Warning: No gpu device found!')
  device = torch.device("cpu");


def generate(directory_output = './generate',
             primer = None, 
             
             beam_search = None,
             target_sequence_length = 1024,
             
             file_model = None,
             n_primer = 10,
             directory_data = '/home/ckirst/Media/Music/AImedia/MLMusic/Data/groove_encoded',   
             max_sequence = 2048,   # maximal midi sequence
             
             n_layers      = 6,     # layers in transformer
             n_heads       = 8,     # multi-head attention
             d_model       = 512,   # dimension of model
             d_feedforward = 1024,  # feed forward layer dimension
             dropout       = 0.1,   # dropout rate
             rpr           = True,  # relative position encoding
             
             result_midi_file = 'generated.mid',
             primer_midi_file = 'primer.mid'
             ):
  
    os.makedirs(directory_output, exist_ok=True); 
     
    if primer is None or isinstance(primer, int):
        _, _, dataset = initialize_datasets(directory_data, n_primer, random_seq=False)
        if primer is None:
            primer = np.rand.randrange(0,len(dataset));
        primer, _ = dataset[primer]
    else:
        primer = encode_midi(primer)
    primer = primer[:n_primer];     

    if primer_midi_file is not None:
        midi = decode_midi(primer);
        midi.write(os.path.join(directory_output, primer_midi_file))
    
    model = MusicTransformer(n_layers=n_layers, n_heads=n_heads,
                             d_model=d_model, d_feedforward=d_feedforward,
                             max_sequence=max_sequence, rpr=rpr).to(device)
    if file_model:
        model.load_state_dict(torch.load(file_model))
        
    model.eval()
    
    if beam_search is None: 
        beam_search = 0;
    with torch.set_grad_enabled(False):
        if beam_search > 0:
            print("Generate via beam search:", beam_search)
            midi_file = os.path.join(directory_output, "beam.mid")
        else:
            print("Generate via random distirbution")
            midi_file = os.path.join(directory_output, "beam.mid")
    
    sequence = model.generate(primer, target_sequence_length, beam=beam_search);
    
    #return model, primer, sequence;
    return decode_midi(sequence[0].cpu().detach().numpy(), filename=midi_file)    


