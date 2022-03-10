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

from aid.model.music_transformer import MusicTransformer
from aid.dataset.groove import initialize_datasets
from aid.utils.midi_encoder import encode_midi, decode_midi, ENCODE_TOKEN_PAD
from aid.utils.utils import plot_midi

if torch.cuda.device_count() > 0:
  device = torch.device("cuda")
else:
  #print('Warning: No gpu device found!')
  device = torch.device("cpu");


def generate(directory_output = './generate',
             primer = None, 
             
             beam_search = None,
             target_sequence_length = 1024,
             
             n_primer = 10,
             directory_data = '/home/ckirst/Media/Music/AImedia/MLMusic/Data/groove_encoded',   
             max_sequence = 2048,   # maximal midi sequence
             
             file_model = None,
             directory_models = '/home/ckirst/Media/Music/AImedia/MLMusic/Develop/AIDrummer/results/models',
             n_layers      = 6,     # layers in transformer
             n_heads       = 8,     # multi-head attention
             d_model       = 512,   # dimension of model
             d_feedforward = 1024,  # feed forward layer dimension
             dropout       = 0.1,   # dropout rate
             rpr           = True,  # relative position encoding
             
             result_midi_file = 'generated.mid',
             primer_midi_file = 'primer.mid',
             
             plot_primer = False,
             return_code = False
             ):
  
    os.makedirs(directory_output, exist_ok=True); 
     
    if primer is None or isinstance(primer, int):
        _, _, dataset = initialize_datasets(directory_data, n_primer, random_seq=False)
        if primer is None:
            primer = np.rand.randrange(0,len(dataset));
        primer, _ = dataset[primer]
        max_length = np.where(primer == ENCODE_TOKEN_PAD)[0];
        if len(max_length) > 0:
            primer = primer[:max_length[0]];
    else:
        primer = encode_midi(primer)
    primer = primer[:n_primer];     

    if primer_midi_file is not None:
        midi = decode_midi(primer);
        midi.write(os.path.join(directory_output, primer_midi_file))
    if plot_primer:
        midi = decode_midi(primer);
        plot_midi(midi);
    
    model = MusicTransformer(n_layers=n_layers, n_heads=n_heads,
                             d_model=d_model, d_feedforward=d_feedforward,
                             max_sequence=max_sequence, rpr=rpr).to(device)
    if file_model == -1:
        import glob
        files = sorted(glob.glob(os.path.join(directory_models, "epoch_*.pickle")));
        file_model = files[-1];
    if file_model is not None:
        print('using model: %s' % file_model)
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
    code = sequence[0].cpu().detach().numpy();
    midi = decode_midi(code, filename=midi_file);
    
    result = (midi,)
    if return_code:
        result += (code,);
    if len(result) == 1:
        result = result[0];
    return result;
        


