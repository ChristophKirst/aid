#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Drummer Project

Project to create an AI drummer that performs with Pianist Jenny Q Chai

Encode midi drum data to data for tranformer network
"""


import os
import natsort
import pickle
import random
import torch
from torch.utils.data import Dataset

import aid.utils.midi_encoder as encoder

SEQUENCE_START = 0

# EPianoDataset
class GrooveDataset(Dataset):
    """
    Groove MIDI Dataset
    https://magenta.tensorflow.org/datasets/groove
    """

    def __init__(self, directory, max_seq=2048, random_seq=True):
        self.directory  = directory
        self.max_seq    = max_seq
        self.random_seq = random_seq

        self.data_files = [os.path.join(directory, f) for f in os.listdir(directory)];
        #if exclude_eval_sessions:
        #    self.data_files = [f for f in self.data_files if 'eval' not in f];
        
        self.data_files = natsort.natsort.natsorted(self.data_files);
    
    
    def __len__(self):
        """number of data files"""
        return len(self.data_files)


    def __getitem__(self, idx):
        """returns input and target sequence"""
        
        i_stream    = open(self.data_files[idx], "rb")
        midi_code   = torch.tensor(pickle.load(i_stream), dtype=torch.long, device=torch.device("cpu"))
        i_stream.close()

        x, tgt = generate_input_target(midi_code, self.max_seq, self.random_seq)

        return x, tgt


# download data set






def download(directory = '/home/ckirst/Media/Music/AImedia/MLMusic/Data'):
    import progressbar
    import urllib
    
    class ProgressBar():
        def __init__(self):
            self.pbar = None
    
        def __call__(self, block_num, block_size, total_size):
            if not self.pbar:
                self.pbar=progressbar.ProgressBar(maxval=total_size)
                self.pbar.start()
    
            downloaded = block_num * block_size
            if downloaded < total_size:
                self.pbar.update(downloaded)
            else:
                self.pbar.finish()
    
    os.makedirs(directory,    exist_ok=True)
    
    target = os.path.join(directory, 'groove-v1.0.0-midionly.zip');    
    urllib.request.urlretrieve('https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip', target, ProgressBar())

    import shutil
    shutil.unpack_archive(target, directory);
    
    data_directory = os.paht.join(directory, 'groove')
    
    return data_directory


# process_midi
def generate_input_target(midi_code, max_seq, random_seq):
    """raw midi to input and target"""

    x   = torch.full((max_seq, ), encoder.ENCODE_TOKEN_PAD, dtype=torch.long, device=torch.device("cpu"))
    tgt = torch.full((max_seq, ), encoder.ENCODE_TOKEN_PAD, dtype=torch.long, device=torch.device("cpu"))

    raw_len     = len(midi_code)
    full_seq    = max_seq + 1 # performing seq2seq

    if(raw_len == 0):
        return x, tgt

    if(raw_len < full_seq):
        x[:raw_len]         = midi_code
        tgt[:raw_len-1]     = midi_code[1:]
        tgt[raw_len]        = encoder.ENCODE_TOKEN_END
    else:
        # randomly selecting a range
        if(random_seq):
            end_range = raw_len - full_seq
            start = random.randint(SEQUENCE_START, end_range)

        # take from the start
        else:
            start = SEQUENCE_START

        end = start + full_seq

        data = midi_code[start:end]

        x = data[:max_seq]
        tgt = data[1:full_seq]

    # print("x:",x)
    # print("tgt:",tgt)

    return x, tgt


def train_validate_test_directories(directory_base):
    directory_train     = os.path.join(directory_base, "train")
    directory_validate  = os.path.join(directory_base, "validate")
    directory_test      = os.path.join(directory_base, "test")
    return directory_train, directory_validate, directory_test


# create_datasets
def initialize_datasets(directory_base, max_seq, random_seq=True):
    """create datasets objects for training"""

    directory_train, directory_validate, directory_test = train_validate_test_directories(directory_base)

    dataset_train    = GrooveDataset(directory_train,     max_seq, random_seq)
    dataset_validate = GrooveDataset(directory_validate,  max_seq, random_seq)
    dataset_test     = GrooveDataset(directory_test,      max_seq, random_seq)

    return dataset_train, dataset_validate, dataset_test


def encode_dataset(directory_midi, directory_encode, verbose = True):
    """Transform midi data set into encoded data for transformer network"""
    directory_train, directory_validate, directory_test = train_validate_test_directories(directory_encode)
    os.makedirs(directory_train,    exist_ok=True)
    os.makedirs(directory_validate, exist_ok=True)
    os.makedirs(directory_test,     exist_ok=True)

    total_count = 0
    train_count = 0
    val_count   = 0
    test_count  = 0
    
    import pandas as pd
    file_info = os.path.join(directory_midi, 'info.csv')
    file_info = pd.read_csv(file_info)
    print('Encoding %s midi files.' % len(file_info))

    for i, info in file_info.iterrows():
        file_midi   = os.path.join(directory_midi, info["midi_filename"])
        split_type  = info["split"]
        file_name   = file_midi.split("/")[-1] + ".pickle"

        if(split_type == "train"):
            file_out = os.path.join(directory_train, file_name)
            train_count += 1
        elif(split_type == "validation"):
            file_out = os.path.join(directory_validate, file_name)
            val_count += 1
        elif(split_type == "test"):
            file_out = os.path.join(directory_test, file_name)
            test_count += 1
        else:
            print("ERROR: Unrecognized split type:", split_type)
            return False

        encoded = encoder.encode_midi(file_midi)

        o_stream = open(file_out, "wb")
        pickle.dump(encoded, o_stream)
        o_stream.close()

        total_count += 1
        if verbose:
            print('processed (%d/%d): %s' % (total_count, len(file_info), file_name));

    print("training  :", train_count)
    print("validation:", val_count)
    print("test      :", test_count)
    return True


def info(directory_midi):
    import pandas as pd
    info = os.path.join(directory_midi, 'info.csv')
    info = pd.read_csv(info)
    info['midi_filename'] = [os.path.join(directory_midi, f) for f in info['midi_filename']]
    return info


def file_info(filename):
    """dataset info from filename"""
    path, name = os.path.split(filename)
    path = path.split(os.path.sep);
    drummer = path[-2];
    session = path[-1];
    index, style, beat, time_signature,  = name.split('_');
    time_signature = time_signature.split('.')[0];
    return drummer, session, index, style, beat, time_signature;




