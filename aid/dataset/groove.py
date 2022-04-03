#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Groove data set
===============

Interface to the Groove MIDI Dataset
https://magenta.tensorflow.org/datasets/groove
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__copyright__ = 'Copyright © 2022 by Christoph Kirst'

import os
import natsort
import pickle
import random
import torch
from torch.utils.data import Dataset, DataLoader

import aid.dataset.midi_encoder as encoder

CODE_PAD  = encoder.CODE_PAD
CODE_END  = encoder.CODE_END
CODE_SIZE = encoder.CODE_SIZE

from aid.model.transformer import Batch


class GrooveDataset(Dataset):
    """
    Groove Encoded MIDI Dataset
    https://magenta.tensorflow.org/datasets/groove
    """

    def __init__(self, directory, max_length = None, random_sequence = True, dtype = None, representation = 'code'):
        super(GrooveDataset, self).__init__()
        
        self.directory       = directory
        self.max_length      = max_length
        self.random_sequence = random_sequence
        self.representation  = representation

        data_files = [os.path.join(directory, f) for f in os.listdir(directory)];     
        data_files = natsort.natsort.natsorted(data_files);
        self.data_files = data_files;
    
    
    def __len__(self):
        """number of data files"""
        return len(self.data_files)


    def __getitem__(self, idx):
        """Returns input and target sequence"""
        
        i_stream = open(self.data_files[idx], "rb")
        code = torch.tensor(pickle.load(i_stream), dtype=torch.long, device=torch.device("cpu"))
        i_stream.close()

        src, tgt = generate_source_target(code, self.max_length, self.random_sequence)

        if self.representation == 'midi':
            src, tgt = encoder.decode_midi(src), encoder.decode_midi(tgt);
           
        return src, tgt
    
    def __repr__(self):
        return "GrooveDataset[%d]" % (self.__len__());


def generate_source_target(code, max_length = None, random_sequence = None, start_sequence = 0, ignore_code = CODE_PAD):
    """midi code to source and target"""

    code = code[start_sequence:]; 
    code_length     = len(code)
    
    if max_length is None:
        max_length = code_length;
    max_length = min(max_length, code_length)
        
    src = torch.full((max_length, ), ignore_code, dtype=torch.long, device=torch.device("cpu"))
    tgt = torch.full((max_length, ), ignore_code, dtype=torch.long, device=torch.device("cpu"))
    
    total_length    = max_length + 1 # performing seq2seq

    if (code_length == 0):
        return src, tgt

    if (code_length < total_length):
        src[:]   = code
        tgt[:code_length-1] = code[1:]
        tgt[code_length-1]  = ignore_code
    else:
        if random_sequence is not None:
            start = random.randint(0, code_length - total_length)
        else:
            start = 0

        end = start + total_length

        data = code[start:end]

        src = data[:max_length]
        tgt = data[1:total_length]

    return src, tgt


def collate_batch(batch, ignore_code = CODE_PAD):
    """Collate a set of data to a batch tensor.
    
    Note
    ----
    To use as collate_fn in DataLoader.
    """
    # batch is [(src1, tgt1), (src2, tgt2), ...]
    batch_size = len(batch);
    max_length = max([len(s) for s,t in batch])
    
    src = torch.full((batch_size, max_length), ignore_code, dtype=torch.long, device=torch.device("cpu"))
    tgt = torch.full((batch_size, max_length), ignore_code, dtype=torch.long, device=torch.device("cpu"))

    for i,b in enumerate(batch):
        n = len(b[0]);
        src[i,:n], tgt[i,:n] = b;

    return Batch(src, tgt, ignore_code=ignore_code);


def train_validate_test_directories(directory):
    directory_train     = os.path.join(directory, "train")
    directory_validate  = os.path.join(directory, "validate")
    directory_test      = os.path.join(directory, "test")
    return directory_train, directory_validate, directory_test


def train_validate_test_datasets(directory, max_length = None, random_sequence = None):
    """Create datasets for training"""

    directory_train, directory_validate, directory_test = train_validate_test_directories(directory)

    dataset_train    = GrooveDataset(directory_train,     max_length, random_sequence)
    dataset_validate = GrooveDataset(directory_validate,  max_length, random_sequence)
    dataset_test     = GrooveDataset(directory_test,      max_length, random_sequence)

    return dataset_train, dataset_validate, dataset_test


def train_test_datasets(directory, max_length = None, random_sequence = None):
    """Create datasets for training"""
    dataset_train, _, dataset_test =  train_validate_test_datasets(directory=directory, max_length=max_length, random_sequence=random_sequence)
    return dataset_train, dataset_test


def train_validate_test_dataloaders(directory, max_length = None, random_sequence = None, batch_size = 10, shuffle = False, **kwargs):
    """Create data loders for training"""
    dataset_train, dataset_validate, dataset_test =  train_validate_test_datasets(directory=directory, max_length=max_length, random_sequence=random_sequence);
     
    loader_train   = DataLoader(dataset_train,     batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch, **kwargs)
    loder_validate = DataLoader(dataset_train,     batch_size=batch_size, shuffle=False,   collate_fn=collate_batch, **kwargs)
    loader_test    = DataLoader(dataset_validate,  batch_size=batch_size, shuffle=False,   collate_fn=collate_batch, **kwargs)

    return loader_train, loder_validate, loader_test


def train_test_dataloaders(directory, max_length = None, random_sequence = None, batch_size = 10, shuffle = False, **kwargs):
    """Create data loaders for training"""
    loader_train, _, loader_test =  train_validate_test_dataloaders(directory=directory, max_length=max_length, random_sequence=random_sequence, batch_size=batch_size, shuffle=shuffle, **kwargs)
    return loader_train, loader_test



# Groove data setup

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
    
    data_directory = os.path.join(directory, 'groove')
    
    return data_directory


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
        
        encoded = encoder.encode_midi(file_midi)
        
        split_type  = info["split"]
        file_name   = file_midi.split("/")[-1] + ".pickle"
        
        # prepend filename with length for efficient batching
        file_name = '%06d_' % len(encoded) + file_name;

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

        o_stream = open(file_out, "wb")
        pickle.dump(encoded, o_stream)
        o_stream.close()

        total_count += 1
        if verbose:
            print('processed (%d/%d): %s' % (total_count, len(file_info), file_name));

    if verbose:
        print("training  :", train_count)
        print("validation:", val_count)
        print("test      :", test_count)
    return True


def info(directory_midi):
    """Groove data set info."""
    import pandas as pd
    info = os.path.join(directory_midi, 'info.csv')
    info = pd.read_csv(info)
    info['midi_filename'] = [os.path.join(directory_midi, f) for f in info['midi_filename']]
    return info


def file_info(filename):
    """Groove dataset info from midi filename."""
    path, name = os.path.split(filename)
    path = path.split(os.path.sep);
    drummer = path[-2];
    session = path[-1];
    index, style, beat, time_signature,  = name.split('_');
    time_signature = time_signature.split('.')[0];
    return drummer, session, index, style, beat, time_signature;




# class MyIterator(data.Iterator):
#     def create_batches(self):
#         if self.train:
#             def pool(d, random_shuffler):
#                 for p in data.batch(d, self.batch_size * 100):
#                     p_batch = data.batch(
#                         sorted(p, key=self.sort_key),
#                         self.batch_size, self.batch_size_fn)
#                     for b in random_shuffler(list(p_batch)):
#                         yield b
#             self.batches = pool(self.data(), self.random_shuffler)
            
#         else:
#             self.batches = []
#             for b in data.batch(self.data(), self.batch_size,
#                                           self.batch_size_fn):
#                 self.batches.append(sorted(b, key=self.sort_key))

# def rebatch(pad_idx, batch):
#     "Fix order in torchtext to match ours"
#     src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
#     return Batch(src, trg, pad_idx)

