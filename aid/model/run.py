#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Music Transformer
=================

Function to train, evaluate and generate music from the model.

Project to create an AI drummer that performs with Pianist Jenny Q Chai
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__copyright__ = 'Copyright © 2022 by Christoph Kirst'


import time
import os
import csv
import pickle

import torch
import torch.nn

from aid.model.transformer import Transformer, Optimizer, Loss
from aid.dataset.groove import train_test_dataloaders, N_TOKENS, TOKEN_PAD, TOKEN_END
import aid.dataset.midi_encoder as encoder

import aid.dataset.midi_utils as utils;


### GPU computing

class DataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
        
def get_device():
    if torch.cuda.device_count() > 0:
        device = torch.device("cuda")
    else:
        #print('Warning: No gpu device found!')
        device = torch.device("cpu");
    return device

def set_device(obj, device = None, multi_gpu = False, device_ids = None, output_device = None):
    if device_ids is not None:
        multi_gpu = True;
    if device is None:
        device = get_device()
    if multi_gpu and torch.cuda.device_count() > 1:
        if isinstance(device_ids, int):
            device_ids = list(range(device_ids));
            if output_device is None:
                output_device = device_ids[0]   
        obj = DataParallel(obj, device_ids=device_ids, output_device=output_device);
    obj = obj.to(device);
    return obj;


### Object creation

def create_model(device = None, multi_gpu = False, device_ids = None, output_device = None, **kwargs):
    for k,v in zip(['n_tokens'], [N_TOKENS]):
        if k not in kwargs.keys():
            kwargs[k] = v;
    model = Transformer.create(**kwargs);
    model = set_device(model, device=device, multi_gpu=multi_gpu, device_ids=device_ids, output_device=output_device);
    return model;
    
def create_optimizer(model, **kwargs):
    optimizer = Optimizer.create(model=model, **kwargs);
    return optimizer

def create_loss(device = None, **kwargs):
    for k,v in zip(['n_tokens', 'ignore_token'], [N_TOKENS, TOKEN_PAD]):
        if k not in kwargs.keys():
            kwargs[k] = v;
    loss = Loss(**kwargs);
    loss = set_device(loss, device=device);
    return loss;

def create_data(directory, device = None, **kwargs):
    return train_test_dataloaders(directory=directory, **kwargs);


### Training and Evaluation 

def train_epoch(epoch, model, data, loss, optimizer, n_batches = None, verbose = True, separator = '=========='):
    """Trains the Music transformer model for one epoch."""
    
    model.train();
    device = model.device();
    
    if n_batches is None:
        n_batches = len(data);
    if verbose is True:
        verbose = n_batches;
    if verbose:
        time_start = time.time();
        
    loss_total = 0.0;
    tokens_total = 0;
    
    for b, batch in enumerate(data):
        if b >= n_batches:
            break;
        #print(b, batch)
        
        if verbose:
            time_batch_start = time.time()
  
        if optimizer:
          optimizer.zero_grad()
  
        src = batch.src.to(device)
        src_mask = batch.src_mask(src);
        tgt = batch.tgt.to(device);
        n_tokens = batch.n_tgt_tokens();
  
        fwd = model(src, src_mask)
         
        if torch.any(torch.isnan(fwd)):
            print('Nans encountered in model')
            return batch;
  
        out = loss(fwd.contiguous().view(-1, fwd.size(-1)), 
                   tgt.contiguous().view(-1)) / n_tokens
        
        if torch.any(torch.isnan(out)):
            print('Nans encountered in loss')
            return batch;
  
        out.backward()
        
        if optimizer:
            optimizer.step()
        
        loss_total += float(out);
        tokens_total += float(n_tokens);
        
        if verbose and ((b+1) % verbose == 0):
            time_batch_end = time.time()
            time_batch_total = time_batch_end - time_batch_start;
              
            if separator: print(separator)
            print('training: epoch %d  batch %d/%d (%d)' % (epoch+1, b+1, n_batches, len(data)))
            print('batch_size, sequence_length = %d, %d'%  (src.shape[0], src.shape[1]))
            print("lrate:    %r" % optimizer.rate())
            print("loss:     %r" % (float(out) / float(n_tokens)))
            print("time (s): %r" % time_batch_total)
            if separator: print(separator)

    if optimizer:
        optimizer.step_epoch();

    loss_mean = loss_total / tokens_total;

    if verbose:
        time_end = time.time();
        time_total = time_end - time_start;
        
        if separator: print(separator + separator)
        print("training: epoch %d done!" % (epoch+1))
        print("lrate:     %r" % optimizer.rate())
        print("mean loss: %r" % loss_mean)
        print("time (s):  %r" % time_total)
        if separator: print(separator + separator)

    return loss_mean;
  

def evaluate_model(model, data, loss, n_batches = None, verbose = False, separator = '=========='):
    """evaluates the music transformer model."""
    
    model.eval()
    device = model.device();
       
    if n_batches is None:
        n_batches = len(data);
    if verbose is True:
        verbose = n_batches;
    if verbose:
        time_start = time.time();
   
    if n_batches == 0:
        return float('inf'), 0;
   
    loss_total   = 0.0
    acc_total    = 0.0
    tokens_total = 0;
    
    with torch.set_grad_enabled(False):
        for b, batch in enumerate(data):
            if b >= n_batches:
                break;
                
            if verbose:
                time_batch_start = time.time()
            
            src = batch.src.to(device);
            src_mask = batch.src_mask(src);
            tgt = batch.tgt.to(device)
            n_tokens = float(batch.n_tgt_tokens());

            fwd = model(src, src_mask)

            acc = float(compute_accuracy(fwd, tgt, src_mask));
            acc_total += acc * n_tokens;
            
            out = loss.forward(fwd.contiguous().view(-1, fwd.size(-1)), 
                               tgt.contiguous().view(-1)) / n_tokens
  
            out = float(out);
            loss_total += out;
            tokens_total += n_tokens

            if verbose and ((b+1) % verbose == 0):
                time_batch_end = time.time()
                time_batch_total = time_batch_end - time_batch_start;
                  
                if separator: print(separator)
                print('evaluation:  batch %d/%d' % (b+1, len(data)))
                print('batch_size, sequence_length = %d, %d'%  (src.shape[0], src.shape[1]))
                print("loss:     %r" % (out / n_tokens))
                print("accuracy: %r" % acc)
                print("time (s): %r" % time_batch_total)
                if separator: print(separator)  
        
        loss_mean = loss_total / tokens_total
        acc_mean  = acc_total / tokens_total

    if verbose:
        time_end = time.time()
        time_total = time_end - time_start;
        
        if separator: print(separator + separator)
        print("evaluation: done!")
        print("mean loss:     %r" % loss_mean)
        print("mean accuracy: %r" % acc_mean)
        print("time (s):      %r" % time_total)
        if separator: print(separator + separator)

    return loss_mean, acc_mean


def compute_accuracy(src, tgt, mask = None):
    """Model accuracy."""

    out = torch.argmax(torch.softmax(src, dim=-1), dim=-1)

    out = out.contiguous().view(-1)
    tgt = tgt.contiguous().view(-1)

    if mask is not None:
      mask = mask.view(-1);
      mask = torch.logical_not(mask);
      out = out[mask]
      tgt = tgt[mask]

    # Empty
    if(len(tgt) == 0):
        return 1.0

    n_correct = (out == tgt)
    n_correct = torch.sum(n_correct).type(torch.float)

    acc = float(n_correct) / len(tgt)

    return acc


### File handling  

model_directory          = 'models';
model_file_name          = 'epoch_%s.pickle'
model_file_format        = '%06d'
model_best_directory     = 'models_best'
model_best_loss_file     = 'best_loss.pickle'
model_best_accuracy_file = 'best_accuracy.pickle'

results_directory        = 'results'
results_file_name        = 'results.csv'
results_best_file_name   = 'best_epochs.csv'

tensorboard_directory    = 'tensorboard'

generate_directory        = 'generate'
generate_midi_file_name   = 'generated_%r_%d.mid'  # primer, n_primer
generate_token_file_name  = 'generated_%r_%d.pickle'  # primer, n_primer
generate_primer_file_name = 'primer_%r_%d.mid'  # primer, n_primer


def directory_default(base_directory = None, directory = None, sub_directory = None, create = False, absolute = True):
    if directory is None:
        if base_directory is None:
            base_directory = '.';
        directory = base_directory;
        if sub_directory is not None:
           directory = os.path.join(directory, sub_directory) 
    if absolute:
        directory = os.path.abspath(directory);
    if create:
        os.makedirs(directory, exist_ok=True)
    return directory;


import re
def file_name_model(source, base_directory = None, directory = None, check_source = False, use_closest = False, return_epoch = False):
    if isinstance(source, int):
        directory = directory_default(base_directory=base_directory, directory=directory, sub_directory=model_directory);
        if source < 0:
          import glob
          files = sorted(glob.glob(os.path.join(directory, model_file_name % '*')));
          source = files[source];
          epoch = int(re.findall('[0-9]+', source)[-1]);
        else:
          epoch = source;
          source = os.path.join(directory,(model_file_name % model_file_format) % source)
    elif isinstance(source, float):  
        source = int(source);
        import glob
        directory = directory_default(base_directory=base_directory, directory=directory, sub_directory=model_directory);
        files = sorted(glob.glob(os.path.join(directory, model_file_name % '*')));
        if use_closest:
            import numpy as np
            model_epochs = np.array([int(re.findall('[0-9]+', file)[-1]) for file in files])
            source = np.argmin(np.abs(model_epochs - source));
        epoch = source;
        source = files[source];
    elif source == '*':
        directory = directory_default(base_directory=base_directory, directory=directory, sub_directory=model_directory);
        source = os.path.join(directory, model_file_name % source) 
        epoch = None;
    elif source in ['best', 'best_loss']:
        directory = directory_default(base_directory=base_directory, directory=directory, sub_directory=model_best_directory);
        source = os.path.join(directory, model_best_loss_file)
        epoch = None;
    elif source in ['best_accuracy', 'best_acc']:
        directory = directory_default(base_directory=base_directory, directory=directory, sub_directory=model_best_directory);
        source = os.path.join(directory, model_best_accuracy_file)
        epoch = None;
        
    if check_source and not isinstance(source, str):
        raise ValueError('source not valid: %r' % source);
    
    if return_epoch:
        return source, epoch;
    else:
        return source;


def load_model(model = None, source = None, base_directory = None, directory = None, use_closest = False, verbose = True, return_epoch = False, **kwargs):
    if model is None:
        model = create_model(**kwargs);
    
    source, epoch = file_name_model(source=source, base_directory=base_directory, directory=directory, use_closest=use_closest, return_epoch=True);
    if source is not None:
        if verbose:
            print('loading model: %s' % source)
        model.load_state_dict(torch.load(source, map_location=model.device()))

    if return_epoch:
        return model, epoch;
    else:
        return model;


def save_model(model, sink = None, base_directory = None, directory = None, verbose = True, prefix = 'saving model:'):
    sink =  file_name_model(source=sink, base_directory=base_directory, directory=directory);
            
    if sink is not None:
        if verbose:
          print(prefix + ' %s' % sink)
          
        torch.save(model.state_dict(), sink)
     
    return sink;



### User control

def train(
        epochs,  
        
        model = None,
        model_parameter = None,
        
        optimizer = None,
        optimizer_parameter = None,  
    
        loss = None,       
        loss_parameter = None,
        
        data_train = None,
        data_eval_train = None,
        data_eval_test  = None,
        data_directory = None,
        data_parameter = None,
         
        ### continue 
        continue_epoch = None,
        continue_model = None,
        
        ### control ###
        n_train_batches = None,
        n_evaluate_train_batches  = None,
        n_evaluate_test_batches   = None,
    
        save = 50,
        use_tensorboard = True,
        verbose = 1,
        separator = '===============================',
        
        ### dirs ###
        directory      = None,
        base_directory = None,
        clean_directories = False,
        
        device = None
    ):
    """Train the music transformer using hyerparameter for the model, optimizer and loss."""  
      
    directory_results     = directory_default(base_directory=base_directory, directory=directory, sub_directory=results_directory, create=True);
    directory_model       = directory_default(base_directory=base_directory, directory=directory, sub_directory=model_directory, create=True);
    directory_model_best  = directory_default(base_directory=base_directory, directory=directory, sub_directory=model_best_directory, create=True);
    if use_tensorboard:
       directory_tensorboard = directory_default(base_directory=base_directory, directory=directory, sub_directory=tensorboard_directory, create=True);
   
    if clean_directories:
        directories = [directory_results, directory_model, directory_model_best];
        if use_tensorboard:
            directories += [directory_tensorboard];
        for d in directories:
            os.system('rm %s' % os.path.join(d, '*')); 
    
    file_results = os.path.join(directory_results, results_file_name)
    file_best_epochs = os.path.join(directory_results, results_best_file_name)
    
    #file_model_best_loss     = file_name_model(source='best_loss',     directory=directory_model)
    #file_model_best_accuracy = file_name_model(source='best_accuracy', directory=directory_model)
  
    if verbose:
      print('train: results directory:     %s' % directory_results);
      print('train: model directory:       %s' % directory_model);  
      print('train: best model directory:  %s' % directory_model_best);
      if use_tensorboard:
          print('train: tensorboard directory: %s' % directory_tensorboard); 
          print('train: tensorboard command:')
          print('tensorboard --logdir %s' % directory_tensorboard)
      
      if not os.path.isfile(file_results) or continue_epoch is None:
        with open(file_results, "w", newline="") as o_stream:
          results_header = ["epoch", "learning rate", "learn train loss", "eval train loss" "eval train accuracy", "eval test loss", "eval test accuracy"]
          writer = csv.writer(o_stream)
          writer.writerow(results_header)
    
    best_test_loss_epoch = -1
    best_test_loss       = float("inf")
    best_test_acc_epoch  = -1
    best_test_acc        = 0.0
    
    ### tensorboard ###
    if use_tensorboard:
      from torch.utils.tensorboard import SummaryWriter
      tensorboard_summary = SummaryWriter(log_dir=directory_tensorboard)
    
    ### device ###
    if device is None:
        device = get_device();
    if verbose:
        print('train: using %r' % device);
    
    ### model and data ###
    if data_parameter is None:
        data_parameter = dict();
    if data_train is None:
       data_train, _ =  create_data(directory=data_directory, **data_parameter);
    if data_eval_train is None or data_eval_test is None:
       data_parameter.update(shuffle=False)
       data_eval_train, data_eval_test = create_data(directory=data_directory, **data_parameter);
    
    
    if model_parameter is None:
        model_parameter = dict();
    if model is None:
       model = create_model(device=device,**model_parameter)
    
    if optimizer_parameter is None:
        optimizer_parameter = dict();
    if optimizer is None:
       optimizer = create_optimizer(model=model, **optimizer_parameter)
     
    if loss_parameter is None:
        loss_parameter = dict();
    if loss is None:
        loss = create_loss(device=device, **loss_parameter);
    
    ### load previous training session ###
    if continue_model is not None:
        model = load_model(model=model, directory=directory_model, source=continue_model);
  
    start_epoch = 0;
    if continue_epoch is not None:
        if continue_model is None:
            model, epoch = load_model(model=model, directory=directory_model, source=continue_epoch, use_closest=True, return_epoch=True)
        start_epoch = epoch     
        optimizer.set_step(epoch);     
    
        if os.path.isfile(file_best_epochs):
          with open(file_best_epochs, newline="") as o_stream:
             reader = csv.reader(o_stream )
             for row in reader:
                 best = row;
          if len(best) > 0:
              best_test_loss_epoch = int(best[0])
              best_test_loss       = float(best[1])
              best_test_acc_epoch  = int(best[2])
              best_test_acc        = float(best[3])
    
    ### training ###
    for epoch in range(start_epoch, epochs):
        # Train
        if verbose:
            if separator: print(separator);
            print("Training epoch: %d/%d" % (epoch+1, epochs))
            if separator: print(separator);

        learn_train_loss = train_epoch(epoch, model, data_train, loss, optimizer, n_batches=n_train_batches, verbose=verbose)
        
        # Evaluation      
        if n_evaluate_train_batches is None or n_evaluate_train_batches > 0: 
            if verbose:
                if separator: print(separator)
                print("Evaluation on training data epoch: %d/%d" % (epoch+1, epochs))
                if separator: print(separator);
                
            eval_train_loss, eval_train_acc = evaluate_model(model, data_eval_train, loss, n_batches=n_evaluate_train_batches, verbose=verbose)
        else:
            eval_train_loss, eval_train_acc = float('inf'), 0.0
            
        if n_evaluate_test_batches is None or n_evaluate_test_batches > 0: 
            if verbose:
                if separator: print(separator)
                print("Evaluation on test data epoch: %d/%d" % (epoch+1, epochs))
                if separator: print(separator);
                
            eval_test_loss, eval_test_acc = evaluate_model(model, data_eval_test, loss, n_batches=n_evaluate_test_batches, verbose=verbose)
        else:
            eval_test_loss, eval_test_acc = float('inf'), 0.0   
          

        # Save results
        learning_rate = optimizer.rate()
        new_best = False
        if eval_test_loss < best_test_loss:
            best_test_loss       = eval_test_loss
            best_test_loss_epoch = epoch+1
            save_model(model, sink='best_loss', directory=directory_model_best, 
                       prefix=('train: epoch %d saving model with best loss %r:' % (epoch, best_test_loss)))
            new_best = True

        if eval_test_acc > best_test_acc:
            best_test_acc = eval_test_acc
            best_test_acc_epoch  = epoch+1
            save_model(model, sink='best_accuracy', directory=directory_model_best,
                       prefix=('train: epoch %d saving model with best accuracy %r:' % (epoch, best_test_acc)))
            new_best = True
         
        if new_best:
            with open(file_best_epochs, "w", newline="") as o_stream:
                writer = csv.writer(o_stream)
                best_header = ['best loss epoch', 'best loss', 'best accuracy epoch', 'best accuracy'];
                writer.writerow(best_header)
                best = [best_test_loss_epoch, best_test_loss, best_test_acc_epoch, best_test_acc]
                writer.writerow(best);
      
        if (epoch % save == 0):
            save_model(model, sink=epoch, directory=directory_model, prefix=('train: epoch %d saving model' % epoch))
            
        with open(file_results, "a", newline="") as o_stream:
            writer = csv.writer(o_stream)
            writer.writerow([epoch, learning_rate, learn_train_loss, eval_train_loss, eval_train_acc, eval_test_loss, eval_test_acc])
        
        if use_tensorboard:
            tensorboard_summary.add_scalar("loss/learn",          learn_train_loss,  global_step=epoch)
            tensorboard_summary.add_scalar("loss/train",          eval_train_loss,   global_step=epoch)     
            tensorboard_summary.add_scalar("loss/test",           eval_test_loss,    global_step=epoch)
            tensorboard_summary.add_scalar("accuracy/train",      eval_train_acc,    global_step=epoch)
            tensorboard_summary.add_scalar("accuracy/test",       eval_test_acc,     global_step=epoch)
            tensorboard_summary.add_scalar("learning_rate/train", learning_rate,  global_step=epoch)
            tensorboard_summary.flush()

    if use_tensorboard:
        tensorboard_summary.flush()

    return model, optimizer, loss;




def generate(
            sequence_length = 1024,
            model_sequence_length = 512,
            primer = None,
            max_primer_tokens = 10, 
            generate_parameter = None,
            
            model = None,
            model_parameter = None,
            
            continue_model = None,
            continue_epoch = None,
            
            data = None,
            data_directory = None,
            data_parameter = None,
                   
            plot_primer = False,
            plot = False,
            play = False,
            
            save_midi   = False,
            save_tokens = False,
            save_primer = False,
            
            return_midi = True,
            return_tokens = False,
            return_primer = False,
            return_full_primer = False,
            return_probabilities = False,
            return_model = False,
            
            directory      = None,
            base_directory = None,
            
            verbose = False
    ):
    
    # model 
    if model_parameter is None:
        model_parameter = dict();
    if model is None:
       model = create_model(**model_parameter)
    if continue_model is not None:
        model = load_model(model=model, base_directory=base_directory, directory=directory, source=continue_model);
    elif continue_epoch is not None:
        model = load_model(model=model, base_directory=base_directory, directory=directory, source=continue_epoch, use_closest=True)
    model.eval()
    
    # primer
    if data_parameter is None:
        data_parameter = dict();
        
    if primer is None or isinstance(primer, int):
        if data is None:
          data, _ =  create_data(directory=data_directory, batch_size=1, **data_parameter); 
        if primer is None:
            import numpy as np
            primer = np.rand.randrange(0,len(data));
        primer_label = primer;
        primer = data[primer].src[0]
        if return_full_primer:
            primer_full = primer;
        n_primer_tokens = torch.where(primer == TOKEN_PAD)[0];
        if len(n_primer_tokens) > 0:
            primer = primer[:n_primer_tokens[0]];
    elif isinstance(primer, utils.pm.PrettyMIDI):
        primer = encoder.encode_midi(primer)
        primer_label = 'custom';
    
    n_primer_tokens = min(len(primer), max_primer_tokens);
    primer = primer[:n_primer_tokens];   
    
    if plot_primer:
        utils.plot(primer);

    # saving
    if save_midi or save_tokens or save_primer:
        directory_generate = directory_default(base_directory=base_directory, directory=directory, sub_directory=generate_directory, create=True);
        if verbose:
            print('generate: directory: %s' % directory_generate);
                
    if save_primer:
        if not isinstance(save_primer, str):
            save_primer = generate_primer_file_name % (primer_label, n_primer_tokens);
            save_primer = os.path.join(directory_generate, save_primer)
        midi = encoder.decode_midi(primer);
        if verbose:
            print('generate: saving primer to %r' % save_primer)
        midi.write(save_primer)

    
    if generate_parameter is None:
        generate_parameter = dict();
    for k,v in zip(['sequence_length', 'model_sequence_length', 'end_token', 'ignore_token', 'verbose'],
                    [sequence_length,   model_sequence_length,   TOKEN_END,   TOKEN_PAD,      verbose]):
       if k not in generate_parameter.keys():
           generate_parameter[k] = v;
    
    with torch.no_grad():
        if return_probabilities:
            tokens, probabilities = model.generate(primer, return_probabilities=return_probabilities, **generate_parameter);
            probabilities = probabilities.cpu().detach().numpy();
        else:
            tokens = model.generate(primer, **generate_parameter);
    tokens = tokens.cpu().detach().numpy();
    
    if return_midi or save_midi or plot or play:
        midi = encoder.decode_midi(tokens);
    
    if save_midi:
        if not isinstance(save_midi, str):
            save_midi = generate_midi_file_name % (primer_label, n_primer_tokens);
            save_midi = os.path.join(directory_generate, save_midi)
        if verbose:
            print('generate: saving midi to %r' % save_midi)
        midi.write(save_midi)
        
    if save_tokens:
        if not isinstance(save_tokens, str):
            save_tokens = generate_token_file_name % (primer_label, n_primer_tokens);
            save_tokens = os.path.join(directory_generate, save_tokens)
        if verbose:
            print('generate: saving tokens to %r' % save_tokens)
        o_stream = open(save_tokens, "wb")
        pickle.dump(tokens, o_stream)
        o_stream.close()
    
    if plot:
        utils.plot(midi);
    if play:
        utils.play(midi);
    
    result = tuple();
    if return_midi:
        result += (midi,)
    if return_tokens:
        result += (tokens,);
    if return_primer:
        result += (primer,);
    if return_full_primer:
        result += (primer_full,)
    if return_probabilities:
        result += (probabilities,);
    if return_model:
        result += (model,)
    if len(result) == 1:
        result = result[0];
    return result;
        





# class Workspace:
    
#     def __init__(self, 
#                  base_directory,
#                  model_directory          = 'models';
#                  model_file_name          = 'epoch_%s.pickle'
#                  model_file_format        = '%06d'
#                  model_directory_best     = 'best_models'
#                  model_file_best_loss     = 'best_loss.pickle'
#                  model_file_best_accuracy = 'best_accuracy.pickle'):
        
#         self.base_directory           = base_directory;
#         self.model_directory          = model_directory
#         self.model_file_name          = model_file_name
#         self.model_file_format        = model_file_format
#         self.model_directory_best     = model_directory_best
#         self.model_file_best_loss     = model_file_best_loss
#         self.model_file_best_accuracy = model_file_best_accuracy