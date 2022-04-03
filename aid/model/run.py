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

import torch
import torch.nn

from aid.model.transformer import Transformer, Optimizer, Loss
from aid.dataset.groove import train_test_dataloaders, CODE_SIZE, CODE_PAD, CODE_END
import aid.dataset.midi_encoder as encoder

import aid.dataset.midi_utils as utils;

### Object creation

def create_model(n_src_vocab = CODE_SIZE, **kwargs):
    return Transformer.create(n_src_vocab=n_src_vocab, **kwargs);
    

def create_optimizer(model, **kwargs):
    return Optimizer.create(model=model, **kwargs);


def create_loss(n_src_vocab = CODE_SIZE, ignore_code = CODE_PAD, **kwargs):
    return Loss(n_vocab=n_src_vocab, ignore_code=ignore_code, **kwargs);


def create_data(directory, **kwargs):
    return train_test_dataloaders(directory=directory, **kwargs);


### Training and Evaluation 

def train_epoch(epoch, model, data, loss, optimizer, n_batches = None, verbose = True, separator = '=========='):
    """Trains the Music transformer model for one epoch."""
    
    model.train();
    
    if n_batches is None:
        n_batches = len(data);
    if verbose is True:
        verbose = n_batches;
    if verbose:
        time_start = time.time();
        
    loss_total = 0.0;
    code_total = 0;
    
    for b, batch in enumerate(data):
        if b >= n_batches:
            break;
        #print(b, batch)
        
        if verbose:
            time_batch_start = time.time()
  
        if optimizer:
          optimizer.zero_grad()
  
        src = batch.src
        src_mask = batch.src_mask();
        tgt = batch.tgt
        nrm = batch.n_tgt_codes();
  
        fwd = model(src, src_mask)
         
        if torch.any(torch.isnan(fwd)):
            print('Nans encountered in model')
            return batch;
  
        out = loss(fwd.contiguous().view(-1, fwd.size(-1)), 
                   tgt.contiguous().view(-1)) / nrm
        
        if torch.any(torch.isnan(out)):
            print('Nans encountered in loss')
            return batch;
  
        out.backward()
        
        if optimizer:
            optimizer.step()
        
        loss_total += float(out);
        code_total += float(nrm);
        
        if verbose and ((b+1) % verbose == 0):
            time_batch_end = time.time()
            time_batch_total = time_batch_end - time_batch_start;
              
            if separator: print(separator)
            print('training: epoch %d  batch %d/%d (%d) batch_size %d' % (epoch+1, b+1, n_batches, len(data), batch.batch_size()))
            print("lrate:    %r" % optimizer.rate())
            print("loss:     %r" % (float(out) / float(nrm)))
            print("time (s): %r" % time_batch_total)
            if separator: print(separator)

    loss_mean = loss_total / code_total;

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
       
    if n_batches is None:
        n_batches = len(data);
    if verbose is True:
        verbose = n_batches;
    if verbose:
        time_start = time.time();
   
    if n_batches == 0:
        return float('inf'), 0;
   
    loss_total = 0.0
    acc_total  = 0.0
    codes_total = 0;
    
    with torch.set_grad_enabled(False):
        for b, batch in enumerate(data):
            if b >= n_batches:
                break;
                
            if verbose:
                time_batch_start = time.time()
            
            src = batch.src
            src_mask = batch.src_mask();
            tgt = batch.tgt
            nrm = float(batch.n_tgt_codes());

            fwd = model(src, src_mask)

            acc = float(compute_accuracy(fwd, tgt, src_mask));
            acc_total += acc * nrm;
            
            out = loss.forward(fwd.contiguous().view(-1, fwd.size(-1)), 
                               tgt.contiguous().view(-1)) / nrm
  
            out = float(out);
            loss_total += out;
            codes_total += nrm

            if verbose and ((b+1) % verbose == 0):
                time_batch_end = time.time()
                time_batch_total = time_batch_end - time_batch_start;
                  
                if separator: print(separator)
                print('evaluation:  batch %d/%d' % (b+1, len(data)))
                print("loss:     %r" % (out / nrm))
                print("accuracy: %r" % acc)
                print("time (s): %r" % time_batch_total)
                if separator: print(separator)  
        
        loss_mean = loss_total / codes_total
        acc_mean  = acc_total / codes_total

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
generate_midi_file_name   = 'generated_%d_%d.mid'  # primer, n_primer
generate_code_file_name   = 'generated_%d_%d.npy'  # primer, n_primer
generate_primer_file_name = 'primer_%d_%d.npy'


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
        model_parameter = dict(),
        
        optimizer = None,
        optimizer_parameter = dict(),  
    
        loss = None,       
        loss_parameter = dict(),
        
        data_train = None,
        data_eval_train = None,
        data_eval_test  = None,
        data_directory = None,
        data_parameter = dict(),
         
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
        base_directory = None
    ):
    """Train the music transformer using hyerparameter for the model, optimizer and loss."""  
      
    directory_results     = directory_default(base_directory=base_directory, directory=directory, sub_directory=results_directory, create=True);
    directory_model       = directory_default(base_directory=base_directory, directory=directory, sub_directory=model_directory, create=True);
    directory_model_best  = directory_default(base_directory=base_directory, directory=directory, sub_directory=model_best_directory, create=True);
    if use_tensorboard:
       directory_tensorboard = directory_default(base_directory=base_directory, directory=directory, sub_directory=tensorboard_directory, create=True);
   
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
    
    ### model and data ###
    if data_train is None:
       data_train, _ =  create_data(directory=data_directory, **data_parameter);
    if data_eval_train is None or data_eval_test is None:
       data_parameter.update(shuffle=False)
       data_eval_train, data_eval_test = create_data(directory=data_directory, **data_parameter);
    
    if model is None:
       model = create_model(**model_parameter)
    
    if optimizer is None:
       optimizer = create_optimizer(model=model, **optimizer_parameter)
    
    if loss is None:
        loss = create_loss(**loss_parameter);
    
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
        if n_evaluate_train_batches > 0: 
            if verbose:
                if separator: print(separator)
                print("Evaluation on training data epoch: %d/%d" % (epoch+1, epochs))
                if separator: print(separator);
                
            eval_train_loss, eval_train_acc = evaluate_model(model, data_eval_train, loss, n_batches=n_evaluate_train_batches, verbose=verbose)
        else:
            eval_train_loss, eval_train_acc = float('inf'), 0.0
            
        if n_evaluate_test_batches > 0: 
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
            tensorboard_summary.add_scalar("accuracy/eval",       eval_test_acc,     global_step=epoch)
            tensorboard_summary.add_scalar("learning_rate/train", learning_rate,  global_step=epoch)
            tensorboard_summary.flush()

    if use_tensorboard:
        tensorboard_summary.flush()

    return model, optimizer, loss;




def generate(
        primer = None,
        n_primer = 10, 
        generate_parameter = dict(),
        
        model = None,
        model_parameter = dict(),
        
        continue_model = None,
        continue_epoch = None,
        
        data = None,
        data_directory = None,
        data_parameter = dict(),
               
        plot_primer = False,
        plot = False,
        play = False,
        
        save_midi = False,
        save_code = False,
        save_primer = False,
        
        return_midi = True,
        return_code = False,
        return_probabilities = False,
        
        directory      = None,
        base_directory = None,
        
        verbose = False
    ):
    
    if save_midi or save_code or save_primer:
        directory_generate = directory_default(base_directory=base_directory, directory=directory, sub_directory=generate_directory, create=True);
    
        if verbose:
            print('generate: directory: %s' % directory_generate);
    
    # model 
    if model is None:
       model = create_model(**model_parameter)
    if continue_model is not None:
        model = load_model(model=model, base_directory=base_directory, directory=directory, source=continue_model);
    elif continue_epoch is not None:
        model = load_model(model=model, base_directory=base_directory, directory=directory, source=continue_epoch, use_closest=True)
    model.eval()
    
    # primer
    primer_save = primer;
    if primer is None or isinstance(primer, int):
        if data is None:
          data, _ =  create_data(directory=data_directory, **data_parameter); 
        if primer is None:
            import numpy as np
            primer = np.rand.randrange(0,len(data));
        primer_save = primer;
        primer = data[primer].src
        max_length = np.where(primer == CODE_PAD)[0];
        if len(max_length) > 0:
            primer = primer[:max_length[0]];
    elif isinstance(primer, utils.pm.PrettyMIDI):
        primer = encoder.encode_midi(primer)
    
    n_primer = min(len(primer), n_primer);
    primer = primer[:n_primer];     

    if save_primer is not None:
        if not isinstance(save_primer, str):
            primer_midi_file = generate_primer_file_name % (primer_save, n_primer);
        midi = encoder.decode_midi(primer);
        midi.write(os.path.join(directory_generate, primer_midi_file))
        
    if plot_primer:
        utils.plot(primer);
    
    generate_parameter.update(
        end_code = CODE_END,
        pad_code = CODE_PAD,
    );
    
    if return_probabilities:
        sequence, probabilities = model.generate(primer, return_probabilities=return_probabilities, **generate_parameter);
    else:
        sequence = model.generate(primer, **generate_parameter);
    
    #return model, primer, sequence;
    code = sequence[0].cpu().detach().numpy();
    midi = encoder.decode_midi(code);
    
    if plot:
        utils.plot(midi);
    if play:
        utils.play(midi);
    
    result = tuple();
    if return_midi:
        result += (midi,)
    if return_code:
        result += (code,);
    if return_probabilities:
        result += (probabilities[0],);
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