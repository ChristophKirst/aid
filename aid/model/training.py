#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Drummer Project

Project to create an AI drummer that performs with Pianist Jenny Q Chai
"""

# Train music transformer for AI drummer / test speed and quality

import os
import csv

import torch
import torch.nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Adam

from aid.dataset.groove import initialize_datasets

from aid.model.music_transformer import MusicTransformer
from aid.model.loss import SmoothCrossEntropyLoss
import aid.model.learning_rate_scheduling as lrs

import aid.utils.midi_encoder as encoder



def device():
  if torch.cuda.device_count() > 0:
    return torch.device("cuda")
  else:
    #print('Warning: No gpu device found!')
    return torch.device("cpu");

separator = "======================"


def train(
      ### parameter ###
      max_sequence = 2048,   # maximal midi sequence
      batch_size   = 2,      # batch size
      epochs       = 100,    # epochs 
      n_workers    = 1,      # threads
      
      n_layers      = 6,     # layers in transformer
      n_heads       = 8,     # multi-head attention
      d_model       = 512,   # dimension of model
      d_feedforward = 1024,  # feed forward layer dimension
      dropout       = 0.1,   # dropout rate
      rpr           = True,  # relative position encoding
    
      learning_rate = None, # None = custom schedyuler or constant
      learning_rate_start    = 1.0,
      learning_warmup_steps  = 4000,
      learning_adam_beta_1   = 0.9,
      learning_adam_beta_2   = 0.98,
      learning_adam_epsilon  = 10e-9,
      
      loss_smoothing = None,
      
      ### control ###
      save_weights = 10,
      use_tensorboard = True,
      verbose   = 1,
      
      ### continue
      continue_weights = None,
      continue_epoch = None,
      
      ### files and dirs ###
      directory_input   = '/home/ckirst/Media/Music/AImedia/MLMusic/Data/groove_encoded',
      directory_results = '/home/ckirst/Desktop/aid_results'
  ):
  """Traiin the music transformer."""  
  
  directory_models = os.path.join(directory_results, 'models')
  directory_output = os.path.join(directory_results, 'results');
  
  for d in [directory_results, directory_models, directory_output]:
    os.makedirs(d, exist_ok=True)
    
  file_results = os.path.join(directory_output, "results.csv")
  file_best_loss = os.path.join(directory_output, "best_loss_weights.pickle")
  file_best_accuracy = os.path.join(directory_output, "best_acc_weights.pickle")
  file_best_epochs = os.path.join(directory_output, "best_epochs.txt")

  if verbose:
    print('Train: results directory: %s' % directory_results);
    print('Train: output directory:  %s' % directory_output);  
    print('Train: models  directory: %s' % directory_models);
  
  results_header = ["Epoch", "Learn rate", "Avg Train loss", "Train Accuracy", "Avg Eval loss", "Eval accuracy"]
  
  
  ### tensorboard #####
  if not use_tensorboard:
    tensorboard_summary = None
  else:
    from torch.utils.tensorboard import SummaryWriter
    directory_tensorboad = os.path.join(directory_results, "tensorboard")
    tensorboard_summary = SummaryWriter(log_dir=directory_tensorboad)
    
    
  ### datasets ###
  dataset_train, dataset_validate, dataset_test = initialize_datasets(directory_input, max_sequence)

  loader_train    = DataLoader(dataset_train,     batch_size=batch_size, num_workers=n_workers, shuffle=True)
  loader_test     = DataLoader(dataset_validate,  batch_size=batch_size, num_workers=n_workers)

  model = MusicTransformer(n_layers=n_layers, n_heads=n_heads,
                           d_model=d_model, d_feedforward=d_feedforward, dropout=dropout,
                           max_sequence=max_sequence, rpr=rpr).to(device())

  
  ### load previous training session ###
  start_epoch = -1;
  if continue_weights is not None:
    if continue_epoch is None:
      print("Train: Need epoch number to continue training")
      return
    else:
      model.load_state_dict(torch.load(continue_weights))
      start_epoch = continue_epoch

  ### learning rater ###
  if learning_rate is None:
    if continue_epoch is None:
      init_step = 0
    else:
      init_step = continue_epoch * len(loader_train)

    learning_rate = learning_rate_start;
    learning_rate_stepper = lrs.LrStepTracker(d_model, learning_warmup_steps, init_step)


  ### loss ###
  loss_function_eval = torch.nn.CrossEntropyLoss(ignore_index=encoder.ENCODE_TOKEN_PAD)

  if loss_smoothing is None:
    loss_function_train = loss_function_eval
  else:
    loss_function_train = SmoothCrossEntropyLoss(loss_smoothing, encoder.ENCODE_SIZE, ignore_index=encoder.ENCODE_TOKEN_PAD)

  ### optimizer ###
  optimizer = Adam(model.parameters(), lr=learning_rate, 
                   betas=(learning_adam_beta_1, learning_adam_beta_2), 
                   eps=learning_adam_epsilon)

  if learning_rate is None:
    learning_rate_scheduler = LambdaLR(optimizer, learning_rate_stepper.step)
  else:
    learning_rate_scheduler = None

  ### best evaluation accuracy ###
  best_eval_acc        = 0.0
  best_eval_acc_epoch  = -1
  best_eval_loss       = float("inf")
  best_eval_loss_epoch = -1

  ### results ###
  if not os.path.isfile(file_results):
    with open(file_results, "w", newline="") as o_stream:
      writer = csv.writer(o_stream)
      writer.writerow(results_header)


  ### training ###
  for epoch in range(start_epoch, epochs):
    # Baseline has no training and acts as a base loss and accuracy (epoch 0 in a sense)
    if(epoch > -1):
      print(separator)
      print("epoch: %d/%d" % (epoch+1, epochs))
      print(separator)
      print("")

      # Train
      train_epoch(epoch+1, model, loader_train, loss_function_train, optimizer, learning_rate_scheduler, verbose)

      print(separator)
      print("Evaluating:")
    else:
      print(separator)
      print("Baseline model evaluation (Epoch 0):")

    # Eval
    train_loss, train_acc = evaluate_model(model, loader_train, loss_function_train)
    eval_loss,  eval_acc  = evaluate_model(model, loader_test,  loss_function_eval)
    
    learning_rate = lrs.get_lr(optimizer)

    print("Epoch:", epoch+1)
    print("Avg train loss:", train_loss)
    print("Avg train acc:", train_acc)
    print("Avg eval loss:", eval_loss)
    print("Avg eval acc:", eval_acc)
    print(separator)
    print("")

    new_best = False

    if eval_acc > best_eval_acc:
      best_eval_acc = eval_acc
      best_eval_acc_epoch  = epoch+1
      torch.save(model.state_dict(), file_best_accuracy)
      new_best = True

    if eval_loss < best_eval_loss:
      best_eval_loss       = eval_loss
      best_eval_loss_epoch = epoch+1
      torch.save(model.state_dict(), file_best_loss)
      new_best = True

    if new_best:
      with open(file_best_epochs, "w") as o_stream:
        print("Best eval acc epoch:", best_eval_acc_epoch, file=o_stream)
        print("Best eval acc:", best_eval_acc, file=o_stream)
        print("")
        print("Best eval loss epoch:", best_eval_loss_epoch, file=o_stream)
        print("Best eval loss:", best_eval_loss, file=o_stream)


    if use_tensorboard:
      tensorboard_summary.add_scalar("Avg_CE_loss/train", train_loss, global_step=epoch+1)
      tensorboard_summary.add_scalar("Avg_CE_loss/eval", eval_loss, global_step=epoch+1)
      tensorboard_summary.add_scalar("Accuracy/train", train_acc, global_step=epoch+1)
      tensorboard_summary.add_scalar("Accuracy/eval", eval_acc, global_step=epoch+1)
      tensorboard_summary.add_scalar("Learn_rate/train", learning_rate, global_step=epoch+1)
      tensorboard_summary.flush()

    if (epoch+1) % save_weights == 0:
      epoch_str = str(epoch+1).zfill(6)
      path = os.path.join(directory_models, "epoch_" + epoch_str + ".pickle")
      torch.save(model.state_dict(), path)

    with open(file_results, "a", newline="") as o_stream:
      writer = csv.writer(o_stream)
      writer.writerow([epoch+1, learning_rate, train_loss, train_acc, eval_loss, eval_acc])

  if use_tensorboard:
    tensorboard_summary.flush()

  return



# train_epoch

import time

def train_epoch(epoch, model, dataloader, loss, optimizer, learning_rate_scheduler=None, verbose=1):
    out = -1
    model.train()
    for batch_num, batch in enumerate(dataloader):
      time_before = time.time()

      optimizer.zero_grad()

      x   = batch[0].to(device())
      tgt = batch[1].to(device())

      y = model(x)

      y   = y.reshape(y.shape[0] * y.shape[1], -1)
      tgt = tgt.flatten()

      out = loss.forward(y, tgt)

      out.backward()
      optimizer.step()

      if(learning_rate_scheduler is not None):
        learning_rate_scheduler.step()

      time_after = time.time()
      time_took = time_after - time_before

      if((batch_num+1) % verbose == 0):
        print(separator)
        print("Epoch", epoch, " Batch", batch_num+1, "/", len(dataloader))
        print("LR:", lrs.get_lr(optimizer))
        print("Train loss:", float(out))
        print("Time (s):", time_took)
        print(separator)
        print("")

    return
  

def evaluate_model(model, dataloader, loss):

    model.eval()

    avg_acc     = -1
    avg_loss    = -1
    with torch.set_grad_enabled(False):
        n_test      = len(dataloader)
        sum_loss   = 0.0
        sum_acc    = 0.0
        for batch in dataloader:
            x   = batch[0].to(device())
            tgt = batch[1].to(device())

            y = model(x)

            sum_acc += float(compute_accuracy(y, tgt))

            y   = y.reshape(y.shape[0] * y.shape[1], -1)
            tgt = tgt.flatten()

            out = loss.forward(y, tgt)

            sum_loss += float(out)

        avg_loss    = sum_loss / n_test
        avg_acc     = sum_acc / n_test

    return avg_loss, avg_acc


# compute_groove_accuracy
def compute_accuracy(out, tgt):
    """accuracy"""

    softmax = torch.nn.Softmax(dim=-1)
    out = torch.argmax(softmax(out), dim=-1)

    out = out.flatten()
    tgt = tgt.flatten()

    mask = (tgt != encoder.ENCODE_TOKEN_PAD)

    out = out[mask]
    tgt = tgt[mask]

    # Empty
    if(len(tgt) == 0):
        return 1.0

    n_correct = (out == tgt)
    n_correct = torch.sum(n_correct).type(torch.float)

    acc = n_correct / len(tgt)

    return acc