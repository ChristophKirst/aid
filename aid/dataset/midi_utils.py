#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Drummer Project

Project to create an AI drummer that performs with Pianist Jenny Q Chai

Utils
"""

import os
import subprocess
import tempfile


import pretty_midi as pm

#fluidsynth
#import aid.external.fluidsynth as fluidsynth
#pm.fluidsynth = fluidsynth
#pm.instrument._HAS_FLUIDSYNTH = True


import scipy.io.wavfile as siow

import music21 as m21

from IPython.display import display, Audio, Image

import aid.dataset.midi_encoder as encoder


def midi_to_waveform(midi, sf2_path = '/usr/share/sounds/sf2/FluidR3_GM.sf2'):
    if isinstance(midi, str):
       midi = pm.PrettyMIDI(midi_file=midi)
    waveform = midi.fluidsynth(sf2_path=sf2_path);
    return waveform


def midi_to_wav(midi, filename_wav, rate = 44100):
    waveform = midi_to_waveform(midi);
    siow.write(filename_wav, rate, waveform)
    return filename_wav;


def play_midi(midi, backend = 'vlc', rate = 44100):
    waveform = midi_to_waveform(midi);
    if backend == 'vlc':
        temp = tempfile.mktemp(suffix='.wav')
        siow.write(temp, rate, waveform)
        os.system('vlc %s' % temp)
        os.remove(temp);
    elif backend == 'jupyter':     
        return Audio(waveform, rate=44100)


def plot_midi(midi, ptype = 'musescore', measures = all, **kwargs):
    
    if isinstance(measures, int):
         measures = (0,measures);
    if not isinstance(measures, tuple):
        measures = None;
        
    if not isinstance(midi, str):
        file_midi = tempfile.mktemp(suffix='.mid');
        midi.write(file_midi);
        midi = file_midi;
        def clean_up_midi():
            os.remove(file_midi);
    else:
        def clean_up_midi():
            pass
    
    if ptype == 'musescore':
        
        file_png = tempfile.mktemp(suffix='.png');
        subprocess.call(['musescore', midi, '-o', file_png, '-T', '0'], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT);
        #os.system('musescore %s -o %s -T 0' % (filename, file_png));
        for number in ('1', '01', '001', '0001', '00001'):
          search = file_png[:-4] + '-' + number + '.png'
          if os.path.isfile(search):
              file_png = search;
        #print(file_png)
    
        with open(file_png, 'rb') as f:
            data = f.read()
        os.remove(file_png);
        clean_up_midi();
        
        display(Image(data=data, retina=True)) 
    
    elif ptype == 'score':
        m = m21.converter.parse(midi)
        if measures:
            m =  m.measures(*measures)
        clean_up_midi();
        return m.show(**kwargs)
    elif ptype == 'bar':
        m = m21.converter.parse(midi)
        if measures:
            m =  m.measures(*measures)
        clean_up_midi();
        return m.plot(**kwargs)
    
    
def play_code(code, **kwargs):
    midi = encoder.decode_midi(code);
    play_midi(midi, **kwargs);

    
def plot_code(code, **kwargs):
    midi = encoder.decode_midi(code);
    plot_midi(midi, **kwargs);
    
    
def play(data, **kwargs):
    if not isinstance(data, pm.PrettyMIDI):
        data = encoder.decode_midi(data);
    play_midi(data, **kwargs);


def plot(data, **kwargs):
    if not isinstance(data, pm.PrettyMIDI):
        data = encoder.decode_midi(data);
    plot_midi(data, **kwargs);        
    


import numpy as np
import matplotlib.pyplot as plt        
from matplotlib.collections import LineCollection    

def plot_midi_bars(code, attention = None, attention_events = None, 
                   
                   pitches = None, pitch_to_index = None, pitch_labels = None,
                   
                   
                   plot_events = ['time_shift', 'velocity'], event_sizes = None, event_extends = None,
                   
                   plot_played_pitches_only = False, label_played_pitches_only = True,
                   
                   fig = None, linewidth = 5,):

    n_code = len(code);    
    events = encoder.decode_midi(code, return_events = True);
    
    if pitches is None:
        pitches = list(encoder.MIDI_PITCH_TO_INDEX.keys())
 
    if pitch_to_index is None:
        pitch_to_index = encoder.MIDI_PITCH_TO_INDEX;            
 
    if pitch_labels is None:
        pitch_labels = encoder.MIDI_INFO['MIDI'].to_list();
 
    if plot_played_pitches_only or label_played_pitches_only:
        played_indices = []
        for event in events:
            if event.type == 'note_on' and event.value not in played_indices:
                played_indices.append(event.value);
        played_indices = sorted(played_indices);
        pitch_labels = [pitch_labels[i] for i in played_indices]           
        
    if plot_played_pitches_only:
        index_to_pitch = { i : p  for p, i in pitch_to_index.items() }
        pitches = [index_to_pitch[i] for i in played_indices]
        pitch_to_index = { p: i for i,p in enumerate(pitches) } 
        played_indices = pitch_to_index.values()   
    #print(played_indices)
        
    n_pitches = len(pitches);
    #print(n_pitches)
    
    if event_sizes is None:
        event_sizes = encoder.CODE_EVENT_SIZES;
    
    if event_extends is None:  
        event_extends = dict(time_shift = 1,
                             velocity   = min(2, n_pitches),
                             control    = 0,
                             delimeter  = 1
                       );
    event_extends['note_on']  = n_pitches;
    event_extends['note_off'] = n_pitches;
    #print(event_extends)
    
    event_offsets = dict(note_on = 0, note_off = 0);
    offset = n_pitches;
    for etype in ['time_shift', 'velocity', 'control', 'delimeter']:
        event_offsets[etype] = offset;
        offset += event_extends[etype];
    ylim = offset;
    #print(event_offsets)
    
    # event_color_extends = dict(
    #     note_on    = 0,
    #     note_off   = 0,
    #     time_shift = n_pitches,
    #     velocity   = n_pitches,
    #     control    = 0,
    #     delimeter  = 1)
     
    # event_color_offsets = dict(note_on = 0, note_off = 0);
    # offset = 0;
    # for etype in ['time_shift', 'velocity', 'control', 'delimeter']:
    #     event_color_offsets[etype] = offset;
    #     offset += event_color_extends[etype];
    # color_max = offset;
    
    # def event_to_color(etype, value):
    #     return event_color_offsets[etype] +  event_color_extends[etype] * value /  event_sizes[etype]
    
    colormaps = dict(note_on    = 'rainbow',
                     time_shift = 'viridis',
                     velocity   = 'inferno',
                     control    = 'rainbow',
                     delimeter   = 'Pastel1')
    
    for k,v in colormaps.items():
        colormaps[k] = plt.get_cmap(v);

    def event_to_color(etype, value):
        return colormaps[etype](value /  event_sizes[etype])

 
    if attention is not None:
        positions = np.full((n_code, 2), np.nan);
    else:
        positions = None;
        
    segments = np.full((n_code, 2, 2), np.nan);
    segment_colors = [];
    
    time = 0;
    note_on_events = dict();
    for index, event in enumerate(events):
        if   event.type == 'time_shift':
             time_shift = encoder.time_shift_from_value(event.value);
             segments[index,:,0] = time, time + time_shift;
             segments[index,:,1] = event_offsets['time_shift'], event_offsets['time_shift'] + event_extends['time_shift'] * event.value / event_sizes['time_shift'];
             segment_colors.append(event_to_color('time_shift', event.value))
             if positions is not None:
                 positions[index,:] = time+time_shift/2.0, event_offsets['time_shift'] + event_extends['time_shift'] * event.value / event_sizes['time_shift'] / 2
             time += time_shift;        
        elif event.type == 'velocity':
            #velocity = encoder.velocity_from_value(event.value);
            velocity = event.value;
            segments[index,:,0] = time, time;
            segments[index,:,1] = event_offsets['velocity'], event_offsets['velocity'] + event_extends['velocity'] * event.value / event_sizes['velocity'];
            segment_colors.append(event_to_color('velocity', event.value))
            if positions is not None:
                 positions[index,:] = time, event_offsets['velocity'] +  event_extends['velocity'] * event.value  / event_sizes['velocity'] / 2
        elif event.type == 'control':
            segments[index,:,0] = time, time;
            segments[index,:,1] = event_offsets['control'], event_offsets['control'] + event_extends['control'] * event.value / event_sizes['control'] ;
            segment_colors.append(event_to_color('control', event.number))
            if positions is not None:
                 positions[index,:] = time, event_offsets['control'] +  event_extends['control'] * event.value / event_sizes['control']  / 2
        elif event.type == 'delimeter':
            segments[index,:,0] = time, time;
            segments[index,:,1] = event_offsets['delimeter'], event_offsets['delimeter'] + event_extends['delimeter'];
            segment_colors.append(event_to_color('delimeter', event.value))
            if positions is not None:
                 positions[index,:] = time, event_offsets['delimeter'] +  event_extends['delimeter'] * event.value / event_sizes['delimeter']  / 2   
        elif event.type == 'note_on':
            event.time = time;
            note_on_events[event.value] = event;
            if positions is not None:
                 positions[index,:] = time, event_offsets['note_on'] +  event_extends['note_on'] * event.value / event_sizes['note_on'];
        elif event.type == 'note_off':
            try:
                on = note_on_events[event.value]
                #off = event
                segments[index,:,0] = on.time, time;
                segments[index,:,1] = event_offsets['note_off'] + event_extends['note_off'] * on.value / event_sizes['note_off'];
                segment_colors.append(event_to_color('velocity', velocity))
                
                if positions is not None:
                 positions[index,:] = time, event_offsets['note_off'] +  event_extends['note_off'] * event.value / event_sizes['note_off'];
                
            except:
                print('removed event: %r' % event);    
        else:
            print('unknnown event: %r' % event);
            
    segments = segments[np.logical_not(np.isnan(segments[:,0,0]))];                 
    lines = LineCollection(segments, colors=segment_colors, linewidth=linewidth);
        
    if fig is None:
        fig = plt.gcf();
        ax = fig.subplots();
    
    dlt = time * 0.1;
    ax.set_xlim(-dlt, time + dlt)
    ax.set_ylim(-0.5, ylim + 0.5)
    
    ax.add_collection(lines)
    
    
    if label_played_pitches_only:
        ticks = list(played_indices);
    else:
        ticks = list(range(n_pitches));
    labels = pitch_labels;
    #print(labels, ticks)
    #print(positions)
    
    for etype in ['time_shift', 'velocity', 'control', 'delimeter']:
        if event_extends[etype] > 0:
            ticks.append(event_offsets[etype]);
            labels.append(etype);
    
    ax.set_yticks(ticks, labels) 
    
    n_attention_line_points = 30;
    def arc_line(start, end, weight, n = n_attention_line_points, h = 0.005):
        delta = end - start;
        l = np.linalg.norm(delta, axis=-1);       
        valid = np.logical_and(l > 0, weight > 0);
        start = start[valid];
        end = end[valid];
        delta = delta[valid];
        l = l [valid];         
        h = h * l;
        q = ((l/2)**2 - h**2) / (2 * h)
        r = q + h;
        center = start + 0.5 * delta + q[:,np.newaxis] / l[:,np.newaxis] * np.array([-delta[...,1], delta[...,0]]).T;
        radius_start = start - center;
        phi_start = np.arctan2(radius_start[...,1], radius_start[...,0]);
        radius_end = end - center;
        phi_end = np.arctan2(radius_end[...,1], radius_end[...,0]);
        correct = np.abs(phi_start - phi_end) > np.pi;
        phi_end[correct] += -np.sign(phi_end[correct] - phi_start[correct]) * 2 * np.pi;
        phis = np.linspace(phi_start, phi_end, n).T;
        line = center[np.newaxis,:] + r[np.newaxis,:,np.newaxis] * np.array([np.cos(phis), np.sin(phis)]).transpose();
        line = line.transpose(1,0,2);
        return line, valid;
    
    if attention is not None:
        if attention.ndim == 1:
           attention = attention[np.newaxis,:];
           attention_refs = positions[-1:] + 0 * np.array([dlt/2,0])[np.newaxis,:];
        else:
           attention_refs = positions;
           
        alines = [];
        alinewidths = [];
        alinecolors = [];
        acmap = plt.get_cmap('Reds')
        amax = attention.max();
        alinewidth = 5;
         
        for i,a in enumerate(attention):
            start = np.array([attention_refs[i]] * (len(a)-1));
            keep = np.ones(len(a), dtype=bool);
            keep[i] = False
            end   = positions[keep];
            a = a[keep]
            lines, valid = arc_line(start, end, a)
            a = a[valid];
            alines.append(lines);
            alinewidths.append(a/amax * alinewidth  + 1);
            alinecolors.append(acmap(a/amax));
        alines = np.concatenate(alines, axis=0);
        alinewidths = np.concatenate(alinewidths, axis=0);
        alinecolors = np.concatenate(alinecolors, axis=0);   
        #print(alines.shape);
        #print(alinecolors.shape)
       
        alines = LineCollection(alines, colors=alinecolors, linewidth=alinewidths);
        ax.add_collection(alines)
    
    # color bars
    colorbar_args = dict(ax=ax)
    n_colbars = 0;
    for etype, esize in list(event_extends.items())[::-1]:
        if etype not in ['note_on', 'note_off', 'delimeter'] and esize > 0:
            n_colbars += 1;
    if attention is not None:
            n_colbars += 1;
    
    colbar_pos = 0; colbar_width = 0.025;
    for etype, esize in list(event_extends.items())[::-1]:
        if etype not in ['note_on', 'note_off', 'delimeter'] and esize > 0:
            cax = ax.inset_axes([1.04, colbar_pos / n_colbars, colbar_width, 0.95/n_colbars], transform=ax.transAxes)
            colbar_pos +=1
            axcb = fig.colorbar(plt.cm.ScalarMappable(cmap=colormaps[etype]),cax=cax, **colorbar_args)
            axcb.set_label(etype)
    if attention is not None:
        cax = ax.inset_axes([1.04, colbar_pos / n_colbars, colbar_width, 1/n_colbars])
        axcb = fig.colorbar(plt.cm.ScalarMappable(cmap=acmap), cax=cax, **colorbar_args)
        axcb.set_label('attention')
    
    plt.tight_layout();
    plt.show();



    
    