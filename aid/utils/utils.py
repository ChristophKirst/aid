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

import aid.utils.midi_encoder as encoder



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