#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Notes and References


https://wiki.python.org/moin/PythonInMusic

#midi
controls
https://www.midi.org/specifications-old/item/table-3-control-change-messages-data-bytes-2
instruments
https://www.midi.org/specifications/item/gm-level-1-sound-set

"""





from IPython.display import Audio
from pretty_midi import PrettyMIDI

sf2_path = '/usr/share/sounds/sf2/FluidR3_GM.sf2'  # path to sound font file
midi_file = '/home/ckirst/Media/Music/AImedia/MLMusic/Data/groove/drummer1/eval_session/1_funk-groove1_138_beat_4-4.mid'

music = PrettyMIDI(midi_file=midi_file)
waveform = music.fluidsynth(sf2_path=sf2_path)

import scipy.io.wavfile as siow
siow.write('test.wav', 44100, waveform)
import os
os.system('vlc test.wav')



Audio(waveform, rate=44100)



#%%

## some consstant for our audio file 

rate = 44100 #44.1 khz
duration =5 # in sec

# this will give us sin with the righ amplitude to use with wav files
normedsin = lambda f,t : 2**13*sin(2*pi*f*t)

time = np.linspace(0,duration, num=rate*duration)



#%%


import simpleaudio.functionchecks as fc

fc.LeftRightCheck.run()





#%% Ceck the midi
example = os.path.join(directory_midi, 'drummer1/session1');
example = os.path.join(example, os.listdir(example)[1])
print(example)

#%% plot the mid as scores 

m = m21.converter.parse(example)
m.measures(1,10).show()

#%%

m.measures(1,10).plot()

#%% play midi


from IPython.display import Audio

sf2_path = '/usr/share/sounds/sf2/FluidR3_GM.sf2'  # path to sound font file
midi_file = '/home/ckirst/Media/Music/AImedia/MLMusic/Data/groove/drummer1/eval_session/1_funk-groove1_138_beat_4-4.mid'

music = pm.PrettyMIDI(midi_file=midi_file)
waveform = music.fluidsynth(sf2_path=sf2_path)
Audio(waveform, rate=44100)



mid = pm.PrettyMIDI(midi_file=exmaple)
inst = mid.instruments[0]




