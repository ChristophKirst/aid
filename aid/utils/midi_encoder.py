#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Drummer Project

Project to create an AI drummer that performs with Pianist Jenny Q Chai

Convert drum midi to sequence of tokens
"""

import io
import numpy as np
import pandas as pd
import pretty_midi as pm

from collections import OrderedDict as odict

### Encoding 

MIDI_INFO_RAW = """pitch	Roland	MIDI	Simplified
36	Kick	Bass Drum 1	Bass (36)
38	Snare (Head)	Acoustic Snare	Snare (38)
40	Snare (Rim)	Electric Snare	Snare (38)
37	Snare X-Stick	Side Stick	Snare (38)
48	Tom 1	Hi-Mid Tom	High Tom (50)
50	Tom 1 (Rim)	High Tom	High Tom (50)
45	Tom 2	Low Tom	Low-Mid Tom (47)
47	Tom 2 (Rim)	Low-Mid Tom	Low-Mid Tom (47)	
43	Tom 3 (Head)	High Floor Tom	High Floor Tom (43)
58	Tom 3 (Rim)	Vibraslap	High Floor Tom (43)
46	HH Open (Bow)	Open Hi-Hat	Open Hi-Hat (46)
26	HH Open (Edge)	N/A	Open Hi-Hat (46)
42	HH Closed (Bow)	Closed Hi-Hat	Closed Hi-Hat (42)
22	HH Closed (Edge)	N/A	Closed Hi-Hat (42)
44	HH Pedal	Pedal Hi-Hat	Closed Hi-Hat (42)
49	Crash 1 (Bow)	Crash Cymbal 1	Crash Cymbal (49)
55	Crash 1 (Edge)	Splash Cymbal	Crash Cymbal (49)
57	Crash 2 (Bow)	Crash Cymbal 2	Crash Cymbal (49)
52	Crash 2 (Edge)	Chinese Cymbal	Crash Cymbal (49)
51	Ride (Bow)	Ride Cymbal 1	Ride Cymbal (51)
59	Ride (Edge)	Ride Cymbal 2	Ride Cymbal (51)
53	Ride (Bell)	Ride Bell	Ride Cymbal (51)	"""

MIDI_INFO = pd.read_csv(io.StringIO(MIDI_INFO_RAW), sep=r"\t", engine='python')

MIDI_PITCH_TO_INDEX   = { p : i for i,p in enumerate(MIDI_INFO['pitch'])}
MIDI_PITCH_FROM_INDEX = { i : p for i,p in enumerate(MIDI_INFO['pitch'])}

MIDI_PITCH_SIZE = max(MIDI_PITCH_TO_INDEX.values()) + 1;

ENCODE_EVENT_SIZES = odict(note_on    = MIDI_PITCH_SIZE, 
                           note_off   = MIDI_PITCH_SIZE,
                           velocity   = 32,
                           time_shift = 100,
                           control    = 0,
                           delimeter  = 2
                          );

ENCODE_EVENTS  = [k for k,v in ENCODE_EVENT_SIZES.items() if v > 0]
ENCODE_SIZE = sum(ENCODE_EVENT_SIZES.values())

ENCODE_START_INDEX = { k : v for k,v in zip(ENCODE_EVENT_SIZES.keys(), np.hstack([[0], np.cumsum(list(ENCODE_EVENT_SIZES.values()))[:-1]])) if ENCODE_EVENT_SIZES[k] > 0}
ENCODE_END_INDEX   = { k : v for k,v in zip(ENCODE_EVENT_SIZES.keys(), np.cumsum(list(ENCODE_EVENT_SIZES.values()))) if ENCODE_EVENT_SIZES[k] > 0}

ENCODE_TOKEN_END = ENCODE_START_INDEX['delimeter'];
ENCODE_TOKEN_PAD = ENCODE_TOKEN_END + 1;

CODE_TO_EVENT = dict();
for event in ENCODE_EVENTS:
  CODE_TO_EVENT.update( { i : (event, i - ENCODE_START_INDEX[event]) for i in range(ENCODE_START_INDEX[event], ENCODE_END_INDEX[event])})


ENCODE_TIME_RESOLUTION = 100;


class Action:
    """Class representing on and off actions of notes or control changes obtained from midi.""" 
    def __init__(self, atype, time, value, velocity = None, number = None):
        self.type     = atype
        self.time     = time
        self.value    = value
        self.velocity = velocity
        self.number   = number

    def __repr__(self):
        s =  'Action[%r, %s, %r' % (self.time, self.type, self.value)
        if self.velocity:
            s += ',%r' % self.velocity;
        if self.number:
            s += ',%r' % self.number;
        s += ']';
        return s;

class Event:
    """Event class representing events of the encoded midi sequence.""" 
    def __init__(self, etype = None, value = None, number = None):
        self.type  = etype
        self.value = value
        self.number = number

    def to_code(self):
        return ENCODE_START_INDEX[self.type] + self.value

    @staticmethod
    def from_code(code):
        try:
            event = CODE_TO_EVENT[int(code)];
        except:
            print('Could not decode %r' % code);
            
        return Event(*event)
        
    def __repr__(self):
        s = "Event[%s, %r" % (self.type, self.value)
        if self.number:
            s+= ', %r' % self.number;
        s += ']';
        return s;


def encode_midi(midi, encode_controls = False, convert_sustain = None, return_events = False):
    """Convert midi file to encoded sequence of integers for machine learning."""
    
    time_shift_max  = ENCODE_EVENT_SIZES['time_shift']
    time_resolution = ENCODE_TIME_RESOLUTION;
    
    velocity_bins = ENCODE_EVENT_SIZES['velocity'];
    encode_velocity = velocity_bins > 0;
    if encode_velocity:
        velocity_discretization = 128 / velocity_bins;
        
    # control_bins = ENCODE_EVENTS['control'];
    # if encode_controls:
    #     n_controls = len(encode_controls);
    #     control_bins = control_bins // n_controls;
    # encode_controls = encode_controls and control_bins > 0;
    # if encode_controls:
    #     control_discretization = 128 / control_bins;

    if isinstance(midi, str):
      midi = pm.PrettyMIDI(midi_file=midi);
    
    #convert to actions
    actions = [];
    for instrument in midi.instruments:
        notes = instrument.notes;
        
        # adapt pedal/sustained notes duration
        if convert_sustain:
            #sustain_number  = handle_sustain['number'];
            #sustain_pitches = handle_sustain['pitches'];
            pass
        
        instrument_actions = sum([[Action('note_on',  note.start, note.pitch, note.velocity),
                                   Action('note_off', note.end,   note.pitch)] for note in notes], []);
            
        if encode_controls:
            pass
         
        actions += instrument_actions;
    
    #time sorting
    actions.sort(key=lambda x: x.time);
    
    #velocities
    for action in actions:
        if action.velocity is not None:
            if encode_velocity:
                action.velocity = int(np.floor(action.velocity / velocity_discretization));
            else:
                action.velociy = None;
                
    #control values
    #TODO
    
    #encoding
    events = [];
    time = 0
    velocity = 0
    for action in actions:
      # time events (TODO: account for resluiton errors )
      time_shift = int(round((action.time - time) * time_resolution))
      while time_shift >= time_shift_max:
         events.append(Event('time_shift', value=time_shift_max-1));
         time_shift -= time_shift_max
         time += time_shift_max / time_resolution
      if time_shift > 0:
         events.append(Event('time_shift', value=time_shift-1))
         time += time_shift / time_resolution;
      #time = action.time * resolution; #discretized time 
      
      # velocity events
      if action.velocity is not None and velocity != action.velocity:
          events.append(Event('velocity', value=action.velocity));
          velocity = action.velocity;
      
      # note/control events
      if action.type in ['note_on', 'note_off']:
        events.append(Event(action.type, value=MIDI_PITCH_TO_INDEX[action.value]));
      else:
        events.append(Event(action.type, value=action.value)); 
    
    #convert to codes
    if return_events:
        return events;
    else: 
        codes = [e.to_code() for e in events];
        return codes;


def decode_midi(codes, filename = None, instrument = None):
    """Decode code to midi sequence."""
    
    # conversions
    time_resolution = ENCODE_TIME_RESOLUTION;
    
    velocity_bins = ENCODE_EVENT_SIZES['velocity'];
    decode_velocity = velocity_bins > 0;
    if decode_velocity:
        velocity_discretization = 128 / velocity_bins;
        
    # decoding
    events = [Event.from_code(code) for code in codes]
    #print(events)
    
    time = 0
    velocity = 0
    actions = []
    for event in events:
        #print(time)
        if   event.type == 'time_shift':
            time += ((event.value+1) / time_resolution)
        elif event.type == 'velocity':
            velocity = min(int(event.value * velocity_discretization), 127)
        elif event.type == 'control':
            actions.append(Action(event.type, time, event.value, number=event.number))
        elif event.type == 'delimeter':
            if event.to_code() == ENCODE_TOKEN_END:
                break;      
        else:
            actions.append(Action(event.type, time, MIDI_PITCH_FROM_INDEX[event.value], velocity=velocity))
    #print(actions)

    notes = [];
    controls = [];
    on_notes = dict();
    for action in actions:
        #print(action)
        if action.type == 'note_on':
            on_notes[action.value] = action
        elif action.type == 'note_off':
            #print(';off')
            try:
                on = on_notes[action.value]
                #print('on')
                off = action
                #print(on.time, off.time)
                if off.time - on.time == 0:
                    continue
                notes.append(pm.Note(on.velocity, action.value, on.time, off.time))
            except:
                print('decode_midi: removed pitch: {}'.format(action.value))
        elif action.type == 'control':
             controls.append(pm.ControlChange(action.number, action.value, action.time))
        else:
            print('decode_midi: unknown action %r' % action);
    #print(notes)       

    notes.sort(key=lambda x:x.start)

    #midi
    if instrument is None:
        instrument = pm.Instrument(0, True, "aid")
    instrument.notes = notes
    instrument.control_changes = controls; 
    mid = pm.PrettyMIDI()
    mid.instruments.append(instrument)
    
    if filename is not None:
        mid.write(filename)
    
    return mid



def _test():
    
    import aid.utils.midi_encoder as encoder
    
    midi = '/home/ckirst/Media/Music/AImedia/MLMusic/Data/groove/drummer1/session1/1_funk_80_beat_4-4.mid'
    midi = encoder.pm.PrettyMIDI(midi_file=midi);
    code = encoder.encode_midi(midi)
    
    decode = encoder.decode_midi(code)
    
    n_notes = 10;
    midi.instruments[0].notes[:n_notes]
    decode.instruments[0].notes[:n_notes]

    import aid.utils.utils as utils
    utils.play_midi(decode)
