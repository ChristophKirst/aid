#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Drummer Project

Project to create an AI drummer that performs with Pianist Jenny Q Chai
"""

#%% Train music transformer for AI drummer / test speed and quality

from aid.model.training import train

train(epochs=1)


#%%

from aid.model.generation import generate

generate(primer=1)
