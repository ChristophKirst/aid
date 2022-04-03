#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities

Utility functions for the transformer model, including visualization.
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__copyright__ = 'Copyright © 2022 by Christoph Kirst'

import matplotlib.pyplot as plt
import seaborn

from aid.dataset.midi_encoder import code_to_label


def plot_attention(model, src, layers = None, heads = None, batch = 0, fig = None, label = None):
    """Visualize attention for the src inputs."""
    

    model.eval();
    y = model(src);
    if label is None:
        label = code_to_label(src[batch].data.numpy())
    
    if layers is None:
        layers = list(range(len(model.encoder.layers)));
    if heads is None:
        heads = list(range(model.encoder.layers[0].attention.n_heads));
   
    n_layers = len(layers);
    n_heads = len(heads);
   
    if fig is None:
        fig = plt.gcf();
    
    axs = fig.subplots(n_layers, n_heads);
     
    def draw(data, x, y, ax):
       seaborn.heatmap(data, square=True, vmin=0.0, vmax=1.0, 
                       xticklabels=x, yticklabels=y, cbar=False, ax=ax)
    
    a = -1;
    for l, layer in enumerate(layers):
        for h, head in enumerate(heads):
            a += 1;
            attn = model.encoder.layers[layer].attention.attention[batch, head].data
            draw(attn, label, label if head == heads[0] else [], ax=axs[a])
            if head == heads[0]:
                axs[a].set_title('Layer %d - head %d' % (layer, head));
            else:
                axs[a].set_title('head %d' % (head));
    plt.tight_layout()
    plt.show()
    
    return axs;

