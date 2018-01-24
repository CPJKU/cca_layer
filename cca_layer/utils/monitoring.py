
from __future__ import print_function

from plotting import BColors

import lasagne
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def print_architecture(net):
    """ Print network architecture """

    col = BColors()
    print('\n')
    print(col.print_colored('Net-Architecture:', BColors.UNDERLINE))

    layers = lasagne.layers.helper.get_all_layers(net)
    max_len = np.max([len(l.__class__.__name__) for l in layers]) + 2
    for l in layers:
        class_name = l.__class__.__name__

        if isinstance(l, lasagne.layers.DropoutLayer):
            class_name += "(%.2f)" % l.p

        class_name = class_name.ljust(max_len)

        if isinstance(l, lasagne.layers.InputLayer):
            class_name = col.print_colored(class_name, BColors.OKBLUE)

        if isinstance(l, lasagne.layers.MergeLayer):
            class_name = col.print_colored(class_name, BColors.WARNING)

        print(class_name, l.output_shape)

