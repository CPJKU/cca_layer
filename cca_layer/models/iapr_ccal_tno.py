#!/usr/bin/env python

import lasagne
from lasagne.layers import batch_norm, DenseLayer, DropoutLayer, SliceLayer
from lasagne.nonlinearities import elu, identity

from cca_layer.utils.monitoring import print_architecture
from cca_layer.models.lasagne_extensions.layers.cca import CCALayer, LearnedCCALayer, LengthNormLayer


INI_LEARNING_RATE = 0.001
BATCH_SIZE = 1000
MOMENTUM = 0.9
MAX_EPOCHS = 1000
PATIENCE = 15
N_DECAY = 50
REFINEMENT_STEPS = 3
LR_MULTIPLIER = 0.1

INPUT_SHAPE_1 = [4096]
INPUT_SHAPE_2 = [2048]

N_HIDDEN_IMG = 1024
N_HIDDEN_TXT = 1024

N_LAYERS_IMG = 0
N_LAYERS_TXT = 2

DIM_LATENT = 128

L1 = None
L2 = 0.0001
GRAD_NORM = None
DROPOUT = 0.0

r1 = r2 = 1e-3
rT = 1e-3

nonlin = elu
init = lasagne.init.HeUniform

ALPHA = 1.0
WEIGHT_TNO = 1.0
USE_CCAL = True


def dense_bn(net_in, num_units, nonlinearity):
    """ Compile convolution layer with batch norm """
    net = DenseLayer(net_in, num_units=num_units, W=init(),
                     nonlinearity=nonlinearity, name='dense_bn')
    return batch_norm(net)


def get_build_model(weight_tno, alpha, dim_latent, use_ccal):
    """ Get model function """

    def model(show_model):
        """ Compile net architecture """

        # --- input layers ---
        l_view1 = lasagne.layers.InputLayer(shape=(None, INPUT_SHAPE_1[0]))
        l_view2 = lasagne.layers.InputLayer(shape=(None, INPUT_SHAPE_2[0]))

        net1 = l_view1
        net2 = l_view2

        # --- feed forward part view 1 ---
        for _ in xrange(N_LAYERS_IMG):
            net1 = dense_bn(net1, num_units=N_HIDDEN_IMG, nonlinearity=nonlin)
            net1 = DropoutLayer(net1, p=DROPOUT)

        l_v1latent = DenseLayer(net1, num_units=dim_latent, nonlinearity=identity, W=init())

        # --- feed forward part view 2 ---
        for _ in xrange(N_LAYERS_TXT):
            net2 = dense_bn(net2, num_units=N_HIDDEN_TXT, nonlinearity=nonlin)
            net2 = DropoutLayer(net2, p=DROPOUT)

        l_v2latent = DenseLayer(net2, num_units=dim_latent, nonlinearity=identity, W=init())

        # --- multi modality part ---

        # merge modalities by cca projection or learned embedding layer
        if use_ccal:
            net = CCALayer([l_v1latent, l_v2latent], r1, r2, rT, alpha=alpha, wl=weight_tno)
        else:
            net = LearnedCCALayer([l_v1latent, l_v2latent], U=init(), V=init(), alpha=alpha)

        # split modalities again
        l_v1 = SliceLayer(net, slice(0, dim_latent), axis=1)
        l_v2 = SliceLayer(net, slice(dim_latent, 2 * dim_latent), axis=1)

        # normalize (per row) output to length 1.0
        l_v1 = LengthNormLayer(l_v1)
        l_v2 = LengthNormLayer(l_v2)

        # --- print architectures ---
        if show_model:
            print_architecture(l_v1)
            print_architecture(l_v2)

        return l_view1, l_view2, l_v1, l_v2

    return model


build_model = get_build_model(weight_tno=WEIGHT_TNO, alpha=ALPHA, dim_latent=DIM_LATENT, use_ccal=USE_CCAL)


def objectives():
    """ Compile objectives """
    from objectives import get_contrastive_cos_loss
    return get_contrastive_cos_loss(1.0 - WEIGHT_TNO, 0.0)


def update_learning_rate(lr, epoch=None):
    """ Update learning rate """
    return lr


def valid_batch_iterator(batch_size=100):
    """ Compile batch iterator """
    from cca_layer.utils.batch_iterators import MultiviewPoolIteratorUnsupervised
    batch_iterator = MultiviewPoolIteratorUnsupervised(batch_size=batch_size, prepare=None)
    return batch_iterator


def train_batch_iterator(batch_size=BATCH_SIZE):
    """ Compile batch iterator """
    from cca_layer.utils.batch_iterators import MultiviewPoolIteratorUnsupervised
    batch_iterator = MultiviewPoolIteratorUnsupervised(batch_size=batch_size, prepare=None, k_samples=17000)
    return batch_iterator


def compute_updates(all_grads, all_params, learning_rate):
    """
    Compute gradients for updates
    """
    return lasagne.updates.adam(all_grads, all_params, learning_rate=learning_rate)
