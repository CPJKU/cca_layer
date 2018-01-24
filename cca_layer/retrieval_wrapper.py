from __future__ import print_function

import pickle
import numpy as np

import theano
import lasagne


class RetrievalWrapper(object):
    """ Wrapper for cross modality retrieval networks """

    def __init__(self, model, param_file, prepare_view_1=None, prepare_view_2=None):
        """ Constructor """

        self.prepare_view_1 = prepare_view_1
        self.prepare_view_2 = prepare_view_2

        self.code_dim = model.DIM_LATENT

        print("Building network ...")
        layers = model.build_model(show_model=False)

        print("Loading model parameters from:", param_file)
        with open(param_file, 'r') as fp:
            params = pickle.load(fp)
        lasagne.layers.set_all_param_values(layers, params)

        print("Compiling prediction functions ...")
        l_view1, l_view2, l_v1latent, l_v2latent = layers
        self.compute_v1_latent = theano.function(inputs=[l_view1.input_var, l_view2.input_var],
                                                 outputs=lasagne.layers.get_output(l_v1latent,
                                                                                   deterministic=True))
        self.compute_v2_latent = theano.function(inputs=[l_view1.input_var, l_view2.input_var],
                                                 outputs=lasagne.layers.get_output(l_v2latent,
                                                                                   deterministic=True))

        # dummy imputs for respective second view
        self.dummy_in_v1 = np.zeros(([1] + list(l_view1.output_shape[1:])), dtype=np.float32)
        self.dummy_in_v2 = np.zeros(([1] + list(l_view2.output_shape[1:])), dtype=np.float32)

    def compute_view_1(self, X):
        """ compute network output of view 1 """

        X = X.copy()

        # normalize data
        if self.prepare_view_1 is not None:
            X = self.prepare_view_1(X)

        return self.compute_v1_latent(X, self.dummy_in_v2)

    def compute_view_2(self, Z):
        """ compute network output of view 2 """

        Z = Z.copy()

        # normalize data
        if self.prepare_view_2 is not None:
            Z = self.prepare_view_2(Z)

        return self.compute_v2_latent(self.dummy_in_v1, Z)


if __name__ == '__main__':
    """ main """
    pass
