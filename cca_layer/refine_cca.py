
from __future__ import print_function

import os
try:
    import cPickle as pickle
except ImportError:
    import pickle
import argparse
import lasagne
import theano
import numpy as np

from config.settings import EXP_ROOT
from run_train import select_model, select_data

from utils.batch_iterators import batch_compute1

from utils.cca import CCA
from cca_layer.models.lasagne_extensions.layers.cca import CCALayer


if __name__ == '__main__':
    """ main """

    # add argument parser
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument('--model', help='model parameters for evaluation.', default="iapr_vgg")
    parser.add_argument('--data', help='select evaluation data.', type=str, default="iapr_vgg")
    parser.add_argument('--n_train', help='number of train samples used for projection.', type=int, default=1000)
    parser.add_argument('--seed', help='query direction.', type=int, default=23)
    parser.add_argument('--tag', help='add tag to grid search dump file.', type=str, default=None)
    args = parser.parse_args()

    # select model
    model, _ = select_model(args.model)

    if not hasattr(model, 'prepare'):
        model.prepare = None

    print("Building network %s ..." % model.EXP_NAME)
    layers = model.build_model(show_model=False)

    # load model parameters
    out_path = os.path.join(os.path.join(EXP_ROOT), model.EXP_NAME)
    dump_file_name = 'params.pkl' if args.tag is None else 'params_%s.pkl' % args.tag
    dump_file = os.path.join(out_path, dump_file_name)
    print("\n")
    print("Loading model parameters from:", dump_file)
    with open(dump_file, 'r') as fp:
         params = pickle.load(fp)
    lasagne.layers.set_all_param_values(layers, params)

    # reset model parameter file
    model.EXP_NAME += "_est_UV"
    out_path = os.path.join(os.path.join(EXP_ROOT), model.EXP_NAME)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    dump_file = os.path.join(out_path, dump_file_name)

    # select data
    print("\nLoading data...")
    data = select_data(args.data, seed=args.seed)

    print("\nCompiling prediction functions...")
    l_view1, l_view2, l_v1latent, l_v2latent = layers

    # get network input variables
    input_1, input_2 = [l_view1.input_var], [l_view2.input_var]

    # get cca layer input
    for l in lasagne.layers.helper.get_all_layers(l_v1latent):
        if isinstance(l, CCALayer):
            print("CCALayer found!")
            cca_layer = l
            l_v1_cca = cca_layer.input_layers[0]
            l_v2_cca = cca_layer.input_layers[1]
            break

    compute_v1_latent = theano.function(inputs=input_1,
                                        outputs=lasagne.layers.get_output(l_v1_cca, deterministic=True))
    compute_v2_latent = theano.function(inputs=input_2,
                                        outputs=lasagne.layers.get_output(l_v2_cca, deterministic=True))

    print("Computing train output...")
    X1, X2 = data['train'][0:args.n_train]
    lv1_tr = batch_compute1(X1, compute_v1_latent, np.min([10, args.n_train]), prepare=model.prepare)
    lv2_tr = batch_compute1(X2, compute_v2_latent, np.min([10, args.n_train]))

    print("Fitting CCA model...")
    cca = CCA(method='svd')
    cca.fit(lv1_tr, lv2_tr, verbose=True)

    # reset layer weights
    cca_layer.mean1.set_value(cca.m1.astype(np.float32))
    cca_layer.mean2.set_value(cca.m2.astype(np.float32))
    cca_layer.U.set_value(cca.U.astype(np.float32))
    cca_layer.V.set_value(cca.V.astype(np.float32))

    print("Dumping refined model...")
    with open(dump_file, 'w') as fp:
        pickle.dump(lasagne.layers.get_all_param_values(layers), fp, protocol=-1)
