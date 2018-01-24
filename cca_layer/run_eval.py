
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

from utils.batch_iterators import batch_compute2
from utils.train_utils import eval_retrieval


def flip_variables(v1, v2):
    """ flip variables """
    tmp = v1.copy()
    v1 = v2
    v2 = tmp
    return v1, v2


if __name__ == '__main__':
    """ main """

    # add argument parser
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument('--model', help='model parameters for evaluation.', default="flickr30")
    parser.add_argument('--data', help='select evaluation data.', type=str, default="flickr30")
    parser.add_argument('--n_test', help='number of test samples used for projection.', type=int, default=None)
    parser.add_argument('--V2_to_V1', help='change query direction.', action='store_true')
    parser.add_argument('--estimate_UV', help='load re-estimated U and V.', action='store_true')
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
    if args.estimate_UV:
        model.EXP_NAME += "_est_UV"
    out_path = os.path.join(os.path.join(EXP_ROOT), model.EXP_NAME)
    dump_file = 'params.pkl' if args.tag is None else 'params_%s.pkl' % args.tag
    dump_file = os.path.join(out_path, dump_file)

    print("\n")
    print("Loading model parameters from:", dump_file)
    with open(dump_file, 'r') as fp:
         params = pickle.load(fp)
    lasagne.layers.set_all_param_values(layers, params)

    print("\nLoading data...")
    data = select_data(args.data, seed=args.seed)

    print("\nCompiling prediction functions...")
    l_view1, l_view2, l_v1latent, l_v2latent = layers

    input_1 = input_2 = [l_view1.input_var, l_view2.input_var]

    compute_v1_latent = theano.function(inputs=input_1,
                                        outputs=lasagne.layers.get_output(l_v1latent, deterministic=True))
    compute_v2_latent = theano.function(inputs=input_2,
                                        outputs=lasagne.layers.get_output(l_v2latent, deterministic=True))

    print("Evaluating on test set...")

    # compute output on test set
    eval_set = 'test'
    n_test = args.n_test if args.n_test is not None else data[eval_set].shape[0]
    X1, X2 = data[eval_set][0:n_test]

    print("Computing embedding ...")
    lv1_latent = batch_compute2(X1, X2, compute_v1_latent, np.min([100, n_test]), prepare=model.prepare)
    lv2_latent = batch_compute2(X1, X2, compute_v2_latent, np.min([100, n_test]), prepare=model.prepare)

    if args.V2_to_V1:
        lv1_latent, lv2_latent = flip_variables(lv1_latent, lv2_latent)

    # reset n_test
    n_test = lv1_latent.shape[0]

    # evaluate retrieval result
    print("lv1_latent.shape:", lv1_latent.shape)
    print("lv2_latent.shape:", lv2_latent.shape)
    mean_rank_te, med_rank_te, dist_te, hit_rates, mrr = eval_retrieval(lv1_latent, lv2_latent)

    # report hit rates
    recall_at_k = dict()
    print("\nHit Rates:")
    for key in np.sort(hit_rates.keys()):
        recall_at_k[key] = float(100 * hit_rates[key]) / n_test
        pk = recall_at_k[key] / key
        print("Top %02d: %.3f (%d) %.3f" % (key, recall_at_k[key], hit_rates[key], pk))

    print("\n")
    print("Median Rank: %.2f (%d)" % (med_rank_te, lv2_latent.shape[0]))
    print("Mean Rank  : %.2f (%d)" % (mean_rank_te, lv2_latent.shape[0]))
    print("Mean Dist  : %.5f " % dist_te)
    print("MRR        : %.3f " % mrr)

    from scipy.spatial.distance import cdist
    dists = np.diag(cdist(lv1_latent, lv2_latent, metric="cosine"))
    print("Min Dist   : %.5f " % np.min(dists))
    print("Max Dist   : %.5f " % np.max(dists))
    print("Med Dist   : %.5f " % np.median(dists))
