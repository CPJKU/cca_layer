#!/usr/bin/env python
# author: Matthias Dorfer

from __future__ import print_function

import os
try:
    import cPickle as pickle
except ImportError:
    import pickle
import argparse

from utils.data import load_iapr, load_audio_score
from config.settings import EXP_ROOT

# init color printer
from utils.plotting import BColors
col = BColors()


def select_model(model_path):
    """ select model and train function """

    model_str = os.path.basename(model_path)
    model_str = model_str.split('.py')[0]
    exec('from models import ' + model_str + ' as model')

    from utils.train_utils import fit

    model.EXP_NAME = model_str
    return model, fit


def select_data(data_name, seed=23):
    """ select train data """

    if str(data_name) == "iapr":
        data = load_iapr(seed=seed)
    elif str(data_name) == "audio_score":
        data = load_audio_score(seed=seed)
    else:
        pass

    return data


if __name__ == '__main__':
    """ main """

    # add argument parser
    parser = argparse.ArgumentParser(description='Train multi-modality model.')
    parser.add_argument('--model', help='select model to train.')
    parser.add_argument('--data', help='select data for training.')
    parser.add_argument('--seed', help='query direction.', type=int, default=23)
    parser.add_argument('--no_dump', help='do not dump model file.', action='store_true')
    parser.add_argument('--tag', help='add tag to grid search dump file.', type=str, default=None)
    parser.add_argument('--show_architecture', help='print model architecture.', action='store_true')
    args = parser.parse_args()

    # create parameter folder
    if not os.path.exists(EXP_ROOT):
        os.makedirs(EXP_ROOT)

    # select model
    model, fit = select_model(args.model)

    # select data
    print("\nLoading data...")
    data = select_data(args.data, args.seed)

    # set path to log and model dump
    out_path = os.path.join(os.path.join(EXP_ROOT), model.EXP_NAME)
    dump_file = 'params.pkl' if args.tag is None else 'params_%s.pkl' % args.tag
    dump_file = os.path.join(out_path, dump_file)
    log_file = 'results.pkl' if args.tag is None else 'results_%s.pkl' % args.tag
    log_file = os.path.join(out_path, log_file)

    print("\nBuilding network...")
    layers = model.build_model(show_model=args.show_architecture)

    # do not dump model
    dump_file = None if args.no_dump else dump_file

    # train model
    # -----------
    train_batch_iter = model.train_batch_iterator(model.BATCH_SIZE)
    valid_batch_iter = model.valid_batch_iterator()
    layers, va_loss = fit(layers, data, model.objectives,
                          train_batch_iter=train_batch_iter, valid_batch_iter=valid_batch_iter,
                          num_epochs=model.MAX_EPOCHS, patience=model.PATIENCE,
                          learn_rate=model.INI_LEARNING_RATE, update_learning_rate=model.update_learning_rate,
                          compute_updates=model.compute_updates, l_2=model.L2, l_1=model.L1,
                          exp_name=model.EXP_NAME, out_path=out_path, dump_file=dump_file, log_file=log_file,
                          refinement_steps=model.REFINEMENT_STEPS, lr_multiplier=model.LR_MULTIPLIER, do_raise=True)
