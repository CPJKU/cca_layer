#!/usr/bin/env python

"""Train neural networks with two inputs (multi-modal inputs, e.g. score alignment)."""

from __future__ import print_function


import os
import copy
import time
import itertools
import numpy as np
import cPickle as pickle

import lasagne
import theano
import theano.tensor as T

from plotting import BColors
from batch_iterators import threaded_generator_from_iterator
from cca import CCA
from cca_layer.models.lasagne_extensions.layers.cca import CCALayer

# init color printer
col = BColors()


def eval_retrieval(lv1_cca, lv2_cca):
    """ Compute retrieval eval measures """

    # get number of samples in lists
    n_v1 = lv1_cca.shape[0]
    n_v2 = lv2_cca.shape[0]

    k = n_v2 / n_v1 if n_v2 > n_v1 else 1
    h = n_v1 / n_v2 if n_v1 > n_v2 else 1

    # compute pairwise distances
    from scipy.spatial.distance import cdist
    dists = cdist(lv1_cca, lv2_cca, metric="cosine")

    # score results
    ranks = []
    aps = []
    hit_rates = {1: 0, 5: 0, 10: 0, 25: 0}
    for i in xrange(n_v1):

        # fix i for multiple indices
        i_fixed = np.floor_divide(i, h)

        # sort indices by distance
        sorted_idx = np.argsort(dists[i])

        for key in hit_rates:

            # get top k results
            top_k_results = sorted_idx[0:key]

            # pre-processing if multible results are coorect
            top_k_results = np.floor_divide(top_k_results, k)

            # check if correct item is in the top k
            if i_fixed in top_k_results:
                hit_rates[key] += 1

        # compute retrieval rank of correct item (+1 as rank starts at 1)
        fixed_sorted_idx = np.floor_divide(sorted_idx, k)
        rank = np.min(np.nonzero(fixed_sorted_idx == i_fixed)[0]) + 1

        # keep ranks
        ranks.append(rank)

        # keep average precision
        aps.append(1.0 / rank)

    # compute some stats
    mean_rank = np.mean(ranks)
    median_rank = np.median(ranks)
    mean_dist = np.diag(dists).mean()
    mrr = np.mean(aps)

    return mean_rank, median_rank, mean_dist, hit_rates, mrr


def create_iter_functions(layers, objectives, compute_updates, learning_rate, l_2, l_1, init_cca=False):
    """ Create functions for training, validation and testing to iterate one epoch. """

    # extract concerned layers
    if len(layers) == 5:
        l_view1, l_view2, l_v1latent, l_v2latent, l_mask = layers
        input_vars = [l_view1.input_var, l_view2.input_var, l_mask.input_var]
    else:
        l_view1, l_view2, l_v1latent, l_v2latent = layers
        input_vars = [l_view1.input_var, l_view2.input_var]

    # extract concerned costs
    latent_costs = objectives()

    # get hidden representations
    tr_v1latent_output = lasagne.layers.get_output(l_v1latent, deterministic=False)
    tr_v2latent_output = lasagne.layers.get_output(l_v2latent, deterministic=False)
    va_v1latent_output = lasagne.layers.get_output(l_v1latent, deterministic=True)
    va_v2latent_output = lasagne.layers.get_output(l_v2latent, deterministic=True)

    # get costs
    tr_out = latent_costs(tr_v1latent_output, tr_v2latent_output)
    va_out = latent_costs(va_v1latent_output, va_v2latent_output)

    if tr_out.__class__ is tuple:
        tr_out = list(tr_out)
        va_out = list(va_out)
    else:
        tr_out = [tr_out]
        va_out = [va_out]

    # collect all parameters of net
    all_params = lasagne.layers.get_all_params([l_v1latent, l_v2latent],
                                               trainable=True)

    # add losses of special layers
    for l in lasagne.layers.helper.get_all_layers(l_v1latent):

        if hasattr(l, 'get_loss'):
            print("Adding Loss of", l.__class__.__name__)
            loss = l.get_loss()
            tr_out[0] += loss
            tr_out += [l.get_corr()]

        if l.name == "norm_reg_rnn":
            print("Adding Loss of", l.__class__.__name__)
            H = lasagne.layers.get_output(l, deterministic=False)
            H_l2 = T.sqrt(T.sum(H ** 2, axis=-1))
            norm_diffs = (H_l2[:, 1:] - H_l2[:, :-1]) ** 2
            norm_preserving_loss = T.mean(norm_diffs)

            beta = 1.0
            tr_out[0] += beta * norm_preserving_loss

    # add weight decay
    if l_2 is not None:
        tr_out[0] += l_2 * lasagne.regularization.apply_penalty(all_params, lasagne.regularization.l2)

    if l_1 is not None:
        tr_out[0] += l_1 * lasagne.regularization.apply_penalty(all_params, lasagne.regularization.l1)

    # compute updates with scaled gradients
    all_grads = lasagne.updates.get_or_compute_grads(tr_out[0], all_params)

    # compute updates
    updates = compute_updates(all_grads, all_params, learning_rate)

    # compile iter functions
    iter_train = theano.function(input_vars, tr_out, updates=updates)
    iter_valid = theano.function(input_vars, va_out)

    # compile compute output function
    compute_output = theano.function(input_vars, [va_v1latent_output, va_v2latent_output])

    # compile burn-in function to initialize CCALayer running averages
    if init_cca:
        init_cca = theano.function(input_vars, [tr_v1latent_output, tr_v2latent_output])

    compute_gradients = theano.function(input_vars, all_grads)

    # get cca layer input
    cca_layer = compute_v1_cca_in = compute_v2_cca_in = None
    for l in lasagne.layers.get_all_layers(l_v1latent):
        if isinstance(l, CCALayer):
            print("CCALayer found!")
            cca_layer = l
            l_v1_cca_in = cca_layer.input_layers[0]
            l_v2_cca_in = cca_layer.input_layers[1]
            break

    # compile compute latent space
    if cca_layer:
        compute_v1_cca_in = theano.function(inputs=[l_view1.input_var],
                                            outputs=lasagne.layers.get_output(l_v1_cca_in, deterministic=True))
        compute_v2_cca_in = theano.function(inputs=[l_view2.input_var],
                                            outputs=lasagne.layers.get_output(l_v2_cca_in, deterministic=True))

    return dict(train=iter_train, valid=iter_valid, test=iter_valid, compute_output=compute_output, init_cca=init_cca,
                compute_gradients=compute_gradients, all_params=all_params, updates=updates,
                cca_layer=cca_layer, compute_v1_cca_in=compute_v1_cca_in, compute_v2_cca_in=compute_v2_cca_in)


def pretrain(iter_funcs, dataset, train_batch_iter, epochs=3):
    """
    Run some epochs over the training data to initialize the CCLayer running
    averages to something meaningful (important for small alpha values)
    """
    if not iter_funcs['init_cca']:
        return
    print("Pretraining for %d epochs..." % epochs)
    for _ in range(epochs):
        iterator = train_batch_iter(dataset['train'])
        generator = threaded_generator_from_iterator(iterator)
        for X_b, Z_b in generator:
            iter_funcs['init_cca'](X_b, Z_b)


def train(iter_funcs, dataset, train_batch_iter, valid_batch_iter):
    """
    Train the model with `dataset` with mini-batch training.
    Each mini-batch has `batch_size` recordings.
    """
    import time
    import sys

    for epoch in itertools.count(1):

        # iterate train batches
        batch_train_evals = []
        batch_train_losses = []
        iterator = train_batch_iter(dataset['train'])
        generator = threaded_generator_from_iterator(iterator)

        batch_times = np.zeros(5, dtype=np.float32)
        start, after = time.time(), time.time()
        for i_batch, train_input in enumerate(generator):

            try:
                batch_res = iter_funcs['train'](*train_input)
            except:
                batch_res[0] = np.nan

            batch_train_losses.append(batch_res[0])
            if len(batch_res) > 1:
                batch_train_evals.append(batch_res[1])

            # compute timing
            batch_time = time.time() - after
            after = time.time()
            train_time = (after - start)

            # estimate updates per second (running avg)
            batch_times[0:4] = batch_times[1:5]
            batch_times[4] = batch_time
            ups = 1.0 / batch_times.mean()

            # report loss during training
            perc = 100 * (float(i_batch + 1) / train_batch_iter.n_batches)
            dec = int(perc // 4)
            progbar = "|" + dec * "#" + (25 - dec) * "-" + "|"
            vals = (perc, progbar, train_time, ups, np.mean(batch_train_losses))
            loss_str = " (%d%%) %s time: %.2fs, ups: %.2f, loss: %.5f" % vals
            print(col.print_colored(loss_str, col.WARNING), end="\r")
            sys.stdout.flush()

        # refine cca projection using entire train set
        if iter_funcs['cca_layer'] and not np.isnan(batch_res[0]):
            batch_iter_copy = copy.copy(valid_batch_iter)
            batch_iter_copy.epoch_counter = 0
            iterator = batch_iter_copy(dataset['train'])
            generator = threaded_generator_from_iterator(iterator)
            V1_tr, V2_tr = None, None
            for i_batch, train_input in enumerate(generator):
                v1_cca_in = iter_funcs['compute_v1_cca_in'](train_input[0])
                v2_cca_in = iter_funcs['compute_v2_cca_in'](train_input[1])
                V1_tr = v1_cca_in if V1_tr is None else np.vstack([V1_tr, v1_cca_in])
                V2_tr = v2_cca_in if V2_tr is None else np.vstack([V2_tr, v2_cca_in])

            cca = CCA(method='svd')
            cca.fit(V1_tr, V2_tr, verbose=False)

            # update layer parameters
            iter_funcs['cca_layer'].mean1.set_value(cca.m1.astype(np.float32))
            iter_funcs['cca_layer'].mean2.set_value(cca.m2.astype(np.float32))
            iter_funcs['cca_layer'].U.set_value(cca.U.astype(np.float32))
            iter_funcs['cca_layer'].V.set_value(cca.V.astype(np.float32))

        # compute network output on train set
        n_valid_cca = np.min([5000, dataset['valid'].shape[0]])
        lv1_cca, lv2_cca = None, None
        batch_iter_copy = copy.copy(train_batch_iter)
        batch_iter_copy.epoch_counter = 0
        iterator = batch_iter_copy(dataset['train'])
        generator = threaded_generator_from_iterator(iterator)
        for i_batch, train_input in enumerate(generator):

            if lv1_cca is None or lv2_cca.shape[0] < n_valid_cca:
                X_o, Z_o = iter_funcs['compute_output'](*train_input)
                lv1_cca = X_o if lv1_cca is None else np.vstack([lv1_cca, X_o])
                lv2_cca = Z_o if lv2_cca is None else np.vstack([lv2_cca, Z_o])

        # evaluate retrieval on train set
        mean_rank_tr, med_rank_tr, dist_tr, hit_rates, mrr_tr = eval_retrieval(lv1_cca, lv2_cca)
        mean_rank_tr = 1.0 - float(hit_rates[10]) / len(lv1_cca)

        print("\x1b[K", end="\r")
        print(' ')
        avg_train_loss = np.mean(batch_train_losses)
        if len(batch_train_evals) > 0:
            batch_train_evals = np.asarray(batch_train_evals).mean(axis=0)
        else:
            batch_train_evals = None

        # evaluate classification power of data set

        # iterate validation batches
        lv1_cca, lv2_cca = None, None
        batch_valid_losses = []
        iterator = valid_batch_iter(dataset['valid'])
        generator = threaded_generator_from_iterator(iterator)
        for train_input in generator:
            batch_res = iter_funcs['valid'](*train_input)
            batch_valid_losses.append(batch_res[0])

            # compute network output
            if lv1_cca is None or lv1_cca.shape[0] < n_valid_cca:
                X_o, Z_o = iter_funcs['compute_output'](*train_input)
                lv1_cca = X_o if lv1_cca is None else np.vstack([lv1_cca, X_o])
                lv2_cca = Z_o if lv2_cca is None else np.vstack([lv2_cca, Z_o])

        avg_valid_loss = np.mean(batch_valid_losses)

        # evaluate retrieval on validation set
        mean_rank_va, med_rank_va, dist_va, hit_rates, mrr_va = eval_retrieval(lv1_cca, lv2_cca)
        mean_rank_va = 1.0 - float(hit_rates[10]) / 1000

        # collect results
        yield {
            'number': epoch,
            'train_loss': avg_train_loss,
            'valid_loss': avg_valid_loss,
            'mean_cos_dist_tr': dist_tr,
            'mean_cos_dist_va': dist_va,
            'mean_rank_tr': mean_rank_tr,
            'mean_rank_va': mean_rank_va,
            'med_rank_tr': med_rank_tr,
            'med_rank_va': med_rank_va,
            'mrr_tr': mrr_tr,
            'mrr_va': mrr_va,
            'evals_tr': batch_train_evals,
        }


def fit(layers, data, objectives,
        train_batch_iter, valid_batch_iter,
        num_epochs=100, patience=20,
        learn_rate=0.01, update_learning_rate=None,
        l_2=None, l_1=None, compute_updates=None,
        exp_name='ff', out_path=None, dump_file=None,
        pretrain_epochs=0, refinement_steps=0, lr_multiplier=0.1, log_file=None,
        do_raise=False):
    """ Train model """

    # check if out_path exists
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    # get output layer
    l_out = layers[-1]

    # log model evolution
    if log_file is None:
        log_file = os.path.join(out_path, 'results.pkl')

    print("\n")
    print(col.print_colored("Running Test Case: " + exp_name, BColors.UNDERLINE))

    # adaptive learning rate
    learning_rate = theano.shared(np.float32(learn_rate))
    if update_learning_rate is None:
        def update_learning_rate(lr, e, li):
            return lr
    learning_rate.set_value(update_learning_rate(learn_rate))

    # initialize evaluation output
    pred_tr_err, pred_val_err = [], []
    dists_tr, dists_va = [], []
    ranks_tr, ranks_va = [], []
    mrrs_tr, mrrs_va = [], []
    evals_evol_tr = []

    print("Building model and compiling functions...")
    iter_funcs = create_iter_functions(layers, objectives, compute_updates, learning_rate, l_2, l_1, init_cca=pretrain_epochs > 0)

    print("Starting training...")
    now = time.time()
    try:
        # run pre-training if requested
        if pretrain_epochs:
            pretrain(iter_funcs, data, train_batch_iter, pretrain_epochs)

        # initialize early stopping
        last_improvement = 0
        best_model = lasagne.layers.get_all_param_values(layers)

        # iterate training epochs
        prev_tr_loss, prev_va_loss = 1e7, 1e7
        prev_tr_dist, prev_va_dist = 1e7, 1e7
        prev_med_rank_tr, prev_med_rank_va = 1e7, 1e7
        prev_mrr_tr, prev_mrr_va = 0.0, 0.0
        for epoch in train(iter_funcs, data, train_batch_iter, valid_batch_iter):

            # --- collect train output ---

            tr_loss, va_loss = epoch['train_loss'], epoch['valid_loss']
            tr_dist, va_dist = epoch['mean_cos_dist_tr'], epoch['mean_cos_dist_va']
            rank_tr, rank_va = epoch['mean_rank_tr'], epoch['mean_rank_va']
            mrr_tr, mrr_va = epoch['mrr_tr'], epoch['mrr_va']
            med_rank_tr, med_rank_va = epoch['med_rank_tr'], epoch['med_rank_va']
            evals_tr = epoch['evals_tr']

            # prepare early stopping
            improvement = mrr_va >= prev_mrr_va
            if improvement:
                last_improvement = 0
                best_epoch = epoch['number']
                best_model = lasagne.layers.get_all_param_values(layers)
                best_opt_state = [_u.get_value() for _u in iter_funcs['updates'].keys()]

                # dump net parameters during training
                if dump_file is not None:
                    with open(dump_file, 'w') as fp:
                        pickle.dump(best_model, fp, protocol=-1)

            last_improvement += 1

            print("Epoch {} of {} took {:.3f}s (patience: {})".format(
                epoch['number'], num_epochs, time.time() - now, patience - last_improvement + 1))
            now = time.time()

            # check for numerical instability
            if np.isnan(tr_loss):
                last_improvement = patience + 1

            # print train output
            txt_tr = 'costs_tr %.5f ' % tr_loss
            if tr_loss < prev_tr_loss:
                txt_tr = col.print_colored(txt_tr, BColors.OKGREEN)
                prev_tr_loss = tr_loss

            txt_val = 'costs_va %.5f ' % va_loss
            if va_loss < prev_va_loss:
                txt_val = col.print_colored(txt_val, BColors.OKGREEN)
                prev_va_loss = va_loss

            txt_tr_dist = 'dist_tr %.5f ' % tr_dist
            if tr_dist < prev_tr_dist:
                txt_tr_dist = col.print_colored(txt_tr_dist, BColors.OKGREEN)
                prev_tr_dist = tr_dist

            txt_va_dist = 'dist_va %.5f ' % va_dist
            if va_dist < prev_va_dist:
                txt_va_dist = col.print_colored(txt_va_dist, BColors.OKGREEN)
                prev_va_dist = va_dist

            txt_tr_mrr = 'mrr_tr %.2f ' % (100 * mrr_tr)
            if mrr_tr > prev_mrr_tr:
                txt_tr_mrr = col.print_colored(txt_tr_mrr, BColors.OKGREEN)
                prev_mrr_tr = mrr_tr

            txt_va_mrr = 'mrr_va %.2f ' % (100 * mrr_va)
            if mrr_va > prev_mrr_va:
                txt_va_mrr = col.print_colored(txt_va_mrr, BColors.OKGREEN)
                prev_mrr_va = mrr_va

            txt_tr_med_rank = 'medr_tr %.2f ' % med_rank_tr
            if med_rank_tr < prev_med_rank_tr:
                txt_tr_med_rank = col.print_colored(txt_tr_med_rank, BColors.OKGREEN)
                prev_med_rank_tr = med_rank_tr

            txt_va_med_rank = 'medr_va %.2f ' % med_rank_va
            if med_rank_va < prev_med_rank_va:
                txt_va_med_rank = col.print_colored(txt_va_med_rank, BColors.OKGREEN)
                prev_med_rank_va = med_rank_va

            print('  lr: %.9f' % learn_rate)
            print('  ' + txt_tr + txt_val)
            print('  ' + txt_tr_dist + txt_va_dist)
            print('  ' + txt_tr_mrr + txt_va_mrr + ' | ' + txt_tr_med_rank + txt_va_med_rank)

            # collect model evolution data
            pred_tr_err.append(tr_loss)
            pred_val_err.append(va_loss)
            dists_tr.append(tr_dist)
            dists_va.append(va_dist)
            ranks_tr.append(rank_tr)
            ranks_va.append(rank_va)
            mrrs_tr.append(mrr_tr)
            mrrs_va.append(mrr_va)
            evals_evol_tr.append(evals_tr)

            # save results
            exp_res = dict()
            exp_res['pred_tr_err'] = pred_tr_err
            exp_res['pred_val_err'] = pred_val_err
            exp_res['dist_tr'] = dists_tr
            exp_res['dist_val'] = dists_va
            exp_res['rank_tr'] = ranks_tr
            exp_res['rank_val'] = ranks_va
            exp_res['mrr_tr'] = mrrs_tr
            exp_res['mrr_val'] = mrrs_va
            exp_res['evals_tr'] = evals_evol_tr

            with open(log_file, 'w') as fp:
                pickle.dump(exp_res, fp, protocol=-1)

            # --- early stopping: preserve best model ---
            if last_improvement > patience:
                print(col.print_colored("Early Stopping!", BColors.WARNING))
                best = (best_epoch, prev_va_loss, prev_va_dist, 100 * prev_mrr_va)
                status = "Best Epoch: %d, Validation Loss: %.5f: Dist: %.5f Map: %.2f" % best
                print(col.print_colored(status, BColors.WARNING))

                # check for additional refinement steps
                if refinement_steps <= 0:
                    break

                else:
                    status = "Loading best parameters so far and refining (%d) with decreased learn rate ..." % refinement_steps
                    print(col.print_colored(status, BColors.WARNING))

                    # decrease refinement steps
                    last_improvement = 0
                    patience = 10
                    refinement_steps -= 1

                    # set net to best model so far
                    lasagne.layers.set_all_param_values(layers, best_model)

                    # reset optimizer
                    for _u, value in zip(iter_funcs['updates'].keys(), best_opt_state):
                        _u.set_value(value)

                    # decrease learn rate
                    learn_rate = np.float32(learn_rate * lr_multiplier)
                    learning_rate.set_value(learn_rate)

            # update learning rate
            learn_rate = update_learning_rate(learn_rate, epoch['number'])
            if learn_rate is not None:
                learning_rate.set_value(learn_rate)

            # maximum number of epochs reached
            if epoch['number'] >= num_epochs:
                break

    except KeyboardInterrupt:
        pass

    except:
        if do_raise:
            raise
        else:
            return l_out, prev_mrr_va

    # set net to best weights
    lasagne.layers.set_all_param_values(layers, best_model)

    return l_out, prev_mrr_va


# --- main --------------------------------------------------------------------
if __name__ == '__main__':
    pass
