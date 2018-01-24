
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('ticks')
cmap = sns.color_palette()

if __name__ == '__main__':
    """
    Plot model evolution
    """

    # add argument parser
    parser = argparse.ArgumentParser(description='Show evaluation plot.')
    parser.add_argument('results', metavar='N', type=str, nargs='+', help='result.pkl files.')
    parser.add_argument('--acc', help='evaluate accuracy.', action='store_true')
    parser.add_argument('--perc', help='show percentage value.', action='store_true')
    parser.add_argument('--max_epoch', type=int, default=None, help='last epoch to plot.')
    parser.add_argument('--ymin', help='minimum y value.', type=float, default=None)
    parser.add_argument('--ymax', help='maximum y value.', type=float, default=None)
    parser.add_argument('--watch', help='refresh plot.', action='store_true')
    parser.add_argument('--key', help='key for evaluation.', type=str, default=None)
    parser.add_argument('--y_label', help='y label text.', type=str, default=None)
    parser.add_argument('--lib', help='low is better.', action='store_true')
    args = parser.parse_args()

    while True:

        # load results
        all_results = dict()
        for result in args.results:
            exp_name = result.split('/')[-2]
            exp_ext = result.split('/')[-1].split('.pkl')[0].split('results')[1]
            exp_name += exp_ext
            with open(result, 'r') as fp:
                exp_res = pickle.load(fp)
                all_results[exp_name] = exp_res

        # present results
        plt.figure("Model Evolution")
        plt.clf()
        ax = plt.subplot(111)
        plt.subplots_adjust(bottom=0.15, left=0.15, right=0.9, top=0.95)

        for i, (exp_name, exp_res) in enumerate(all_results.iteritems()):

            if args.acc:

                if args.max_epoch is not None:
                    max_epoch = int(args.max_epoch)
                    exp_res['tr_accs'] = exp_res['tr_accs'][0:max_epoch]
                    exp_res['va_accs'] = exp_res['va_accs'][0:max_epoch]

                # train accuracy
                tr_accs = np.asarray(exp_res['tr_accs'], dtype=np.float)
                tr_accs[np.equal(tr_accs, None)] = np.nan
                indices = np.nonzero(~np.isnan(tr_accs))[0]
                tr_accs = tr_accs[indices]
                if args.perc:
                    acc = " (%.2f%%)" % tr_accs[-1]
                    label = exp_name + '_tr' + acc
                else:
                    label = exp_name + '_tr'
                plt.plot(indices, tr_accs, '-', color=cmap[i], linewidth=2.0, label=label)

                # validation accuracy
                va_accs = np.asarray(exp_res['va_accs'], dtype=np.float)
                va_accs[np.equal(va_accs, None)] = np.nan
                indices = np.nonzero(~np.isnan(va_accs))[0]
                va_accs = va_accs[indices]
                if args.perc:
                    acc = " (%.2f%%)" % np.mean(va_accs[-10::])
                    label = exp_name + '_va' + acc
                else:
                    label = exp_name + '_va'
                plt.plot(indices, va_accs, '-', color=cmap[i], linewidth=1.5, label=label)

                # plot smoothed accuracy
                smooth_va_accs = np.convolve(va_accs, np.asarray(5 * [1.0 / 5]), mode='valid')
                if len(va_accs) >= 5:
                    plt.plot(indices[2:-2], smooth_va_accs, '--', color=cmap[i], linewidth=1)

            else:

                if args.key is None:
                    key_tr = 'pred_tr_err'
                    key_va = 'pred_val_err'
                    label = "Loss"

                else:
                    key_tr = args.key % 'tr'
                    key_va = args.key % 'val'
                    label = args.key.replace("%s", "")

                plt.plot(exp_res[key_tr], '-', color=cmap[i], linewidth=2.5, label=exp_name + '_tr')
                plt.plot(exp_res[key_va], '-', color=cmap[i], linewidth=1.5, label=exp_name + '_va')

                # plot best validation loss
                if args.lib:
                    val = np.min(exp_res[key_va])
                    va = 'top'
                else:
                    val = np.max(exp_res[key_va])
                    va = 'bottom'
                plt.plot([0, len(exp_res[key_va]) - 1], [val] * 2, '--', color=cmap[i], alpha=0.5)
                plt.text(len(exp_res[key_va]) - 1, val, val, color=cmap[i], va=va, ha='right')

        if args.acc:
            plt.ylabel("Accuracy", fontsize=20)
            plt.legend(loc="best", fontsize=18).draggable()
            plt.ylim([args.ymin, 102])
        else:
            plt.ylabel(label, fontsize=20)
            plt.legend(loc="best", fontsize=20).draggable()

        if args.ymin is not None and args.ymax is not None:
            plt.ylim([args.ymin, args.ymax])

        if args.max_epoch is not None:
            plt.xlim([0, args.max_epoch])

        plt.xlabel("Epoch", fontsize=20)
        plt.grid('on')

        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        
        if args.y_label is not None:
            plt.ylabel(args.y_label, fontsize=20)
        
        plt.draw()

        if args.watch:
            plt.pause(10.0)
        else:
            plt.show(block=True)
            break
