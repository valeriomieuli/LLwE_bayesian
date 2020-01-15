import matplotlib.pyplot as plt
import numpy as np


def train_plot(train_result, fname):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figheight(4)
    fig.set_figwidth(9)

    ax1.plot(train_result.epoch, train_result.history["loss"], label="train_loss", color='red')
    ax1.set(xlabel='Epochs', ylabel='Loss')

    ax2.plot(train_result.epoch, train_result.history["acc"], label="train_acc", color='blue')
    ax2.set_ylim(bottom=0, top=1)
    ax2.set(xlabel='Epochs', ylabel='Accuracy')

    plt.subplots_adjust(wspace=0.3)
    plt.savefig(fname)


def accuracy_barchart(accuracies, phase, data_split):
    assert phase in ['autoencoders', 'experts']
    assert data_split in ['train', 'valid', 'test']

    datasets = list(accuracies.keys())
    accuracies = [accuracies[dataset] for dataset in datasets]
    n_tasks = len(accuracies)
    x_labels = ['Task: 1:' + str(i + 2) for i in range(n_tasks - 1)]
    x = np.arange(n_tasks - 1)

    width = .1
    fig, ax = plt.subplots()
    fig.set_figheight(9)
    fig.set_figwidth(12)
    fig.tight_layout(pad=12)
    ax.set_ylim(bottom=0, top=1.1)
    ax.set_ylabel('Accuracy')
    ax.set_title(phase.upper())
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)

    bars = [None for _ in range(len(x) + 1)]
    if len(x) % 2 == 0:
        bars[int(n_tasks / 2)] = ax.bar(x, accuracies[int(n_tasks / 2)], width=width, label=datasets[int(n_tasks / 2)])
        for i in reversed(range(0, int(n_tasks / 2))):
            bars[i] = ax.bar(x - width * (int(n_tasks / 2 - i)), accuracies[i], width=width, label=datasets[i])
        for i in range(int(n_tasks / 2 + 1), n_tasks):
            bars[i] = ax.bar(x + width * (i - int(n_tasks / 2)), accuracies[i], width=width, label=datasets[i])
    else:
        for i in reversed(range(0, int(n_tasks / 2))):
            bars[i] = ax.bar(x - (width / 2 + width * (n_tasks / 2 - i - 1)), accuracies[i], width=width,
                             label=datasets[i])
        for i in range(int(n_tasks / 2), n_tasks):
            bars[i] = ax.bar(x + (width / 2 + width * (i - n_tasks / 2)), accuracies[i], width=width, label=datasets[i])

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.92))
    plt.savefig(data_split + '_' + phase + '_acc_barchart.jpg')


'''def uncertainty_barchart(uncertainties, dataset, uncertainty_type, precision=2, granularity=0.05):
    uncertainties = [round(granularity * round(float(uncertainty) / granularity), precision)
                     for uncertainty in uncertainties]
    Xs = [round((i + 1) * 0.02, 2) for i in range(int(max(uncertainties) * 100 / (granularity * 100)))]
    d = dict((key, uncertainties.count(key)) for key in Xs)

    fig, ax = plt.subplots()
    fig.set_figheight(9)
    fig.set_figwidth(18)
    height = d.values()
    bars = d.keys()
    y_pos = np.arange(len(bars))
    ax.bar(y_pos, height)
    ax.set_title('Uncertainty distribution')
    plt.xticks(y_pos, bars, rotation=90)
    plt.savefig(dataset + '_' + uncertainty_type + 'unc__barchart.jpg')'''


def plot_uncertainty_per_sample(uncertainties, uncertainty_type, labels, bayesian_model, trainig_dataset, colors):
    n_charts = len(uncertainties)
    #n_samples = min([len(uncertainty) for uncertainty in uncertainties])

    '''unc_max, unc_min = None, None
    for unc in uncertainties:
        if unc_max is None or max(unc) > unc_max:
            unc_max = max(unc)
        if unc_min is None or min(unc) < unc_min:
            unc_min = min(unc)
    #print(unc_min, unc_max)'''

    fig, axs = plt.subplots(n_charts, sharex=True, sharey=True)
    fig.set_figheight(16)
    fig.set_figwidth(32)
    fig.text(0.5, 0.1, "SAMPLES", ha="center", va="center")
    fig.text(0.1, 0.5, uncertainty_type.upper() + " uncertainty", ha="center", va="center", rotation=90)
    fig.suptitle(trainig_dataset.upper() + '-' + bayesian_model)
    for i in range(n_charts):
        x = np.arange(len(uncertainties[i]))
        y = uncertainties[i]#[:n_samples]
        axs[i].scatter(x, y, label=labels[i].upper(), s=.75, color=colors[i])
        #axs[i].set_ylim(unc_min, unc_max)
        axs[i].legend()
        axs[i].grid()
    plt.subplots_adjust(wspace=0, hspace=0.025)
    plt.savefig(trainig_dataset + '-' + bayesian_model + '_' + uncertainty_type.upper() + '-unc_plot.jpg')
