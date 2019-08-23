import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm

from transfer.config.config import TRAIN_DATA_VECTOR


def plot_smoothness_per_model(model_index):
    means = []

    # takes all vectors by train size --> (vec_1000train,vec_200train ...)
    for i in range(len(TRAIN_DATA_VECTOR)):
        cur_train_data = TRAIN_DATA_VECTOR[i]
        st = f'smoothness_vector_of_model_{str(model_index)}_and_{str(cur_train_data)}_train_data.npy'
        data_array = np.load(st)
        means.append(tuple(data_array))

    # taking the size, all the arays has the same size..
    num_of_layers = len(data_array)

    # data to plot
    n_groups = num_of_layers

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.1
    opacity = 0.9

    color = iter(cm.rainbow(np.linspace(0, 1, len(means) + 1)))

    for i in range(len(means)):
        plt.bar(index + bar_width * i, means[i], bar_width, alpha=opacity, color=next(color)
                , label='train_' + str(TRAIN_DATA_VECTOR[i]))

    plt.xlabel('Layers')
    plt.ylabel('Smoothness')
    plt.title('Smoothness by Layers')
    plt.xticks(index + bar_width, ('1', '2', '3', '4', '5', '6', '7', '8', '9'))
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'smoothness_graph_of_model_{model_index}')


def main():
    for i in range(5):
        plot_smoothness_per_model(i)


if __name__ == '__main__':
    main()
