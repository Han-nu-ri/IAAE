import pandas as pd
import matplotlib.pyplot as plt
import glob
import wandb
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import os


def log_pca_table_in_wandb(pca_list, dataset, distributions):
    wandb.login()
    wandb.init(project="AAE_plot",
               config={
                   "dataset": dataset
               })
    for i in range(0, len(distributions)):
        wandb.log({f"IsNet feature {distributions[i]}": wandb.Table(dataframe=pca_list[i])})


def get_pca(dataset, distributions, folder):
    pca_list = list()
    for each_distribution in distributions:
        if each_distribution == 'real':
            each_file = glob.glob(f"{folder}/*{dataset}_standard_normal_199_*")
            each_pca_data = pd.read_csv(each_file[0])
            each_table_data = {
                "distribution": [each_distribution] * len(each_pca_data['real_pca_x']),
                "pca_x": list(each_pca_data['real_pca_x']),
                "pca_y": list(each_pca_data['real_pca_y']),
                "pca_z": list(each_pca_data['real_pca_z'])
            }
        else:
            each_file = glob.glob(f"{folder}/*{dataset}_{each_distribution}_199_*")
            each_pca_data = pd.read_csv(each_file[0])
            each_table_data = {
                "distribution": [each_distribution] * len(each_pca_data['fake_pca_x']),
                "pca_x": list(each_pca_data['fake_pca_x']),
                "pca_y": list(each_pca_data['fake_pca_y']),
                "pca_z": list(each_pca_data['fake_pca_z'])
            }
        pca_list.append(pd.DataFrame(each_table_data))
    return pca_list


def plot_3d(pca_list, distributions):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    markers = ['.', ',', 'o', 'v', '^', '<', '>']
    for i in range(0, len(pca_list)):
        ax.scatter(pca_list[i]['pca_x'], pca_list[i]['pca_y'], pca_list[i]['pca_z'], marker=markers[i], alpha=0.3,
                   edgecolors='none', label=distributions[i])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend()
    plt.show()


def get_target_data_of_tsne(dataset, distributions):
    folder = 'feature_data_ffhq128'
    target_data = None
    for each_distribution in distributions:
        if each_distribution == 'real':
            each_file = glob.glob(f"{folder}/*{dataset}_standard_normal_199_real_*")
        else:
            each_file = glob.glob(f"{folder}/*{dataset}_{each_distribution}_199_fake_*")

        each_target_data = pd.read_csv(each_file[0], header=None)
        if target_data is None:
            target_data = each_target_data.values
        else:
            target_data = np.concatenate((target_data, each_target_data.values))
    return target_data



def main():
    dataset = 'ffhq_128'
    folder = 'pca_data_ffhq128'
    distributions = ['real', 'standard_normal', 'uniform', 'gamma', 'beta', 'chi', 'laplace']
    plot_pca = False
    if plot_pca:
        pca_list = get_pca(dataset, distributions, folder)
        plot_3d(pca_list, distributions)
    make_tsne = False
    if make_tsne:
        target_data = get_target_data_of_tsne(dataset, distributions)
        tsne = TSNE(n_components=2, verbose=1)
        tsne_results = tsne.fit_transform(target_data)
        labels = []
        for i in distributions:
            labels = labels + [i]*int(target_data.shape[0]/len(distributions))
        tsne_for_plot = pd.DataFrame(data={'label': labels, 'tsne1d': tsne_results[:, 0], 'tsne2d': tsne_results[:, 1]})
        folder_name = 'tsne_data/'
        os.makedirs(folder_name, exist_ok=True)
        tsne_for_plot.to_csv(f"{folder_name}tsne_data.csv")
    plot_tsne = True
    if plot_tsne:
        tsne = pd.read_csv('tsne_data/image_size_128_epoch_200_test_1/tsne_data.csv')
        real_sn = tsne[(tsne["label"] == 'real') | (tsne["label"] == 'standard_normal')]
        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x='tsne1d', y='tsne2d',
            hue="label",
            palette=sns.color_palette("hls", 2),
            data=real_sn,
            legend="full",
            alpha=0.3
        )
        plt.savefig('tsne_real_sn')

        real_uniform = tsne[(tsne["label"] == 'real') | (tsne["label"] == 'uniform')]
        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x='tsne1d', y='tsne2d',
            hue="label",
            palette=sns.color_palette("hls", 2),
            data=real_uniform,
            legend="full",
            alpha=0.3
        )
        plt.savefig('tsne_real_uniform')

        real_sn_uniform = tsne[(tsne["label"] == 'real') | (tsne["label"] == 'standard_normal') |
                               (tsne["label"] == 'uniform')]
        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x='tsne1d', y='tsne2d',
            hue="label",
            palette=sns.color_palette("hls", 3),
            data=real_sn_uniform,
            legend="full",
            alpha=0.3
        )
        plt.savefig('tsne_real_sn_uniform')


if __name__ == "__main__":
    main()
