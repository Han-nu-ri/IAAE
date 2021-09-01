import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import wandb

# dataset = 'emnist'
# wandb.login()
# wandb.init(project="AAE_plot",
#            config={
#                "dataset": dataset
#            })
# distributions = ['standard_normal', 'uniform', 'gamma', 'beta', 'chi', 'dirichlet', 'laplace']
# for each_epoch in range(0, 200, 5):
#     real_pca_drawn = False
#     plt.rcParams["figure.figsize"] = (15, 15)
#     for each_distribution in distributions:
#         each_file = glob.glob(f"pca_data/emnist*{each_distribution}_{each_epoch}_*")
#         each_pca_data = pd.read_csv(each_file[0])
#         if not real_pca_drawn:
#             plt.scatter(each_pca_data['real_pca_x'], each_pca_data['real_pca_y'], label='real', alpha=0.8, s=0.1)
#             real_pca_drawn = True
#         plt.scatter(each_pca_data['fake_pca_x'], each_pca_data['fake_pca_y'], label=f'fake_{each_distribution}',
#                     alpha=0.6, s=0.1)
#     plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=len(distributions)+1, fontsize=10, markerscale=10)
#     plt.xlabel('pca dim1')
#     feature_pca_plot = plt.ylabel('pca dim2')
#     feature_pca_plot = wandb.log({'IsNet feature': wandb.Image(feature_pca_plot)}, step=each_epoch)
#     plt.close()

# dataset = 'emnist'
# wandb.login()
# wandb.init(project="AAE_plot",
#            config={
#                "dataset": dataset
#            })
# distributions = ['real', 'standard_normal', 'uniform', 'gamma', 'beta', 'chi', 'dirichlet', 'laplace']
# tables = pd.DataFrame()
# for each_distribution in distributions:
#     if each_distribution == 'real':
#         each_file = glob.glob(f"pca_data/*{dataset}_standard_normal_195_*")
#         each_pca_data = pd.read_csv(each_file[0])
#         each_table_data = {
#             "distribution": [each_distribution] * len(each_pca_data['real_pca_x']),
#             "pca_x": list(each_pca_data['real_pca_x']),
#             "pca_y": list(each_pca_data['real_pca_y'])
#         }
#     else:
#         each_file = glob.glob(f"pca_data/*{dataset}_{each_distribution}_195_*")
#         each_pca_data = pd.read_csv(each_file[0])
#         each_table_data = {
#             "distribution": [each_distribution] * len(each_pca_data['fake_pca_x']),
#             "pca_x": list(each_pca_data['fake_pca_x']),
#             "pca_y": list(each_pca_data['fake_pca_y'])
#         }
#     each_table = pd.DataFrame(each_table_data)
#     tables = tables.append(each_table, ignore_index=True)
#
# #table = wandb.Table(data=tables, columns=["distribution" "pca_x", "pca_y"])
# wandb.log({"IsNet feature": wandb.Table(dataframe=tables)})


dataset = 'ffhq32'
distributions = ['real', 'standard_normal', 'uniform', 'gamma', 'beta', 'chi', 'laplace']
tables = list()
for each_distribution in distributions:
    if each_distribution == 'real':
        each_file = glob.glob(f"pca_data/*{dataset}_standard_normal_195_*")
        each_pca_data = pd.read_csv(each_file[0])
        each_table_data = {
            "distribution": [each_distribution] * len(each_pca_data['real_pca_x']),
            "pca_x": list(each_pca_data['real_pca_x']),
            "pca_y": list(each_pca_data['real_pca_y'])
        }
    else:
        each_file = glob.glob(f"pca_data/*{dataset}_{each_distribution}_195_*")
        each_pca_data = pd.read_csv(each_file[0])
        each_table_data = {
            "distribution": [each_distribution] * len(each_pca_data['fake_pca_x']),
            "pca_x": list(each_pca_data['fake_pca_x']),
            "pca_y": list(each_pca_data['fake_pca_y'])
        }
    tables.append(pd.DataFrame(each_table_data))

wandb.login()
wandb.init(project="AAE_plot",
           config={
               "dataset": dataset
           })
for i in range(0, len(distributions)):
    wandb.log({f"IsNet feature {distributions[i]}": wandb.Table(dataframe=tables[i])})
