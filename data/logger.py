import numpy as np
import torch
import os


class LOGGER:
    def __init__(self, metrics):
        self.metrics = metrics
        self.logger = {}
        for metric in self.metrics:
            self.logger[metric] = []

    def log(self, metric_values):
        for metric, value in metric_values:
            self.logger[metric].append(value)


def log_data(x, y, p, handler, dataset_name, dataset_split):
    base_dir = os.path.join("data", dataset_name)
    os.makedirs(base_dir, exist_ok=True)
    print(dataset_split)
    unique_y, unique_p = np.unique(y), np.unique(p)
    for y_value in unique_y:
        for p_value in unique_p:
            group_idxs = np.arange(len(y))[np.where((y == y_value) & (p == p_value))]
            print(f"Group Counts for y == {y_value} and p == {p_value}: {len(group_idxs)}")
            #dataset= handler([x[i] for i in group_idxs], torch.Tensor(y[group_idxs]).long(), isTrain=False)
            #img = dataset.__getimage__(0)
            #img.save(os.path.join(base_dir, f"y={y_value}-p={p_value}.jpg"))
