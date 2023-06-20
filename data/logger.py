import numpy as np
import torch
import os
import pickle
import matplotlib.pyplot as plt


class LOGGER:
    def __init__(self, metrics, save_dir):
        self.save_dir = save_dir
        self.metrics = metrics
        self.logger = {}
        for metric in self.metrics:
            self.logger[metric] = []

    def log(self, metric_values):
        for metric, value in metric_values:
            self.logger[metric].append(value)

    def print(self, epoch):
        print(f"Results for epoch {epoch}")
        results_string = ""
        for metric, value in self.logger:
            assert len(value) == epoch - 1
            results_string += f"{metric}: {value[-1]}\t"
        print(results_string)

    def save(self):
        save_path = os.path.join(self.save_dir, "logger.pkl")
        with open(save_path, 'wb') as f:  # open a text file
            pickle.dump(self.logger, f)  # serialize the list

    def plot(self):
        # Plot Train Accuracies
        plt.plot(self.logger["Train Accuracy"], label="Train Accuracy", color="red")
        plt.plot(self.logger["Train Minority Accuracy"], label="Train Minority Accuracy", color="green")
        plt.plot(self.logger["Train Majority Accuracy"], label="Train Majority Accuracy", color="blue")
        plt.title("Train Accuracies")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        save_path = os.path.join(self.save_dir, "train_accuracies.png")
        plt.savefig(save_path)
        plt.clf()

        # Plot Train Losses
        plt.plot(self.logger["Train Loss"], label="Train Loss", color="red")
        plt.plot(self.logger["Train Minority Loss"], label="Train Minority Loss", color="green")
        plt.plot(self.logger["Train Majority Loss"], label="Train Majority Loss", color="blue")
        plt.title("Train Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        save_path = os.path.join(self.save_dir, "train_losses.png")
        plt.savefig(save_path)
        plt.clf()

        # Plot Validation Accuracies
        plt.plot(self.logger["Val Accuracy"], label="Validation Accuracy", color="red")
        plt.plot(self.logger["Val Minority Accuracy"], label="Validation Minority Accuracy", color="green")
        plt.plot(self.logger["Val Majority Accuracy"], label="Validation Majority Accuracy", color="blue")
        plt.title("Validation Accuracies")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        save_path = os.path.join(self.save_dir, "validation_accuracies.png")
        plt.savefig(save_path)
        plt.clf()


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
