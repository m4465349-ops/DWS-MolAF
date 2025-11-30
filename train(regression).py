import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as loader
import numpy as np, math
import numpy as np
from tqdm import tqdm

from model import *

import pandas as pd

from torch.utils.data import Dataset

from sklearn import metrics
from sklearn import model_selection
import random

import time

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from sklearn.metrics import roc_curve, precision_recall_curve, auc, r2_score
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def stable(dataloader, seed):
    seed_everything(seed)
    return dataloader


seed = 0
seed_everything(seed)


def Id_k_folds(seqs_num, k_folds, ratio):
    train_ids = []
    test_ids = []
    valid_ids = []
    if k_folds == 1:
        train_num = int(seqs_num * 0.7)
        test_num = seqs_num - train_num
        valid_num = int(train_num * ratio)
        train_num = train_num - valid_num
        index = range(seqs_num)
        train_ids.append(np.asarray(index[:train_num]))
        valid_ids.append(np.asarray(index[train_num:train_num + valid_num]))
        test_ids.append(np.asarray(index[train_num + valid_num:]))
    else:
        each_fold_num = int(math.ceil(seqs_num / k_folds))
        for fold in range(k_folds):
            index = range(seqs_num)
            index_slice = index[fold * each_fold_num: (fold + 1) * each_fold_num]
            index_left = list(set(index) - set(index_slice))
            test_ids.append(np.asarray(index_slice))
            train_num = len(index_left) - int(len(index_left) * ratio)
            train_ids.append(np.asarray(index_left[:train_num]))
            valid_ids.append(np.asarray(index_left[train_num:]))

    return (train_ids, test_ids, valid_ids)


class earlystopping:

    def __init__(self, patience=25, verbose=False, delta=0, path='checkpoint101.pt', trace_func=print):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint101(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint101(val_loss, model)
            self.counter = 0

    def save_checkpoint101(self, val_loss, model):
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class SSDataset(Dataset):

    def __init__(self, data_set, labels):
        self.data_set = data_set.astype(np.float32)
        self.labels = labels.astype(np.float32)

    def __getitem__(self, item):
        return self.data_set[item], self.labels[item]

    def __len__(self):
        return self.data_set.shape[0]


class Constructor:

    def __init__(self, model, stop, model_name=''):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device=self.device)
        self.model_name = model_name
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.scheduler = None

        self.loss_function = nn.MSELoss()

        self.early_stopping = stop

        self.batch_size = 180
        self.epochs = 200
        self.seed = 0

    def learn(self, TrainLoader, ValidateLoader):
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer=self.optimizer,
            max_lr=3e-3,
            steps_per_epoch=len(TrainLoader),
            epochs=self.epochs,
            pct_start=0.3,
            anneal_strategy='cos'
        )
        for epoch in range(self.epochs):
            self.model.train()
            ProgressBar = tqdm(TrainLoader)
            for data in ProgressBar:
                self.optimizer.zero_grad()

                ProgressBar.set_description("Epoch %d" % epoch)
                x, y = data

                output = self.model(x=x.to(self.device))
                loss = self.loss_function(output, y.float().to(self.device))
                ProgressBar.set_postfix(loss=loss.item())

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            valid_loss = []
            self.model.eval()
            with torch.no_grad():
                for valid_x, valid_y in ValidateLoader:
                    valid_output = self.model(x=valid_x.to(self.device))
                    valid_y = valid_y.float().to(self.device)
                    valid_loss.append(self.loss_function(valid_output, valid_y).item())

                valid_loss_avg = torch.mean(torch.Tensor(valid_loss))

                print("Validation loss:{}".format(valid_loss_avg))
                self.early_stopping(valid_loss_avg, self.model)
                if self.early_stopping.early_stop:
                    print("Early stopping!")
                    break

    def inference(self, TestLoader):
        self.model.load_state_dict(torch.load('checkpoint101.pt'))
        predicted_value = []
        ground_label = []
        self.model.eval()
        for x, y in TestLoader:
            output = self.model(x.to(self.device))
            predicted_value.append(output.squeeze(dim=0).detach().cpu().numpy().tolist())
            ground_label.append(y.squeeze(dim=0).detach().cpu().numpy().tolist())

        return predicted_value, ground_label

    def measure(self, predicted_value, ground_label):
        y_true = np.array(ground_label).flatten()
        y_pred = np.array(predicted_value).flatten()

        mse = metrics.mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        pearson = np.corrcoef(y_true, y_pred)[0, 1]
        spearman = pd.Series(y_true).corr(pd.Series(y_pred), method='spearman')
        r2 = r2_score(y_true, y_pred)

        def concordance_index(y_true, y_pred):
            y_true = np.array(y_true).flatten()
            y_pred = np.array(y_pred).flatten()

            n = len(y_true)
            total = 0
            concordant = 0

            for i in range(n):
                for j in range(i + 1, n):
                    if y_true[i] != y_true[j]:
                        total += 1
                        if (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]) or \
                                (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]):
                            concordant += 1

            return concordant / total if total > 0 else 0

        c_index = concordance_index(y_true, y_pred)

        print("\nEvaluation Metrics:")
        print(f"RMSE: {rmse:.6f}")
        print(f"MSE: {mse:.6f}")
        print(f"Pearson: {pearson:.6f}")
        print(f"Spearman: {spearman:.6f}")
        print(f"C-index: {c_index:.6f}")
        print(f"R-squared: {r2:.6f}")

        return rmse, mse, pearson, spearman, c_index, r2

    def run(self, Train_Set, Vaild_Set, Test__Set):
        Train_Loader = stable(loader.DataLoader(dataset=Train_Set, drop_last=True,
                                                batch_size=self.batch_size, shuffle=True, num_workers=0), seed)

        Vaild_Loader = stable(loader.DataLoader(dataset=Vaild_Set, drop_last=True,
                                                batch_size=self.batch_size, shuffle=False, num_workers=0), seed)

        Test_Loader = stable(loader.DataLoader(dataset=Test__Set,
                                               batch_size=1, shuffle=False, num_workers=0), seed)

        self.learn(Train_Loader, Vaild_Loader)
        predicted_value, ground_label = self.inference(Test_Loader)

        rmse, mse, pearson, spearman, c_index, r2 = self.measure(predicted_value, ground_label)

        return rmse, mse, pearson, spearman, c_index, r2, predicted_value, ground_label


def resu():
    K = 8
    ratio_k = 0.1

    data1 = pd.read_csv(r"intraperitoneal-500.csv")

    X = np.array(data1.iloc[:, 1:])
    Y = np.array(data1.iloc[:, 0])
    Y = Y.reshape((Y.shape[0], 1))
    seqs_num = X.shape[0]
    in_size = X.shape[1]

    indices = np.arange(seqs_num)
    np.random.seed(0)
    np.random.shuffle(indices)
    seqs_data_train = X[indices]
    intensity_train = Y[indices]

    train_ids, test_ids, valid_ids = Id_k_folds(seqs_num, k_folds=K, ratio=ratio_k)
    RMSE = []
    MSE = []
    Pearson = []
    Spearman = []
    C_index = []
    R2 = []
    pre_value = []
    true_label = []
    start_time = time.time()

    for fold in range(K):
        x_train = seqs_data_train[train_ids[fold]]
        y_train = intensity_train[train_ids[fold]]

        x_valid = seqs_data_train[valid_ids[fold]]
        y_valid = intensity_train[valid_ids[fold]]

        x_test = seqs_data_train[test_ids[fold]]
        y_test = intensity_train[test_ids[fold]]

        Train_Set = SSDataset(data_set=x_train, labels=y_train)
        Vaild_Set = SSDataset(data_set=x_valid, labels=y_valid)
        Test__Set = SSDataset(data_set=x_test, labels=y_test)

        early_stopping = earlystopping(patience=25, verbose=True)

        Train = Constructor(model=M_Acute(in_size), stop=early_stopping)

        print("\n_______________fold", fold, "_____________\n")
        rmse, mse, pearson, spearman, c_index, r2, predicted_value, ground_label = Train.run(Train_Set, Vaild_Set,
                                                                                             Test__Set)
        RMSE.append(rmse)
        MSE.append(mse)
        Pearson.append(pearson)
        Spearman.append(spearman)
        C_index.append(c_index)
        R2.append(r2)

        pre_value += np.array(predicted_value).flatten().tolist()
        true_label += np.array(ground_label).flatten().tolist()

    end_time = time.time()
    total_time = end_time - start_time
    print("total_time:", total_time / K)

    print("\nAverage Evaluation Metrics:")
    print(f"RMSE: {np.mean(RMSE):.6f}")
    print(f"MSE: {np.mean(MSE):.6f}")
    print(f"Pearson: {np.mean(Pearson):.6f}")
    print(f"Spearman: {np.mean(Spearman):.6f}")
    print(f"C-index: {np.mean(C_index):.6f}")
    print(f"R-squared: {np.mean(R2):.6f}")

    return [np.mean(RMSE), np.mean(MSE), np.mean(Pearson), np.mean(Spearman), np.mean(C_index), np.mean(R2)]


if __name__ == '__main__':
    resu()