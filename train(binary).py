import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as loader
import math
import numpy as np
from tqdm import tqdm

from model import *

import pandas as pd

from torch.utils.data import Dataset

from sklearn import metrics
import random

import time

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import matplotlib

matplotlib.use('Agg')


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
        train_num = int(seqs_num * 0.8)
        test_num = seqs_num - train_num

        index = range(seqs_num)
        indices = np.random.permutation(seqs_num)

        train_ids.append(np.asarray(indices[:train_num]))
        valid_ids.append(np.asarray([], dtype=np.int64))
        test_ids.append(np.asarray(indices[train_num:]))
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


class SSDataset(Dataset):

    def __init__(self, data_set, labels):
        self.data_set = data_set.astype(np.float32)

        self.labels = labels

    def __getitem__(self, item):
        return self.data_set[item], self.labels[item]

    def __len__(self):
        return self.data_set.shape[0]


class Constructor:

    def __init__(self, model, model_name=''):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device=self.device)
        self.model_name = model_name
        self.optimizer = optim.Adam(self.model.parameters(), lr=6.00029e-5, weight_decay=0.01)

        self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=5,
            eta_min=1e-6
        )

        self.loss_function = nn.BCELoss()

        self.batch_size = 64
        self.epochs = 200
        self.seed = 0

        self.epoch_metrics = {
            'AUC': [], 'SN': [], 'MCC': [], 'ACC': [], 'Pr': [], 'Loss': []
        }

    def learn(self, TrainLoader, TestLoader):
        for epoch in range(self.epochs):
            self.model.train()
            ProgressBar = tqdm(TrainLoader)
            epoch_losses = []

            for data in ProgressBar:
                self.optimizer.zero_grad()

                ProgressBar.set_description("Epoch %d" % epoch)
                x, y = data

                output = self.model(x=x.to(self.device))
                loss = self.loss_function(output, y.float().to(self.device))
                ProgressBar.set_postfix(loss=loss.item())

                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())

                self.cosine_scheduler.step()

            avg_loss = np.mean(epoch_losses)
            self.epoch_metrics['Loss'].append(avg_loss)

            if epoch % 1 == 0:
                metrics_dict = self.evaluate_metrics(TestLoader)
                for key in ['AUC', 'SN', 'MCC', 'ACC', 'Pr']:
                    self.epoch_metrics[key].append(metrics_dict[key])

                print(
                    f"Epoch {epoch} , Loss: {avg_loss:.4f}, AUC: {metrics_dict['AUC']:.4f}, MCC: {metrics_dict['MCC']:.4f}")

        print("\n=== Train Set Metrics ===")
        train_metrics = self.evaluate_metrics(TrainLoader)
        print(f"Train AUC: {train_metrics['AUC']:.4f}")
        print(f"Train MCC: {train_metrics['MCC']:.4f}")
        print(f"Train ACC: {train_metrics['ACC']:.4f}")
        print(f"Train SN: {train_metrics['SN']:.4f}")
        print(f"Train Pr: {train_metrics['Pr']:.4f}")

    def evaluate_metrics(self, TestLoader):
        self.model.eval()
        predicted_prob = []
        ground_label = []

        with torch.no_grad():
            for x, y in TestLoader:
                x = x.to(self.device)
                output = self.model(x)
                predicted_prob.extend(torch.sigmoid(output).flatten().detach().cpu().numpy())
                ground_label.extend(y.flatten().cpu().numpy())

        prob_array = np.array(predicted_prob)
        label_array = np.array(ground_label)

        if len(np.unique(label_array)) > 1:
            fpr, tpr, thresholds = metrics.roc_curve(label_array, prob_array)
            best_threshold = thresholds[np.argmax(tpr - fpr)]
            predicted_label = (prob_array > best_threshold).astype(int)

            sn = metrics.recall_score(y_true=label_array, y_pred=predicted_label, pos_label=1)
            mcc = metrics.matthews_corrcoef(y_true=label_array, y_pred=predicted_label)
            acc = metrics.accuracy_score(y_true=label_array, y_pred=predicted_label)
            auroc = metrics.roc_auc_score(y_true=label_array, y_score=prob_array)
            pr = metrics.precision_score(y_true=label_array, y_pred=predicted_label, pos_label=1)
        else:
            sn = mcc = acc = auroc = pr = 0.0

        return {
            'AUC': auroc, 'SN': sn, 'MCC': mcc, 'ACC': acc, 'Pr': pr
        }

    def plot_training_metrics(self):
        pass

    def inference(self, TestLoader):
        torch.save(self.model.state_dict(), 'final_model11.pt')
        predicted_prob = []
        ground_label = []
        self.model.eval()
        for x, y in TestLoader:
            x = x.to(self.device)
            output = self.model(x)
            predicted_prob.extend(torch.sigmoid(output).flatten().detach().cpu().numpy())
            ground_label.extend(y.flatten().cpu().numpy())
        return predicted_prob, ground_label

    def measure(self, predicted_prob, ground_label):
        prob_array = np.array(predicted_prob)
        label_array = np.array(ground_label)

        fpr, tpr, thresholds = metrics.roc_curve(label_array, prob_array)
        best_threshold = thresholds[np.argmax(tpr - fpr)]
        predicted_label = (prob_array > best_threshold).astype(int)
        print(f"\nBest threshold: {best_threshold:.4f}")

        sn = metrics.recall_score(y_true=label_array, y_pred=predicted_label, pos_label=1)
        sp = metrics.recall_score(y_true=label_array, y_pred=predicted_label, pos_label=0)
        mcc = metrics.matthews_corrcoef(y_true=label_array, y_pred=predicted_label)
        acc = metrics.accuracy_score(y_true=label_array, y_pred=predicted_label)
        auroc = metrics.roc_auc_score(y_true=label_array, y_score=prob_array)
        f1 = metrics.f1_score(y_true=label_array, y_pred=predicted_label)
        ap = metrics.average_precision_score(y_true=label_array, y_score=prob_array)
        pr = metrics.precision_score(y_true=label_array, y_pred=predicted_label, pos_label=1)

        print("\nEvaluation Metrics:")
        print(f"auROC (AUC): {auroc:.6f}")
        print(f"Sn : {sn:.6f}")
        print(f"Sp : {sp:.6f}")
        print(f"Pr : {pr:.6f}")
        print(f"Acc : {acc:.6f}")
        print(f"Mcc : {mcc:.6f}")
        print(f"F1-score: {f1:.6f}")
        print(f"Ap : {ap:.6f}")

        return sn, sp, mcc, acc, auroc, f1, ap, pr

    def run(self, Train_Set, Test__Set):
        Train_Loader = stable(loader.DataLoader(dataset=Train_Set, drop_last=True,
                                                batch_size=self.batch_size, shuffle=True, num_workers=0), seed)

        Test_Loader = stable(loader.DataLoader(dataset=Test__Set,
                                               batch_size=1, shuffle=False, num_workers=0), seed)

        self.learn(Train_Loader, Test_Loader)

        self.plot_training_metrics()

        predicted_value, ground_label = self.inference(Test_Loader)

        sn, sp, mcc, acc, auroc, f1, ap, pr = self.measure(predicted_value, ground_label)

        return sn, sp, mcc, acc, auroc, f1, ap, pr, predicted_value, ground_label

def resu():
    K = 1
    ratio_k = 0

    data1 = pd.read_csv(r"Carcin_LightGBM5.csv")

    X = np.array(data1.iloc[:, 1:])
    Y = np.array(data1.iloc[:, 0])
    Y = Y.reshape((Y.shape[0], 1))
    seqs_num = X.shape[0]
    in_size = X.shape[1]

    indices = np.arange(seqs_num)
    np.random.seed(0)
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]

    train_ids, test_ids, valid_ids = Id_k_folds(seqs_num, k_folds=K, ratio=ratio_k)
    Sn = []
    Sp = []
    Acc = []
    Mcc = []
    auROC = []
    F1 = []
    Ap = []
    Pr = []
    pre_value = []
    true_label = []
    start_time = time.time()

    for fold in range(K):
        train_index = train_ids[fold]
        test_index = test_ids[fold]

        overlap = np.intersect1d(train_index, test_index)
        if len(overlap) > 0:
            print(f" Warning: fold{fold} data leakage! Overlap count: {len(overlap)}")
            print(f"Overlap indices: {overlap[:5]}")
        else:
            print(f" Fold{fold} check passed: no data leakage")

        x_train = X[train_index]
        y_train = Y[train_index]

        x_test = X[test_index]
        y_test = Y[test_index]

        Train_Set = SSDataset(data_set=x_train, labels=y_train)
        Test__Set = SSDataset(data_set=x_test, labels=y_test)

        Train = Constructor(model=M_Cardio(in_size))

        print("\n_______________fold", fold, "_____________\n")
        sn, sp, mcc, acc, auroc, f1, ap, pr, predicted_value, ground_label = Train.run(Train_Set, Test__Set)
        Sn.append(sn)
        Sp.append(sp)
        Mcc.append(mcc)
        Acc.append(acc)
        auROC.append(auroc)
        F1.append(f1)
        Ap.append(ap)
        Pr.append(pr)

        pre_value += np.array(predicted_value).flatten().tolist()
        true_label += np.array(ground_label).flatten().tolist()

    end_time = time.time()
    total_time = end_time - start_time
    print("total_time:", total_time / K)

    print("\nAverage Evaluation Metrics:")
    print(f"auROC (AUC): {np.mean(auROC):.6f}")
    print(f"Sn : {np.mean(Sn):.6f}")
    print(f"Sp : {np.mean(Sp):.6f}")
    print(f"Pr : {np.mean(Pr):.6f}")
    print(f"Acc : {np.mean(Acc):.6f}")
    print(f"Mcc : {np.mean(Mcc):.6f}")
    print(f"F1-score: {np.mean(F1):.6f}")
    print(f"Ap : {np.mean(Ap):.6f}")

    return (
        [np.mean(Sn), np.mean(Sp), np.mean(Mcc), np.mean(Acc), np.mean(auROC), np.mean(F1), np.mean(Ap), np.mean(Pr)])


if __name__ == '__main__':
    resu()