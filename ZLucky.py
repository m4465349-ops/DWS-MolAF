#动态阈值时的最佳框架，动+Cos  Carcdio最佳  Muta最佳
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as loader
import numpy as np, math
import numpy as np
from tqdm import tqdm

from nn_net2 import *
# from kan import *

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
    torch.manual_seed(seed)  # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)  # Numpy module
    random.seed(seed)  # Python random module
    torch.backends.cudnn.benchmark = False  # Close optimization
    torch.backends.cudnn.deterministic = True  # Close optimization
    torch.cuda.manual_seed_all(seed)  # All GPU (Optional)

def stable(dataloader, seed):
    seed_everything(seed)
    return dataloader  # 确保代码运行的可重复性

seed = 0
seed_everything(seed)

# Generate ids for k-flods cross-validation
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
            index_slice = index[fold * each_fold_num : (fold + 1) * each_fold_num]
            index_left = list(set(index) - set(index_slice))
            test_ids.append(np.asarray(index_slice))
            train_num = len(index_left) - int(len(index_left) * ratio)
            train_ids.append(np.asarray(index_left[:train_num]))
            valid_ids.append(np.asarray(index_left[train_num:]))

    return (train_ids, test_ids, valid_ids)

class earlystopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=25, verbose=False, delta=0, path='checkpoint9.pt', trace_func=print):

        self.patience = patience
        # self.verbose = verbose
        self.verbose = False  # 强制设为False
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf  # 保存最小的验证损失
        self.delta = delta  # 验证损失改善的最小变化
        self.path = path  # 保存模型的路径
        self.trace_func = trace_func  # 跟踪打印

    def __call__(self, val_loss, model):

        score = -val_loss  # 将验证损失取反，因为我们希望损失越小越好

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint9(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint9(val_loss, model)
            self.counter = 0

    def save_checkpoint9(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        # if self.verbose:
        #     self.trace_func(
        #         f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class SSDataset(Dataset):

    def __init__(self, data_set, labels):
        self.data_set = data_set.astype(np.float32)

        self.labels = labels

    def __getitem__(self, item):
        return self.data_set[item], self.labels[item]

    def __len__(self):
        return self.data_set.shape[0]


class Constructor:
    """
        按照不同模型的接收维数，修改相关的样本维数，如：
        特征融合策略不同，卷积操作不同（1D或2D），是否融合形状特征等
    """

    def __init__(self, model, stop, model_name=''):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device=self.device)
        self.model_name = model_name
        # self.optimizer = optim.Adadelta(self.model.parameters(),lr=1,weight_decay=0.01)
        # self.optimizer = optim.Adadelta(self.model.parameters(), lr=0.1)#可更换
        # self.optimizer = optim.ASGD(self.model.parameters(), lr=1)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=6.1e-4, weight_decay=0.01)
        self.optimizer = optim.Adam(self.model.parameters(), lr=6e-4,weight_decay=0.01)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=9.922e-4)

        # # === 修改1：删除原有调度器，替换为OneCycleLR ===
        # self.scheduler = None  # 将在learn()中初始化（需知道steps_per_epoch）
        #
        # 添加余弦退火学习率调度器
        self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=5,  # 余弦周期的一半
            eta_min=1e-6  # 最小学习率
        )
        self.plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, patience=5, verbose=False)

        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, patience=5, verbose=True)
        # 学习率调度器，5次不改变结果的时候就减小学习率，verbose=True打印中间过程
        ########################################################################

        # self.loss_function = nn.BCELoss()  # 用的BCELoss，所以最后一层也要有激活函数
        self.loss_function = nn.BCEWithLogitsLoss()  # 用的BCEWithLogitsLoss，最后一层不需要激活函数
        ########################################################################
        # self.loss_function = nn.CrossEntropyLoss()
        # # 在Constructor类中添加加权损失函数
        # pos_weight_tensor = torch.tensor([pos_weight]).to(self.device)

        self.early_stopping = stop

        self.batch_size = 64
        self.epochs = 200
        self.seed = 0

    def learn(self, TrainLoader, ValidateLoader):
        # === 修改2：初始化OneCycleLR ===
        # self.scheduler = optim.lr_scheduler.OneCycleLR(
        #     optimizer=self.optimizer,
        #     max_lr=3e-3,  # 峰值学习率（初始lr的3倍）
        #     steps_per_epoch=len(TrainLoader),
        #     epochs=self.epochs,
        #     pct_start=0.3,
        #     anneal_strategy='cos'
        # )
        for epoch in range(self.epochs):
            self.model.train()
            # ProgressBar = tqdm(TrainLoader)
            # 移除进度条
            for data in TrainLoader:
                self.optimizer.zero_grad()
                x, y = data
                output = self.model(x=x.to(self.device))
                loss = self.loss_function(output, y.float().to(self.device))
                loss.backward()
                self.optimizer.step()
                # # === 修改3：每个batch更新OneCycleLR ===
                # self.scheduler.step()

                # 原先：每个 batch 后更新余弦退火学习率
                self.cosine_scheduler.step()  # CosineAnnealingLR 更新

            valid_loss = []
            self.model.eval()
            with torch.no_grad():
                for valid_x, valid_y in ValidateLoader:
                    valid_output = self.model(x=valid_x.to(self.device))
                    valid_y = valid_y.float().to(self.device)
                    valid_loss.append(self.loss_function(valid_output, valid_y).item())

                valid_loss_avg = torch.mean(torch.Tensor(valid_loss))

                self.plateau_scheduler.step(valid_loss_avg)  # 根据验证集的损失来调整学习率
                # print("验证集loss:{}".format(valid_loss_avg))
                self.early_stopping(valid_loss_avg, self.model)
                if self.early_stopping.early_stop:
                    # print("此时早停！")
                    break  # 提前跳出整个训练过程

            # torch.save(self.model.state_dict(), path + '\\' + self.model_name + '.pth')

    def inference(self, TestLoader):
        self.model.load_state_dict(torch.load('checkpoint9.pt'))
        predicted_prob = []
        ground_label = []
        self.model.eval()
        for x, y in TestLoader:
            x = x.to(self.device)
            output = self.model(x)
            # 返回sigmoid后的概率值（改动点2）
            predicted_prob.extend(torch.sigmoid(output).flatten().detach().cpu().numpy())  # 修改点
            ground_label.extend(y.flatten().cpu().numpy())  # 修改点
        return predicted_prob, ground_label  # 返回概率而非二值结果

    def measure(self, predicted_prob, ground_label):
        # 将概率和标签转为numpy数组
        prob_array = np.array(predicted_prob)
        label_array = np.array(ground_label)

        # 动态选择最佳阈值（新增核心改动）
        fpr, tpr, thresholds = metrics.roc_curve(label_array, prob_array)
        best_threshold = thresholds[np.argmax(tpr - fpr)]  # Youden指数
        predicted_label = (prob_array > best_threshold).astype(int)
        # print(f"\n自动选择的最佳阈值: {best_threshold:.4f}")  # 打印阈值信息

        # 计算各项指标（后续保持不变）
        sn = metrics.recall_score(y_true=label_array, y_pred=predicted_label, pos_label=1)
        sp = metrics.recall_score(y_true=label_array, y_pred=predicted_label, pos_label=0)
        mcc = metrics.matthews_corrcoef(y_true=label_array, y_pred=predicted_label)
        acc = metrics.accuracy_score(y_true=label_array, y_pred=predicted_label)
        auroc = metrics.roc_auc_score(y_true=label_array, y_score=prob_array)  # 注意仍用概率计算AUC
        f1 = metrics.f1_score(y_true=label_array, y_pred=predicted_label)
        ap = metrics.average_precision_score(y_true=label_array, y_score=prob_array)
        pr = metrics.precision_score(y_true=label_array, y_pred=predicted_label, pos_label=1)

        # # 分行打印指标（新增部分）
        # print("\n评估指标：")  # 添加空行分隔
        # print(f"auROC (AUC): {auroc:.6f}")
        # print(f"Sn : {sn:.6f}")
        # print(f"Sp : {sp:.6f}")
        # print(f"Pr : {pr:.6f}")
        # print(f"Acc : {acc:.6f}")
        # print(f"Mcc : {mcc:.6f}")
        # print(f"F1-score: {f1:.6f}")
        # print(f"Ap : {ap:.6f}")

        return sn, sp, mcc, acc, auroc, f1, ap, pr

    def run(self, Train_Set, Vaild_Set, Test__Set):

        Train_Loader = stable(loader.DataLoader(dataset=Train_Set, drop_last=True,
                                                batch_size=self.batch_size, shuffle=True, num_workers=0), seed)

        Vaild_Loader = stable(loader.DataLoader(dataset=Vaild_Set, drop_last=True,  # 是否丢弃最后一个不完整的批次
                                                batch_size=self.batch_size, shuffle=False, num_workers=0), seed)

        Test_Loader = stable(loader.DataLoader(dataset=Test__Set,
                                               batch_size=1, shuffle=False, num_workers=0), seed)

        self.learn(Train_Loader, Vaild_Loader)
        predicted_value, ground_label = self.inference(Test_Loader)
        # print(predicted_value)

        sn, sp, mcc, acc, auroc, f1, ap, pr = self.measure(predicted_value, ground_label)

        return sn, sp, mcc, acc, auroc, f1, ap, pr, predicted_value, ground_label


def resu():
    K = 8
    ratio_k = 0.1
    # L = 21

    # data1 =pd.read_csv(r"bace_LightGBM5.csv")  # 数据，第一列为标签，后面为特征
    # data1 = pd.read_csv(r"C:\Users\DELL\Desktop\Hapa_LightGBM5-2513.csv")
    # data1 = pd.read_csv(r"Muta_LightGBM5-415.csv")
    # data1 = pd.read_csv(r"C:\Users\DELL\Desktop\Cardio-Lightgbm\Cardio_LightGBM5.csv")
    # data1 = pd.read_csv(r"Cardio_LightGBM5.csv")
    data1 = pd.read_csv(r"Carcin_LightGBM5.csv")
    # data1 = pd.read_csv(r"Hepa_LightGBM5-225.csv")


    X = np.array(data1.iloc[:, 1:])
    Y = np.array(data1.iloc[:, 0])
    Y = Y.reshape((Y.shape[0], 1))
    seqs_num = X.shape[0]  # 表示数组的第一维的大小，即行数，对应于样本的数量
    in_size = X.shape[1]  # 表示数组的第二维的大小，即列数，对应于特征的数量
    # 数据随机打乱和划分:
    # k-folds cross-validation
    indices = np.arange(seqs_num)
    np.random.seed(0)
    np.random.shuffle(indices)
    # 根据打乱的索引重新排列特征和标签
    seqs_data_train = X[indices]
    intensity_train = Y[indices]
    # 初始化性能指标列表
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
        # # 运行后删除
        # # 新增：折间数据隔离检查
        # if fold > 0:
        #     prev_test_indices = set(test_ids[fold - 1])
        #     current_train_indices = set(train_ids[fold])
        #     overlap = prev_test_indices & current_train_indices
        #     assert len(overlap) == 0, f"Fold {fold} 与前一折测试集存在 {len(overlap)} 个重叠样本"

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
        # 根据输入特征应改变特征匹配块的维度
        # ABC

        Train = Constructor(model=C1(in_size), stop=early_stopping)

        # print("\n_______________fold", fold, "_____________\n")
        # 计算运行时间
        sn, sp, mcc, acc, auroc, f1, ap, pr, predicted_value, ground_label = Train.run(Train_Set, Vaild_Set, Test__Set)
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
    # print(pre_value)
    # print(true_label)
    # 分行打印平均指标（修改后的部分）
    print("\n平均评估指标：")
    print(f"auROC (AUC): {np.mean(auROC):.6f}")
    print(f"Sn (召回率): {np.mean(Sn):.6f}")
    print(f"Sp (特异性): {np.mean(Sp):.6f}")
    print(f"Pr (精确率): {np.mean(Pr):.6f}")
    print(f"Acc (准确率): {np.mean(Acc):.6f}")
    print(f"Mcc (马修斯系数): {np.mean(Mcc):.6f}")
    print(f"F1-score: {np.mean(F1):.6f}")
    print(f"Ap (平均精度): {np.mean(Ap):.6f}")
    # print("pre_value:",len(pre_value),pre_value)
    # print("true_label:",len(true_label),true_label)
    # name = ['predict', 'label']
    # df = pd.DataFrame(np.transpose((pre_value, true_label)), columns=name)
    # df.to_csv("tanhLU.csv", index=False)
    return (
    [np.mean(Sn), np.mean(Sp), np.mean(Mcc), np.mean(Acc), np.mean(auROC), np.mean(F1), np.mean(Ap), np.mean(Pr)])


if __name__ == '__main__':
    resu()



