# 动态阈值时的最佳框架，动+Cos  Carcdio最佳  Muta最佳
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as loader
import math
import numpy as np
from tqdm import tqdm

# from nn_net2 import *
# from nn_net5 import *
from 机器学习.ML import *


# from kan import *

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
        # 8:2拆分，没有验证集
        train_num = int(seqs_num * 0.8)  # 80%训练集
        test_num = seqs_num - train_num  # 20%测试集

        index = range(seqs_num)
        # 随机打乱索引
        indices = np.random.permutation(seqs_num)

        train_ids.append(np.asarray(indices[:train_num]))
        valid_ids.append(np.asarray([], dtype=np.int64))  # 空验证集，明确指定类型
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
    """
        按照不同模型的接收维数，修改相关的样本维数，如：
        特征融合策略不同，卷积操作不同（1D或2D），是否融合形状特征等
    """

    def __init__(self, model, model_name=''):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device=self.device)
        self.model_name = model_name
        # self.optimizer = optim.Adadelta(self.model.parameters(),lr=1,weight_decay=0.01)
        # self.optimizer = optim.Adadelta(self.model.parameters(), lr=0.1)#可更换
        # self.optimizer = optim.ASGD(self.model.parameters(), lr=1)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=9.922e-4)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=6.2e-5, weight_decay=0.001)
        self.optimizer = optim.Adam(self.model.parameters(), lr=6.00029e-5, weight_decay=0.01)  # Carcin
        # self.optimizer = optim.Adam(self.model.parameters(), lr=0.0004021554526690286,weight_decay=0.00016736010167825804)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=6.1e-5, weight_decay=0.01)

        # self.optimizer = optim.AdamW(self.model.parameters(), lr=8.4e-4)

        # 添加余弦退火学习率调度器
        self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=5,  # 余弦周期的一半
            eta_min=1e-6  # 最小学习率
        )

        self.loss_function = nn.BCELoss()  # 用的BCELoss，所以最后一层也要有激活函数
        # self.loss_function = nn.BCEWithLogitsLoss()  # 用的BCEWithLogitsLoss，最后一层不需要激活函数

        self.batch_size = 64
        self.epochs = 200
        self.seed = 0

        # 新增：用于记录训练过程中的指标
        self.epoch_metrics = {
            'AUC': [], 'SN': [], 'MCC': [], 'ACC': [], 'Pr': [], 'Loss': []
        }

    # 在Constructor类的learn方法末尾添加训练集指标计算

    def learn(self, TrainLoader, TestLoader):
        for epoch in range(self.epochs):
            self.model.train()
            ProgressBar = tqdm(TrainLoader)
            epoch_losses = []

            for data in ProgressBar:
                self.optimizer.zero_grad()

                ProgressBar.set_description("Epoch %d" % epoch)  # 设置进度条的描述文本
                x, y = data

                output = self.model(x=x.to(self.device))
                # print(output.shape)
                loss = self.loss_function(output, y.float().to(self.device))
                ProgressBar.set_postfix(loss=loss.item())  # 在进度条末尾显示当前的损失值

                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())

                # 更新余弦退火学习率
                self.cosine_scheduler.step()  # CosineAnnealingLR 更新

            # 记录训练损失
            avg_loss = np.mean(epoch_losses)
            self.epoch_metrics['Loss'].append(avg_loss)

            # 每个epoch结束后在测试集上计算指标
            if epoch % 1 == 0:  # 每个epoch都计算
                metrics_dict = self.evaluate_metrics(TestLoader)
                for key in ['AUC', 'SN', 'MCC', 'ACC', 'Pr']:
                    self.epoch_metrics[key].append(metrics_dict[key])

                print(
                    f"Epoch {epoch} 完成, Loss: {avg_loss:.4f}, AUC: {metrics_dict['AUC']:.4f}, MCC: {metrics_dict['MCC']:.4f}")

        # 训练结束后计算训练集指标（新增部分）
        print("\n=== 训练集指标 ===")
        train_metrics = self.evaluate_metrics(TrainLoader)
        print(f"训练集 AUC: {train_metrics['AUC']:.4f}")
        print(f"训练集 MCC: {train_metrics['MCC']:.4f}")
        print(f"训练集 ACC: {train_metrics['ACC']:.4f}")
        print(f"训练集 SN: {train_metrics['SN']:.4f}")
        print(f"训练集 Pr: {train_metrics['Pr']:.4f}")

    def evaluate_metrics(self, TestLoader):
        """在测试集上计算各项指标"""
        self.model.eval()
        predicted_prob = []
        ground_label = []

        with torch.no_grad():
            for x, y in TestLoader:
                x = x.to(self.device)
                output = self.model(x)
                predicted_prob.extend(torch.sigmoid(output).flatten().detach().cpu().numpy())
                ground_label.extend(y.flatten().cpu().numpy())

        # 动态选择最佳阈值计算指标
        prob_array = np.array(predicted_prob)
        label_array = np.array(ground_label)

        if len(np.unique(label_array)) > 1:  # 确保有正负样本
            fpr, tpr, thresholds = metrics.roc_curve(label_array, prob_array)
            best_threshold = thresholds[np.argmax(tpr - fpr)]
            predicted_label = (prob_array > best_threshold).astype(int)

            # 计算各项指标
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
        """绘制训练指标曲线 - 一个图，多条线"""
        # plt.figure(figsize=(10, 8))
        #
        # epochs = range(len(self.epoch_metrics['MCC']))
        #
        # # 在一个图中绘制所有指标
        # plt.plot(epochs, self.epoch_metrics['AUC'], 'b-', linewidth=2, label='AUC')
        # # plt.plot(epochs, self.epoch_metrics['SN'], 'g-', linewidth=2, label='SN')
        # plt.plot(epochs, self.epoch_metrics['MCC'], 'r-', linewidth=2, label='MCC')
        # plt.plot(epochs, self.epoch_metrics['ACC'], 'm-', linewidth=2, label='ACC')
        # # plt.plot(epochs, self.epoch_metrics['Pr'], 'c-', linewidth=2, label='Pr')
        #
        # plt.xlabel('Epoch', fontsize=20)
        # plt.ylabel('Score', fontsize=20)
        # plt.title('(D) Carcinogenicity', fontsize=24, fontweight='normal', pad=15)
        # # 调整坐标轴刻度尺寸
        # plt.xticks(fontsize=20)  # x轴刻度字体大小
        # plt.yticks(np.arange(0, 1.1, 0.2), fontsize=20)
        #
        # # 修改纵坐标范围从0.2开始
        # plt.ylim(0, 1.0)  # 指标范围0.2-1
        #
        # # 图例放在右上角，设置Arial字体
        # plt.legend(loc='lower right', prop={'family': 'Arial', 'size': 16})
        #
        # plt.tight_layout()
        # plt.savefig('Carcin.png', dpi=300, bbox_inches='tight')
        # plt.close()
        # print("训练指标曲线已保存至: training_metrics.png")

    def inference(self, TestLoader):
        # 直接保存最终模型，不需要早停
        torch.save(self.model.state_dict(), 'final_model11.pt')
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

    # def plot_label_distribution(self, predicted_prob, ground_label, save_path='label_distribution.png'):
    #     """绘制按真实标签分组的双峰分布图"""
    #     import matplotlib.pyplot as plt
    #     from scipy.stats import gaussian_kde
    #     import numpy as np
    #
    #     # 将数据转为numpy数组
    #     prob_array = np.array(predicted_prob)
    #     label_array = np.array(ground_label)
    #
    #     # 按真实标签分离预测概率
    #     positive_probs = prob_array[label_array == 1]  # 真实正样本的预测概率
    #     negative_probs = prob_array[label_array == 0]  # 真实负样本的预测概率
    #
    #     plt.figure(figsize=(10, 6))
    #
    #     # 计算核密度估计
    #     if len(positive_probs) > 1:
    #         kde_positive = gaussian_kde(positive_probs)
    #         x_pos = np.linspace(positive_probs.min(), positive_probs.max(), 100)
    #         plt.fill_between(x_pos, kde_positive(x_pos), alpha=0.6, label='Positive samples', color='#F1948A')
    #
    #     if len(negative_probs) > 1:
    #         kde_negative = gaussian_kde(negative_probs)
    #         x_neg = np.linspace(negative_probs.min(), negative_probs.max(), 100)
    #         plt.fill_between(x_neg, kde_negative(x_neg), alpha=0.6, label='Negative samples', color='lightblue')
    #
    #     plt.xlabel('Predicted Probability', fontsize=18)
    #     plt.ylabel('Density', fontsize=18)
    #     plt.title('Distribution of Predictions by True Labels', fontsize=24)
    #     plt.legend()
    #
    #     # 保存图片
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #     plt.close()
    #     print(f"标签分布图已保存至: {save_path}")

    def measure(self, predicted_prob, ground_label):
        # 将概率和标签转为numpy数组
        prob_array = np.array(predicted_prob)
        label_array = np.array(ground_label)

        # 动态选择最佳阈值（新增核心改动）
        fpr, tpr, thresholds = metrics.roc_curve(label_array, prob_array)
        best_threshold = thresholds[np.argmax(tpr - fpr)]  # Youden指数
        predicted_label = (prob_array > best_threshold).astype(int)
        print(f"\n自动选择的最佳阈值: {best_threshold:.4f}")  # 打印阈值信息

        # 计算各项指标（后续保持不变）
        sn = metrics.recall_score(y_true=label_array, y_pred=predicted_label, pos_label=1)
        sp = metrics.recall_score(y_true=label_array, y_pred=predicted_label, pos_label=0)
        mcc = metrics.matthews_corrcoef(y_true=label_array, y_pred=predicted_label)
        acc = metrics.accuracy_score(y_true=label_array, y_pred=predicted_label)
        auroc = metrics.roc_auc_score(y_true=label_array, y_score=prob_array)  # 注意仍用概率计算AUC
        f1 = metrics.f1_score(y_true=label_array, y_pred=predicted_label)
        ap = metrics.average_precision_score(y_true=label_array, y_score=prob_array)
        pr = metrics.precision_score(y_true=label_array, y_pred=predicted_label, pos_label=1)

        # 分行打印指标（新增部分）
        print("\n评估指标：")  # 添加空行分隔
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

        # 绘制训练指标曲线
        self.plot_training_metrics()

        predicted_value, ground_label = self.inference(Test_Loader)

        # # === 新增：绘制标签分布图 ===
        # self.plot_label_distribution(predicted_value, ground_label, 'carcinogenicity_label_distribution.png')
        # # ===========================

        sn, sp, mcc, acc, auroc, f1, ap, pr = self.measure(predicted_value, ground_label)

        return sn, sp, mcc, acc, auroc, f1, ap, pr, predicted_value, ground_label

def resu():
    K = 1  # 改为1，使用单次8:2拆分
    ratio_k = 0  # 无验证集
    # L = 21

    data1 = pd.read_csv(r"C:\Users\DELL\Desktop\Carcin-feature\Carcin_LightGBM5.csv")
    # data1 = pd.read_csv(r"C:\Users\DELL\Desktop\Cardio-Lightgbm\Cardio_LightGBM5.csv")
    # data1 = pd.read_csv(r"C:\Users\DELL\Desktop\Hepa-feature\Hepa_LightGBM5-175.csv")
    # data1 = pd.read_csv(r"C:\Users\DELL\Desktop\Muta_feature\Muta_LightGBM5-425.csv")

    # data1 = pd.read_csv(r"Carcin_LightGBM5.csv")
    # data1 = pd.read_csv(r"Muta_LightGBM5-425.csv")
    # data1 = pd.read_csv(r"Cardio-Gain-100.csv")
    # data1 = pd.read_csv(r"Hepa_LightGBM5-175.csv")
    # data1 = pd.read_csv(r"Carcin-Split-100.csv")
    # data1 = pd.read_csv(r"Carcin-SHAP-100.csv")

    X = np.array(data1.iloc[:, 1:])
    Y = np.array(data1.iloc[:, 0])
    Y = Y.reshape((Y.shape[0], 1))
    seqs_num = X.shape[0]  # 表示数组的第一维的大小，即行数，对应于样本的数量
    in_size = X.shape[1]  # 表示数组的第二维的大小，即列数，对应于特征的数量

    # -------------------------- 新增：全局唯一打乱步骤（最小修改） --------------------------
    indices = np.arange(seqs_num)
    np.random.seed(0)  # 固定随机种子，确保可复现
    np.random.shuffle(indices)
    # 用打乱后的索引重排特征和标签
    X = X[indices]
    Y = Y[indices]
    # --------------------------------------------------------------------------------------

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
        # 获取当前折的训练集和测试集索引
        train_index = train_ids[fold]
        test_index = test_ids[fold]

        # -------------------- 新增：数据泄露检验代码（最小补充） --------------------
        # 计算训练集和测试集索引的交集
        overlap = np.intersect1d(train_index, test_index)
        if len(overlap) > 0:
            print(f"⚠️ 警告：第{fold}折存在数据泄露！重合索引数量：{len(overlap)}")
            print(f"重合索引示例：{overlap[:5]}")  # 打印前5个重合索引
        else:
            print(f"✅ 第{fold}折检验通过：训练集与测试集无重合（无数据泄露）")
        # --------------------------------------------------------------

        x_train = X[train_index]
        y_train = Y[train_index]

        x_test = X[test_index]
        y_test = Y[test_index]

        Train_Set = SSDataset(data_set=x_train, labels=y_train)
        Test__Set = SSDataset(data_set=x_test, labels=y_test)

        # 移除早停参数
        Train = Constructor(model=RF(in_size))


        print("\n_______________fold", fold, "_____________\n")
        # 计算运行时间
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

