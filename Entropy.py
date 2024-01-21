import math

import numpy as np
import pandas as pd
from data_process.ProcessedData import ProcessedData

class Entropy(ProcessedData):

    def __init__(self, raw_data):
        super().__init__(raw_data)
        self.rest_columns = None

    def process(self, components_percent=0.5, eigenvalue_percent=0.5):
        if len(self.label_df) > 1:
            def calculate_entropy(y):
                # 计算熵
                unique, counts = np.unique(y, return_counts=True)
                probabilities = counts / counts.sum()
                entropy = -np.sum(probabilities * np.log2(probabilities))
                return entropy

            def calculate_information_gain(X, y, feature_index):
                # 计算特定特征的信息增益
                total_entropy = calculate_entropy(y)

                # 对特征进行分割并计算条件熵
                values, counts = np.unique(X[:, feature_index], return_counts=True)
                weighted_entropy_sum = 0
                for value, count in zip(values, counts):
                    subset_y = y[X[:, feature_index] == value]
                    weighted_entropy_sum += (count / y.size) * calculate_entropy(subset_y)

                # 信息增益是总熵和条件熵的差
                information_gain = total_entropy - weighted_entropy_sum
                return information_gain

            # 假设 X 是我们要降维的01矩阵，y 是标签
            X = self.feature_df.values  # 特征矩阵
            y = self.label_df.values  # 标签向量
            #print(X)
            # 计算每个特征的信息增益
            information_gains = [calculate_information_gain(X, y, i) for i in range(X.shape[1])]
            print(information_gains)
            # 选择信息增益最高的 N 个特征
            N = math.trunc(self.feature_df.shape[1]*components_percent)  # 假设我们想要选择的特征数
            #print(information_gains)
            contri_index=np.argsort(information_gains)[::-1]
            print(contri_index)
            selected_index = contri_index[:N]
            rest_index=np.sort(contri_index[N:])
            self.rest_columns= self.feature_df.columns[rest_index]
            #print(self.rest_columns)
            # 构建降维后的矩阵
            selected_index=np.sort(selected_index)
            X_reduced = X[:, selected_index]

            #print("Selected features based on information gain:", selected_index)
            #print("Reduced matrix:\n", X_reduced)

            columns = self.feature_df.columns[selected_index]
            self.feature_df  = pd.DataFrame(X_reduced, columns=columns)
            self.data_df = pd.concat([self.feature_df, self.label_df], axis=1)
            #print(self.data_df)



