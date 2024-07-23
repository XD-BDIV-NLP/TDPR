import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import csv
from collections import defaultdict
from sklearn.decomposition import PCA
from kmeans_pytorch import kmeans, kmeans_predict


class KMeans:
    def __init__(self, train_cls_path, test_cls_path, dev_cls_path,
                 num_clusters, output_path, test_save_path, dev_save_path) -> None:
        
        # 获取value
        self.train_cls_value = list(np.load(train_cls_path, allow_pickle=True).item().values())
        self.test_cls_value = list(np.load(test_cls_path, allow_pickle=True).item().values())
        self.dev_cls_value = list(np.load(dev_cls_path, allow_pickle=True).item().values())

        #
        self.new_train_cls = None
        self.new_test_cls = None
        self.new_dev_cls = None

        #
        self.stacked_train_vector = None
        self.stacked_test_vector = None
        self.stacked_dev_vector = None

        #default cluster value
        self.num_clusters = num_clusters
        self.device = None

        #
        self.clusters_ids_x = None
        self.clusters_centers = None

        #
        self.test_list = []
        self.dev_list = []

        #
        self.output_path = output_path
        self.test_save_path = test_save_path
        self.dev_save_path = dev_save_path

    def set_np_seed(self, seed):
        np.random.seed(seed=seed)


    def new_axis(cls):
        new_cls = []
        for vector in cls:
            vector = vector[np.newaxis, :, :]
            new_cls.append(vector)
        return new_cls

    def axis_process(self):
        # 添加维度
        self.new_train_cls = self.new_axis(self.train_cls_value)
        self.new_test_cls = self.new_axis(self.test_cls_value)
        self.new_dev_cls = self.new_axis(self.dev_cls_value)

        # 在第一个维度折叠
        self.stacked_train_vector = torch.from_numpy(np.concatenate(self.new_train_cls, axis=0))
        self.stacked_test_vector = torch.from_numpy(np.concatenate(self.new_test_cls, axis=0))
        self.stacked_dev_vector = torch.from_numpy(np.concatenate(self.new_dev_cls, axis=0))

    # 自行设置num_clusters
    def set_clusters_num(self, num):
        self.num_clusters = num

    def set_device(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

    def kmeans_and_save(self):
        self.clusters_ids_x, self.clusters_centers = kmeans(
            X=self.stacked_train_vector, num_clusters=self.num_clusters, distance='euclidean', device=self.device
        )
        with open(self.output_path, 'wb') as file:
            pickle.dump(self.clusters_ids_x, file=file)

    def assign_to_clusters(self, new_data):
        # 计算数据点到聚类中心的欧式距离
        distance = torch.cdist(new_data, self.clusters_centers)
        # 找到每个数据点最近的聚类中心的索引
        cluster_assignments = torch.argmin(distance, dim=0)

        return cluster_assignments

    def get_dev_and_test_list(self):
        for i in range(len(self.stacked_test_vector)):
            cluster_assignments = self.assign_to_clusters(self.stacked_test_vector[i])
            self.test_list.append(cluster_assignments[0][0].item())

        for j in range(len(self.stacked_dev_vector)):
            cluster_assignments = self.assign_to_clusters(self.stacked_dev_vector[j])
            self.dev_list.append(cluster_assignments[0][0].item())

    def save_cluster_list(self):
        # 对于test和dev需要传入不同的path
        with open(self.test_save_path, 'wb') as file_test:
            pickle.dump(self.test_list, file=file_test)
        with open(self.dev_save_path, 'wb') as file_dev:
            pickle.dump(self.dev_list, file=file_dev)
