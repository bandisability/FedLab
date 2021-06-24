"""
functions associated with data and dataset operations

暂时放着，未修改

work list
1. random_cutting(dataset, num_clients, file_path)
    将dataset随机切分为num_clients份，存到file_path
2. noniid_cutting(dataset, num_clients, file_path)
    将dataset随机切分为num_clients份，存到file_path
3. 实现上述函数切割的文件的读取和相应dataset的创建，目前仅考虑分类数据即cifar10和mnist等数据集（参考torch.utils.data.Dataset类的重写）
    
实现参考思路：
    *** 1.完整数据集的切割、存储、读取
    *   2.数据集index的切割、index存储、index映射到完整数据集的读取
"""

import warnings
import numpy as np

def noniid_slicing(dataset, num_clients, num_shards):
    """
    将dataset划分为每块大小为num_shards的非独立同分布的块
    按块数量平均分配给num_clients数量的参与者

    返回 各参与者数据集在dataset对应的索引表
    """
    size_of_shards = int(len(dataset)/num_shards)
    if len(dataset) % num_shards != 0:
        warnings.warn(
            "warning: the length of dataset isn't divided exactly by num_shard.some samples will be wasted.")
    # the number of shards that each one of clients can get
    shard_pc = int(num_shards/num_clients)
    if num_shards % num_clients != 0:
        warnings.warn(
            "warning: num_shard isn't divided exacly by num_clients. some samples will be wasted.")

    dict_users = {i: np.array([], dtype='int64') for i in range(num_clients)}

    labels = np.array(dataset.targets)
    idxs = np.arange(len(labels))

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]  # 将标签按索引排序，调换顺序
    idxs = idxs_labels[0, :]

    # assign
    idx_shard = [i for i in range(num_shards)]
    for i in range(num_clients):
        rand_set = set(np.random.choice(idx_shard, shard_pc, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*size_of_shards:(rand+1)*size_of_shards]), axis=0)
    
    return dict_users

def random_slicing(dataset, num_clients):
    """
    将dataset随机划分分配给num_clients数量的参与者
    返回 各参与者数据集在dataset对应的索引表
    """
    num_items = int(len(dataset)/num_clients)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_clients):
        dict_users[i] = set(np.random.choice(
            all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def divide_dataset(slicing_dict, dataset_train):
    """
    返回数据集分割后的元数据组
    """
    datasets = []
    data = dataset_train.data
    label = np.array(dataset_train.targets)
    for _, dic in slicing_dict.items():
        dic = np.array(list(dic))
        client_data = data[dic]
        client_label = list(label[dic])
        client_dataset = (client_data, client_label)
        datasets.append(client_dataset)
    return datasets