import json
import os

import numpy as np
import torch


def data_load_coast(dataset, has_v=True, has_a=True, has_t=True):
    dir_str = "./Data/" + dataset
    with open(os.path.join(dir_str, "coast_meta.json")) as f:
        meta = json.load(f)

    num_user = int(meta["num_user"])
    num_item = int(meta["num_item"])
    num_warm_item = int(meta["num_warm_item"])

    train_data = np.load(dir_str + "/train.npy", allow_pickle=True)
    val_data = np.load(dir_str + "/val_full.npy", allow_pickle=True)
    val_warm_data = np.load(dir_str + "/val_warm.npy", allow_pickle=True)
    val_cold_data = np.load(dir_str + "/val_cold.npy", allow_pickle=True)
    test_data = np.load(dir_str + "/test_full.npy", allow_pickle=True)
    test_warm_data = np.load(dir_str + "/test_warm.npy", allow_pickle=True)
    test_cold_data = np.load(dir_str + "/test_cold.npy", allow_pickle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v_feat = torch.tensor(np.load(dir_str + "/feat_v.npy", allow_pickle=True), dtype=torch.float).to(device)
    a_feat = None
    t_feat = None

    return (
        num_user,
        num_item,
        num_warm_item,
        train_data,
        val_data,
        val_warm_data,
        val_cold_data,
        test_data,
        test_warm_data,
        test_cold_data,
        v_feat,
        a_feat,
        t_feat,
    )
