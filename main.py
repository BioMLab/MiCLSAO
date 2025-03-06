import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from arguments import arg_parse
from kan import KAN  
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

def create_graph_data(similarity_matrix):
    edge_index = []
    edge_weight = []
    num_nodes = similarity_matrix.shape[0]

    for i in range(num_nodes):
        for j in range(num_nodes):
            if similarity_matrix[i, j] >= args.T: 
                edge_index.append([i, j])
                edge_weight.append(similarity_matrix[i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_weight)
    return data

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class LncRNAGO_Dataset(Dataset):
    def __init__(self, pairs_array):
        super().__init__()
        self.data = pairs_array  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        lnc_i = self.data[idx, 0]
        go_j = self.data[idx, 1]
        label = self.data[idx, 2]
        return (lnc_i, go_j, label)


def main():
    args = arg_parse()
    setup_seed(args.seed)
    paths = args.lncRNA_paths
    device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    epochs = args.epochs
    data_views = []
    for path in paths:
        sim_mat = pd.read_csv(path, header=None).values
        data = create_graph_data(sim_mat)
        data_views.append(data)
    A_num_features = data_views[0].x.shape[1]
    mvcc_model = MVCCL(
        A_num_features=A_num_features,
        hidden_dim=args.hidden_dim,
        num_gc_layers=args.num_gc_layers,
        pooling=args.pooling
    ).to(device)
    go_features = pd.read_csv(args.go_features_path, header=None).values
    go_features_torch = torch.tensor(go_features, dtype=torch.float32, device=device)
    go_dim = go_features_torch.size(1)  
    l_g_association = pd.read_csv(args.l_g_association_path, header=None).values
    num_lnc = l_g_association.shape[0]  
    num_go = l_g_association.shape[1]   
    pos_pairs = []
    for i in range(num_lnc):
        for j in range(num_go):
            if l_g_association[i, j] == 1:
                pos_pairs.append((i, j, 1))
    neg_candidates = []
    for i in range(num_lnc):
        for j in range(num_go):
            if l_g_association[i, j] == 0:
                neg_candidates.append((i, j, 0))
    random.shuffle(neg_candidates)
    neg_pairs = neg_candidates[:len(pos_pairs)]

    pairs_all = pos_pairs + neg_pairs
    random.shuffle(pairs_all)
    pairs_array = np.array(pairs_all, dtype=np.long)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    fold_idx = 0
    for train_idx, val_idx in kfold.split(pairs_array):
        fold_idx += 1
        print(f"\n=== Fold {fold_idx} ===")
        train_pairs = pairs_array[train_idx]
        val_pairs   = pairs_array[val_idx]

        train_dataset = LncRNAGO_Dataset(train_pairs)
        val_dataset   = LncRNAGO_Dataset(val_pairs)

        train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)

        mvcc_model = MVCCL(
            A_num_features=A_num_features,
            hidden_dim=args.hidden_dim,
            num_gc_layers=args.num_gc_layers,
            pooling=args.pooling
        ).to(device)

        kan_model = KAN(layers_hidden=[3*args.hidden_dim + go_dim, 64, 2]).to(device)

        end2end_model = LncRNA_GO_End2End(mvcc_model, kan_model).to(device)

        optimizer = Adam(end2end_model.parameters(), lr=args.lr, weight_decay=1e-5)
        epochs = args.epochs
        alpha = args.alpha

        for epoch in range(epochs):
            end2end_model.train()
            train_loss = 0.0
            for batch_data in train_loader:
                lnc_idx_batch, go_idx_batch, label_batch = batch_data
                lnc_idx_batch = lnc_idx_batch.to(device)
                go_idx_batch  = go_idx_batch.to(device)
                label_batch   = label_batch.to(device)

                go_feats_batch = go_features_torch[go_idx_batch]

                optimizer.zero_grad()
                logits, lnc_emb_all = end2end_model(data_views, lnc_idx_batch, go_feats_batch)
                loss_cls = F.binary_cross_entropy_with_logits(logits, label_batch)
                loss_con = nt_xent_loss(lnc_emb_all, batch_size=lnc_emb_all.size(0))

                loss = loss_cls + alpha * loss_con
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            end2end_model.eval()
            val_loss = 0.0
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for batch_data in val_loader:
                    lnc_idx_batch, go_idx_batch, label_batch = batch_data
                    lnc_idx_batch = lnc_idx_batch.to(device)
                    go_idx_batch  = go_idx_batch.to(device)
                    label_batch   = label_batch.to(device)

                    go_feats_batch = go_features_torch[go_idx_batch]
                    logits, lnc_emb_all = end2end_model(data_views, lnc_idx_batch, go_feats_batch)
                    loss_cls = F.binary_cross_entropy_with_logits(logits, label_batch)
                    val_loss += loss_cls.item()

                    preds = logits.argmax(dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(label_batch.cpu().numpy())


if __name__ == "__main__":
    main()