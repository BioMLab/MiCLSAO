import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from kan import KAN  
import torch.nn
import torch.nn as nn
from torch_geometric.utils import softmax
from torch_geometric.nn import GCNConv, global_add_pool
from torch.nn import BatchNorm1d, ReLU, Linear,Dropout

class Translator(nn.Module):
    def __init__(self, num_features, dim, num_gc_layers):
        super(Translator, self).__init__()
        self.num_gc_layers = num_gc_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.acts = nn.ModuleList()

        for i in range(self.num_gc_layers):
            if i == 0:
                conv = GCNConv(num_features, dim)
            else:
                conv = GCNConv(dim, dim if i != self.num_gc_layers - 1 else 1)
            bn = BatchNorm1d(dim if i != self.num_gc_layers - 1 else 1)
            act = ReLU()
            
            self.convs.append(conv)
            self.bns.append(bn)
            self.acts.append(act)

    def forward(self, x, edge_index, edge_weight, batch):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(x.device)
        xs = []
        for i in range(self.num_gc_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.bns[i](x)
            if i != self.num_gc_layers - 1:
                x = self.acts[i](x)
            xs.append(x)
        node_prob = xs[-1]
        node_prob = softmax(node_prob / 5.0, batch)

        return node_prob

class Encoder(nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, pooling):
        super(Encoder, self).__init__()
        self.pooling = pooling
        self.dim = dim
        self.fc = nn.Linear(num_features, dim, bias=False)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.acts = nn.ModuleList()

        for i in range(num_gc_layers):
            conv = GCNConv(dim, dim)
            bn = BatchNorm1d(dim)
            act = ReLU()
            self.convs.append(conv)
            self.bns.append(bn)
            self.acts.append(act)

        self.init_parameters()

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, edge_weight, batch):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(x.device)
        x = self.fc(x)  
        xs = []
        for conv, act, bn in zip(self.convs, self.acts, self.bns):
            x = conv(x, edge_index, edge_weight)
            x = act(x)
            x = bn(x)
            xs.append(x)
        xpool = [global_add_pool(x, batch) for x in xs]
        
        if self.pooling == "last":
            x = xpool[-1]
        elif self.pooling == "all":
            x = torch.cat(xpool, 1)
        elif self.pooling == "add":
            x = sum(xpool)
        
        return x, torch.cat(xs, 1)

class MLPHead(nn.Module):
    def __init__(self,in_channels, hidden_dim, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels,hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim,out_channels)
        )
    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.d_k = hidden_dim // num_heads
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1) 
    
    def forward(self, a, b, c):
        b_c_concat = torch.cat([b, c], dim=-1) 
        b_c_proj = self.multihead_attn.in_proj_weight[:self.d_k * 2, :] @ b_c_concat

        attn_output, attn_weights = self.multihead_attn(a, b_c_proj, b_c_proj)
        attn_output = attn_output * self.d_k
        
        attn_output = self.dropout(attn_output)
        return self.layer_norm(attn_output + a)
    
class MVCCL(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, pooling="all"):
        super(MVCCL, self).__init__()

        if pooling == "last":
            self.embedding_dim = hidden_dim
        elif pooling == "all":
            self.embedding_dim = hidden_dim * num_gc_layers
        else:
            self.embedding_dim = hidden_dim

        self.pooling = pooling
        self.translator = Translator(A_num_features, hidden_dim, num_gc_layers)
        self.encoder = Encoder(A_num_features, hidden_dim, num_gc_layers, pooling=self.pooling)
        self.project = nn.Sequential(
            Linear(self.embedding_dim, self.embedding_dim),
            ReLU(),
            Dropout(0.5),
            Linear(self.embedding_dim, self.embedding_dim),
            ReLU()
        )
        self.cross_attention = CrossAttention(hidden_dim)
        self.init_parameters()

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x_a, x_b, x_c, edge_index_a, edge_index_b, edge_index_c, edge_weight_a, edge_weight_b, edge_weight_c):

        batch_a = torch.zeros(x_a.size(0), dtype=torch.long, device=x_a.device)
        batch_b = torch.ones(x_b.size(0), dtype=torch.long, device=x_a.device)
        batch_c = torch.ones(x_c.size(0), dtype=torch.long, device=x_a.device) 

        node_prob_a = self.translator(x_a, edge_index_a, edge_weight_a, batch_a)
        node_prob_b = self.translator(x_b, edge_index_b, edge_weight_b, batch_b)
        node_prob_c = self.translator(x_c, edge_index_c, edge_weight_c, batch_c)

    
        y_a = self.cross_attention(y_a, y_b, y_c) 
        y_b = self.cross_attention(y_b, y_c, y_a)
        y_c = self.cross_attention(y_c, y_a, y_b) 

        y_a = self.project(y_a)
        y_b = self.project(y_b)
        y_c = self.project(y_c)
        lncRNA_emb_all = torch.cat([y_a, y_b, y_c], dim=1)  
        return lncRNA_emb_all

class LncRNA_GO(nn.Module):

    def __init__(self, 
                 lnc_mvccl_model: MVCCL,
                 input_dim_for_kan: int, 
                 kan_layers: list):
        super().__init__()
        self.lnc_mvccl = lnc_mvccl_model  
        self.kan = KAN(kan_layers)
    
    def forward(self, 
                data_views, 
                go_features_batch, 
                lnc_idx_batch,
                device):

        data_a, data_b, data_c = data_views
        lnc_emb_all = self.mvcc_model(data_a, data_b, data_c)
        lnc_emb_batch = lnc_emb_all[lnc_idx_batch]   
        combined = torch.cat([lnc_emb_batch, go_features_batch], dim=1)
        logits = self.kan(combined)  
        return logits, lnc_emb_all  
