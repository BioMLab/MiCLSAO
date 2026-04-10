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

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )

