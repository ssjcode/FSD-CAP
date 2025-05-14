import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, BatchNorm1d, Sequential
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    JumpingKnowledge,
)


def get_pre_model(model_name, num_features, num_classes, edge_index, x, args, mask=None):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return PreGNN(
        num_features=num_features,
        num_classes=num_classes,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        dropout=args.pre_dropout,
        conv_type=model_name,
        T = args.T,
    )  

class PreGNN(torch.nn.Module):
    def __init__(
        self, num_features, num_classes, hidden_dim, num_layers=3, dropout=0, conv_type="GCN", T = 10,jumping_knowledge=False,
    ):
        super(PreGNN, self).__init__()

        self.convs = ModuleList([get_conv(conv_type, num_features, hidden_dim)])
        for _ in range(num_layers - 2):
            self.convs.append(get_conv(conv_type, hidden_dim, hidden_dim))
        output_dim = hidden_dim if jumping_knowledge else num_classes
        self.convs.append(get_conv(conv_type, hidden_dim, output_dim))

        if jumping_knowledge:
            self.lin = Linear(hidden_dim, num_classes)
            self.jump = JumpingKnowledge(mode="max", channels=hidden_dim, num_layers=num_layers)

        self.num_layers = num_layers
        self.dropout = dropout
        self.jumping_knowledge = jumping_knowledge
        self.embedding = 0
        self.T = T
        self.pred_y = 0

    def forward(self, x, edge_index=None, adjs=None, full_batch=True):
        return self.forward_full_batch(x, edge_index) if full_batch else self.forward_sampled(x, adjs)

    def forward_full_batch(self, x, edge_index):
        xs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

            if i != len(self.convs) - 1 or self.jumping_knowledge:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            if i ==len(self.convs) - 2:       
                self.embedding = torch.nn.functional.log_softmax(x, dim=1) 

            xs += [x]



        if self.jumping_knowledge:
            x = self.jump(xs)
            x = self.lin(x)
        self.pred_x = x 
        self.pred_y = torch.nn.functional.softmax(x/self.T, dim=1)
        return torch.nn.functional.log_softmax(x, dim=1)

    def forward_sampled(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[: size[1]]  
            x = self.convs[i]((x, x_target), edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x.log_softmax(dim=1)

    def inference(self, x_all, inference_loader, device):
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in inference_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[: size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)

        return x_all

def get_conv(conv_type, input_dim, output_dim):
    if conv_type == "gcn":
        return GCNConv(input_dim, output_dim)
    elif conv_type == "gat":
        return GATConv(input_dim, output_dim, heads=1)
    else:
        raise ValueError(f"Convolution type {conv_type} not supported")

class GNNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, conv_type):
        super(GNNEncoder, self).__init__()
        if conv_type == "gcn":
            self.conv = GCNConv(in_channels, out_channels, cached=True)
        elif conv_type == "gat":
            self.conv = GATConv(in_channels, out_channels, heads=1)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)

def get_GNN(in_channels, out_channels, conv_type):
    return GNNEncoder(in_channels, out_channels, conv_type)

def get_activation(activation):
    if activation == 'relu':
        return torch.nn.ReLU()
    elif activation == 'prelu':
        return torch.nn.PReLU()
    elif activation == 'tanh':
        return torch.nn.Tanh()
    elif activation == 'sigmoid':
        return torch.nn.Sigmoid()
    elif (activation is None) or (activation == 'none'):
        return torch.nn.Identity()
    else:
        raise NotImplementedError


class MLPNet(torch.nn.Module):
    def __init__(self,
         		input_dims, output_dim,
                hidden_dim=32,
         		hidden_layer_sizes=(64,),
         		hidden_activation='relu',
         		output_activation=None,
                dropout=0.05):
        super(MLPNet, self).__init__()

        layers = ModuleList()
        input_dim = np.sum(input_dims)

        layer = Sequential(
            Linear(input_dim, output_dim),
            get_activation(output_activation),
        )
        layers.append(layer)
        self.layers = layers

    def forward(self, inputs):
        if torch.is_tensor(inputs):
            inputs = [inputs]
        input_var = torch.cat(inputs,-1)
        for layer in self.layers:
            input_var = layer(input_var)
        return input_var