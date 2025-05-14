import random
import numpy as np
import torch
import torch_geometric.utils
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch_scatter import scatter_add
import torch.nn.functional as F

from modelsPre import get_pre_model
from train import train
from evaluation import test
from utilsDiff import oursDiff

def fsdcap(args, data, x, missing_feature_mask, num_classes,device):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    propagation_model = FsdCap(args=args,num_classes = num_classes)
    return propagation_model.propagate(args, data, x, missing_feature_mask, num_classes,device)

class FsdCap(torch.nn.Module):
    def __init__(self, args,num_classes):
        super(FsdCap, self).__init__()
        self.num_iterations = args.num_iterations
        self.T = args.T
        self.lamda = args.lamda
        self.gamma = args.gamma
        self.num_classes = num_classes

    def propagate(self,args, data, x, missing_feature_mask, num_classes,device) -> Tensor:  

        filled_features = self.stage1(x,data.edge_index,missing_feature_mask,args.mask_type)

        pre_out_best,max_values,sorted_trainset = self.pre_GNN(args, data, x, missing_feature_mask, num_classes, device,filled_features)

        opt_feature =self.stage2(args,missing_feature_mask,pre_out_best,filled_features,sorted_trainset,max_values,device)

        return opt_feature

    
    def pre_GNN(self,args, data, x, missing_feature_mask, num_classes, device,filled_features):
        pre_out_best = data.y.clone()
        feature_num = data.num_features

        pre_model = get_pre_model(
                model_name=args.model,
                num_features=feature_num,
                num_classes=num_classes,
                edge_index=data.edge_index,
                x=x,
                mask=missing_feature_mask,
                args=args,
            ).to(device)

        params = list(pre_model.parameters())
            
        optimizer = torch.optim.Adam(params, lr=args.pre_lr)
        criterion = torch.nn.NLLLoss()
        pre_test_acc = 0
        pre_val_accs = []
            

        for epoch in range(0, args.epochs):
            pre_x = filled_features
            
            train(pre_model, pre_x, data, optimizer, criterion)
            
            (_, pre_val_acc, _), pre_out = test(pre_model, pre_x, data, evaluator=None,  device=device, )
            if epoch == 0 or pre_val_acc > max(pre_val_accs):
                max_values,pre_out = pre_out.max(dim=1)
                pre_out_best[data.test_mask] = pre_out[data.test_mask]  
                
            pre_val_accs.append(pre_val_acc)
            if epoch > args.patience and max(pre_val_accs[-args.patience :]) <= max(pre_val_accs[: -args.patience]):
                break

        pred_x,pred_y = pre_model.pred_y.max(dim=1)
        pred_x = pred_x 
        max_values = pred_x.unsqueeze(1)
        sorted_trainset = oursDiff(data,pre_out_best,device)
        
        return pre_out_best,max_values,sorted_trainset

    
    def stage2(self,args, mask, label, embedding,node_difficulties,max_values, device):
        node_difficulties = node_difficulties.to(device)
        weights = node_difficulties  
        normalized_embedding = embedding * weights.unsqueeze(1)
        classes = label.unique()
        class_features = {}

        for c in classes:     
            class_nodes = torch.nonzero(label == c).squeeze(1)
            class_weight = weights.index_select(0, class_nodes).sum(dim=0)    
            node_features = normalized_embedding.index_select(0, class_nodes)
            class_feature = node_features.sum(dim=0) / class_weight
            class_features[c.item()] = class_feature

        class_embedding = torch.zeros_like(embedding)   
        
        for c in classes:  
            class_nodes = torch.nonzero(label == c).squeeze(1)   

            class_embedding[class_nodes] = class_features[c.item()]  

        if torch.isnan(class_embedding).any():
            class_embedding = torch.where(torch.isnan(class_embedding), torch.zeros_like(class_embedding), class_embedding)
            
        embedding_new = max_values * embedding + (1-max_values) *   class_embedding    
 
        embedding = embedding_new * (~mask) + embedding * mask

        return embedding

    
    def compute_edge_weight_c2(self, edge_index, edge_weight_matrix, nv):
        row, col = edge_index[0], edge_index[1]

        edge_weight_c = torch.tensor(edge_weight_matrix[row, col])  
        deg_W = scatter_add(edge_weight_c, row, dim_size=nv)
        deg_W_inv = deg_W.pow_(-1.0)
        deg_W_inv.masked_fill_(deg_W_inv == float("inf"), 0)
        A_Dinv = edge_weight_c * deg_W_inv[row]  
        adj = torch.sparse_coo_tensor(edge_index, values=A_Dinv, size=[nv, nv]).to(edge_index.device)  

        return adj

    def stage1(self, x: Tensor, edge_index: Adj, mask: Tensor, mask_type: str) -> Tensor:
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        nv = x.shape[0] 
        feat_dim = x.shape[1] 
       
        if mask is not None:
            out = torch.zeros_like(x)
            out[mask] = x[mask] 
        
        out_prev = out.clone()
        
        if mask_type == 'structural':
            v_0 = torch.nonzero(mask[:, 0]).view(-1)
            v_0_to_now = v_0
            hop = 1
            count = 0

            old_len = -19999
            while True:
                
                subgraph_nodes, subgraph_edges, _, _ = torch_geometric.utils.k_hop_subgraph(v_0_to_now, hop, edge_index, num_nodes=nv)
                v_unknown = torch.from_numpy(np.setdiff1d(torch.arange(nv), v_0_to_now.cpu())).to(v_0.device)

                new_len = len(v_unknown)
                if new_len == old_len:
                    break
                old_len =  new_len

                

                adj_matrix = torch.sparse_coo_tensor(subgraph_edges, torch.ones(subgraph_edges.shape[1], device=edge_index.device), size=(nv, nv)).coalesce()
                adj_c = normalize_adj_matrix(adj_matrix)  
               
                adj_c = adj_c ** self.gamma
                adj_c = adj_c.to_dense()
                adj_c = self.compute_edge_weight_c2(subgraph_edges, adj_c, nv) 
               
                for _ in range(self.num_iterations):
                    out = torch.sparse.mm(adj_c, out)
                    out[subgraph_nodes] = out[subgraph_nodes]+ self.lamda * out_prev[subgraph_nodes] 
                    out[mask] = x[mask]  
                    
                
                out_prev[subgraph_nodes] = out[subgraph_nodes]
                
                out_prev[subgraph_nodes] = out_prev[subgraph_nodes] / torch.norm(out_prev[subgraph_nodes])
                
                v_0_to_now = subgraph_nodes 
                hop += 1
                count += 1
           

        else:
            for i in range(feat_dim):
                v_0 = torch.nonzero(mask[:,i]).view(-1) 
                v_0_to_now = v_0
                hop = 1

                old_len = -19999
                while True:
                    
                    subgraph_nodes, subgraph_edges, _, _ = torch_geometric.utils.k_hop_subgraph(v_0_to_now, hop, edge_index, num_nodes=nv)
                    
                    v_unknown = torch.from_numpy(np.setdiff1d(torch.arange(nv), v_0_to_now.cpu())).to(v_0.device)
                    
                    new_len = len(v_unknown)
                    if new_len == old_len:
                        break
                    old_len =  new_len

                    
                    adj_matrix = torch.sparse_coo_tensor(subgraph_edges, torch.ones(subgraph_edges.shape[1], device=edge_index.device), size=(nv, nv)).coalesce()
                    adj_c = normalize_adj_matrix(adj_matrix)  
                    
                    adj_c = adj_c ** self.gamma
                    adj_c = adj_c.to_dense()
                    adj_c = self.compute_edge_weight_c2(subgraph_edges, adj_c, nv) 
                
                    
                    for _ in range(self.num_iterations):
                        out[:, i] = torch.sparse.mm(adj_c, out[:, i].reshape(-1, 1)).reshape(-1)
                        out[subgraph_nodes,i] = out[subgraph_nodes,i]+ self.lamda * out_prev[subgraph_nodes,i]
                        out[mask[:, i], i] = x[mask[:, i], i]  
                    
                    
                    out_prev[subgraph_nodes,i] = out[subgraph_nodes,i]
                    
                    norm = torch.norm(out_prev[subgraph_nodes, i], p=2, keepdim=True)
                    if norm > 0:
                        out_prev[subgraph_nodes, i] = out_prev[subgraph_nodes, i] / norm

                    v_0_to_now = subgraph_nodes 

                    hop += 1
                
        return out 
    
    
def normalize_adj_matrix(adj_matrix: torch.Tensor) -> torch.Tensor:
    deg_inv_sqrt = torch.sparse.sum(adj_matrix, dim=1).to_dense().pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    adj_matrix = adj_matrix * deg_inv_sqrt.view(-1, 1) * deg_inv_sqrt.view(1, -1)
    return adj_matrix.coalesce()

