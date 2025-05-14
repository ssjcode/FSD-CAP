import torch_geometric.utils
from torch_geometric.utils import add_self_loops ,remove_self_loops 
import torch.nn.functional as F
import torch
from torch_geometric.typing import OptTensor
import numpy as np

def oursDiff(data,label,device):

    now_edge_index,_ = remove_self_loops(data.edge_index)
    neighbor_label, _ = add_self_loops(edge_index=now_edge_index.to(device),num_nodes=label.shape[0])
    
    neighbor_label[1] = label[neighbor_label[1]]
    neighbor_label = torch.transpose(neighbor_label, 0, 1) 
    index, count = torch.unique(neighbor_label, sorted=True, return_counts=True, dim=0)
    neighbor_class = torch.sparse_coo_tensor(index.T, count)
    neighbor_class = neighbor_class.to_dense().float()   
    row_sums = neighbor_class.sum(dim=1).unsqueeze(1)
    neighbor_class = neighbor_class
    neighbor_class = F.normalize(neighbor_class, 1.0, 1)                     
    log_10_entropy = torch.log(neighbor_class + torch.exp(torch.tensor(-20))) 
    log_c_entropy  = log_10_entropy / torch.log(row_sums + torch.exp(torch.tensor(-20)) )
    neighbor_entropy = -1 * neighbor_class * log_c_entropy 
    local_difficulty = neighbor_entropy.sum(1)   
    
    return torch.tensor(1.0, dtype=torch.float32).to(device)-local_difficulty.to(device)


def compute_f_n2d(self, edge_index, feature_mask, mask_type, feat_dim: OptTensor = None):
    nv = feature_mask.shape[0] 
    if mask_type == 'structural':
        len_v_0tod_list = []
        f_n2d = torch.zeros(nv, dtype = torch.int)
        v_0 = torch.nonzero(feature_mask[:, 0]).view(-1)
        len_v_0tod_list.append(len(v_0))
        v_0_to_now = v_0 
        f_n2d[v_0] = 0
        d = 1
        while True:
            v_d_hop_sub = torch_geometric.utils.k_hop_subgraph(v_0, d, edge_index, num_nodes=nv)[0] 
            v_d = torch.from_numpy(np.setdiff1d(v_d_hop_sub.cpu(), v_0_to_now.cpu())).to(v_0.device)
            if len(v_d) == 0:
                break
            f_n2d[v_d] = d
            v_0_to_now = torch.cat([v_0_to_now, v_d], dim=0)
            len_v_0tod_list.append(len(v_d))
            d += 1
    else:
        f_n2d = torch.zeros(feat_dim, nv)

        for i in range(feat_dim):
            v_0 = torch.nonzero(feature_mask[:,i]).view(-1) 
            v_0_to_now = v_0
            f_n2d[i, v_0] = 0 
            d=1

            while True: 
                v_d_hop_sub = torch_geometric.utils.k_hop_subgraph(v_0, d, edge_index, num_nodes=nv)[0] 
                v_d = torch.from_numpy(np.setdiff1d(v_d_hop_sub.cpu(), v_0_to_now.cpu())).to(v_0.device) 

                if len(v_d) == 0: 
                    break
                f_n2d[i, v_d] = d 
                v_0_to_now = torch.cat([v_0_to_now, v_d], dim=0) 
                d += 1

    return f_n2d
