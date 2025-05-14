import numpy as np
from tqdm import tqdm
import argparse
import torch
from data_loading import get_dataset
from utils import get_missing_feature_mask
from seeds import seeds
from utils_link import train, test

import random
from torch_geometric.utils import train_test_split_edges
from models_link import GCNEncoder
from torch_geometric.nn import GAE

from fsdcap import fsdcap
from data_utils import set_train_val_test_split


parser = argparse.ArgumentParser("Setting for graphs with partially known features")
parser.add_argument(
    "--mask_type", type=str, help="Type of missing feature mask", default="structural", choices=["uniform", "structural"],
)
parser.add_argument("--gamma", type=float, help="Fractional exponent of A", default=1)
parser.add_argument("--lamda", type=float, help="propertion of prev propFeature to preserve", default=0.1)
parser.add_argument("--T", type=float, default=1)
parser.add_argument("--lr", type=float, help="GAE Learning Rate", default=0.001)
parser.add_argument(
    "--dataset_name",
    type=str,
    help="Name of dataset",
    default="Cora",
    choices=[
        "Cora",
        "CiteSeer",
        "PubMed",
        "Photo",
        "Computers",
    ],
)
parser.add_argument("--pre_dropout", type=float, help="pre GNN Feature dropout", default=0.5)
parser.add_argument("--pre_lr", type=float, help="Pre Gnn Learning Rate", default=0.01)
parser.add_argument("--pre_patience", type=int, help="Patience for early stopping", default=200)
parser.add_argument("--patience", type=int, help="Patience for early stopping", default=200)
parser.add_argument("--dropout", type=float, help="Feature dropout", default=0.5)
parser.add_argument("--model_name", type=str, help="Name of model", default="fsdcap" )
parser.add_argument("--epochs", type=int, help="Max number of epochs", default=100000)
parser.add_argument(
    "--model",
    type=str,
    help="Type of model to make a prediction on the downstream task",
    default="gcn",
    choices=["gcn"],
)
parser.add_argument("--gpu_idx", type=int, help="Indexes of gpu to run program on", default=0)
parser.add_argument("--missing_rate", type=float, help="Rate of node features missing", default=0.995)
parser.add_argument(
    "--num_iterations", type=int, help="Number of diffusion iterations for feature reconstruction", default=100,
)
parser.add_argument("--n_runs", type=int, help="Max number of runs", default=10)
parser.add_argument("--hidden_dim", type=int, help="Hidden dimension of model", default=64)
parser.add_argument("--num_layers", type=int, help="Number of GNN layers", default=3)



def run(args):

    device = torch.device(
        f"cuda:{args.gpu_idx}"
        if torch.cuda.is_available() and not (args.dataset_name == "OGBN-Products" and args.model == "lp")
        else "cpu"
    )

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset, _ = get_dataset(name=args.dataset_name)

    n_nodes, n_features = dataset.data.x.shape
    num_classes = dataset.num_classes
    aucs, aps = [], []
    
    
    for seed in tqdm(seeds[: 1]): #args.n_runs
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        data = dataset.data.clone().to(device)
        data_node = dataset.data.clone().to(device)
        
        missing_feature_mask = get_missing_feature_mask(
            rate=args.missing_rate, n_nodes=n_nodes, n_features=n_features, seed = seed, type=args.mask_type,
        ).to(device)

        data = train_test_split_edges(data, val_ratio=0.05, test_ratio=0.1).to(device)
        
        data_node = set_train_val_test_split( 
            seed=seed, data=data_node, split_idx=None, dataset_name=args.dataset_name,
        ).to(device)
        
        data_node.edge_index = data.train_pos_edge_index.clone()
        
        x = data_node.x.clone()
        x[~missing_feature_mask] = float("nan")


        filled_features = fsdcap(args, data_node, x, missing_feature_mask, num_classes,device)   

        #属性
        x_end = torch.where(missing_feature_mask, data.x, filled_features)

        model = GAE(GCNEncoder(dataset.num_features, out_channels=16))
        model.to(device)
        train_pos_edge_index = data.train_pos_edge_index
        optimizer = torch.optim.Adam(model.parameters(), lr= args.lr)

        val_aucs = []
        val_aps = []
        test_auc = 0 
        test_ap = 0

        for epoch in range(0, 201): #args.epochs
            loss = train(model, x_end, train_pos_edge_index, optimizer)
            val_auc, val_ap = test(model, x_end, train_pos_edge_index, data.val_pos_edge_index, data.val_neg_edge_index)
            auc, ap = test(model, x_end, train_pos_edge_index, data.test_pos_edge_index, data.test_neg_edge_index)    
            if epoch == 0 or val_auc > max(val_aucs):
                test_auc = auc
                test_ap = ap

            val_aucs.append(val_auc) 
            val_aps.append(val_ap) 

            if epoch > args.patience and max(val_aucs[-args.patience :]) <= max(val_aucs[: -args.patience]):
                break

            if epoch % 200 == 0:
                print('epoch: {:03d}, AUC: {:.4f}, AP: {:.4f},loss: {:.6f}'.format(epoch, auc, ap,loss))
                           
        aucs.append(test_auc)
        aps.append(test_ap)

    test_auc_mean, test_auc_std = np.mean(aucs), np.std(aucs)
    test_ap_mean, test_ap_std = np.mean(aps), np.std(aps)
    
    str_auc = f"\n=============test_auc: {test_auc_mean * 100:.2f}% +- {test_auc_std * 100:.2f}======================="
    str_ap =  f"\n=============test_ap: {test_ap_mean * 100:.2f}% +- {test_ap_std * 100:.2f}======================="
   
    print(str_auc)
    print(str_ap)


if __name__ == "__main__":
    args = parser.parse_args()
    run(args)  

