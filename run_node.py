import numpy as np
import argparse
import torch
from data_loading import get_dataset
from data_utils import set_train_val_test_split
from utils import get_missing_feature_mask
from models import get_model
from seeds import seeds
from evaluation import test
from train import train
import random
from mlp import *
from fsdcap import fsdcap
import warnings

parser = argparse.ArgumentParser("Setting for graphs with partially known features")
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
parser.add_argument("--gamma", type=float, help="Fractional exponent of A", default=1)
parser.add_argument("--lamda", type=float, help="propertion of prev propFeature to preserve", default=0.1)
parser.add_argument("--T", type=float, default=10)
parser.add_argument("--lr", type=float, help="Learning Rate", default=0.01)
parser.add_argument("--dropout", type=float, help="Feature dropout", default=0.5)
parser.add_argument("--pre_dropout", type=float, help="pre GNN Feature dropout", default=0.5)
parser.add_argument("--pre_lr", type=float, help="Pre Gnn Learning Rate", default=0.01)
parser.add_argument("--pre_patience", type=int, help="Patience for early stopping", default=200)
parser.add_argument("--gpu_idx", type=int, help="Indexes of gpu to run program on", default=0)
parser.add_argument(
    "--mask_type", type=str, help="Type of missing feature mask", default="structural", choices=["uniform", "structural"],
)
parser.add_argument("--model_name", type=str, help="Name of model", default="fsdcap")
parser.add_argument("--missing_rate", type=float, help="Rate of node features missing", default=0.995)
parser.add_argument(
    "--num_iterations", type=int, help="Number of diffusion iterations for feature reconstruction", default=200,
)
parser.add_argument("--model",    type=str,    default="gcn")
parser.add_argument("--patience", type=int, help="Patience for early stopping", default=200)
parser.add_argument("--epochs", type=int, help="Max number of epochs", default=10000)
parser.add_argument("--n_runs", type=int, help="Max number of runs", default=10)
parser.add_argument("--hidden_dim", type=int, help="Hidden dimension of model", default=64)
parser.add_argument("--num_layers", type=int, help="Number of GNN layers", default=3)


def run(args):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    random.seed(0)

    device = torch.device(
        f"cuda:{args.gpu_idx}"
        if torch.cuda.is_available() else "cpu"
    )

    dataset, evaluator = get_dataset(name=args.dataset_name)                        

    split_idx = dataset.get_idx_split() if hasattr(dataset, "get_idx_split") else None
    
    n_nodes, n_features = dataset.data.x.shape
    
    test_accs = []

    for seed in seeds[ : args.n_runs ]: 
        
        num_classes = dataset.num_classes
        data = set_train_val_test_split( 
            seed=seed, data=dataset.data, split_idx=split_idx, dataset_name=args.dataset_name,
        ).to(device)
        missing_feature_mask = get_missing_feature_mask(
            rate=args.missing_rate, n_nodes=n_nodes, n_features=n_features, seed = seed, type=args.mask_type,
        ).to(device) 
        x = data.x.clone()
        x[~missing_feature_mask] = float("nan") 

        
        filled_features = fsdcap(args, data, x, missing_feature_mask, num_classes,device)   

        feature_num = data.num_features
        model = get_model(
            model_name=args.model,
            num_features=feature_num,
            num_classes=num_classes,
            edge_index=data.edge_index,
            x=x,
            mask=missing_feature_mask,
            args=args,
        ).to(device)
        params = list(model.parameters())

        optimizer = torch.optim.Adam(params, lr=args.lr)
        criterion = torch.nn.NLLLoss()
        test_acc = 0
        val_accs = []
        for epoch in range(0, args.epochs):
            x = filled_features
            train(model, x, data, optimizer, criterion )
            ( _ , val_acc, tmp_test_acc), _ = test(
                model, x, data, evaluator=evaluator, device=device,
            )
            if epoch == 0 or val_acc > max(val_accs):
                test_acc = tmp_test_acc
            val_accs.append(val_acc)
            if epoch > args.patience and max(val_accs[-args.patience :]) <= max(val_accs[: -args.patience]):
                break

        test_accs.append(test_acc)
       
    test_acc_mean, test_acc_std = np.mean(test_accs), np.std(test_accs)
    str_test = f"\n=============test_auc: {test_acc_mean * 100:.2f}% +- {test_acc_std * 100:.2f}======================="

    print(f"{str_test}")


if __name__ == "__main__":
    # 忽略所有 UserWarning
    warnings.filterwarnings("ignore", category=UserWarning)
    args = parser.parse_args()
    run(args)
    

