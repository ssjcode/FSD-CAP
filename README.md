# FSD-CAP
This is the Code of "FSD-CAP: Fractional Subgraph Diffusion with Class-Aware Propagation for Graph Feature Imputation"

# For Code

## Environmental requirements
Any relevant dependencies and versions in the code can be queried in :FSD-CAP.yml
To reduce unnecessary troubles, please note: 
``python=3.10.15`` & ``pytorch >=2.20`` & ``torch-geometric==2.6.1`` & ``ogb==1.3.6``

## Datasets
Two ways of obtaining data are provided

(1)Through ``torch_geometric.datasets.Planetoid`` and ``torch_geometric.datasets.Amazon`` download Cora & Citeseer & PubMed and Photo & Computers respectively .
When you run FSD-CAP, it also relies on these two functions to download the dataset

(2) we can be downloaded at https://github.com/kimiyoung/planetoid and https://github.com/shchur/gnn-benchmark respectively Cora & Citeseer & PubMed and  Photo & Computers


## RUN FSD-CAP
  For the node classification task，you can be run through the following command:
  ```
  python run_node.py --dataset_name Cora --mask_type structural --missing_rate 0.995
  ```
  Among them, dataset_name represents the database you want to use. Here are five options - "Cora", "CiteSeer", "PubMed", "Photo", "Computers", and mask_type represents the type of missing data.     "missing_rate" represents the data missing rate.
  
  For the edge prediction task，you can be run through the following command:
  ```
  python run_link.py --dataset_name Photo --mask_type uniform --missing_rate 0.995
  ```

  ## Main Baseline Codes
  --
  --FP  : " On the Unreasonable Effectiveness of Feature Propagation in Learning on Graphs with Missing Node Features" (https://github.com/twitter-research/feature-propagation )
  - PCFI:  " CONFIDENCE-BASED FEATURE IMPUTATION FOR GRAPHS WITH PARTIALLY KNOWN FEATURES" (https://github.com/daehoum1/pcfi)
  - SAT: "Learning on Attribute-Missing Graphs" (https://github.com/xuChenSJTU/SAT-master-online)
  - SVGA: "Accurate Node Feature Estimation with Structured Variational Graph Autoencoder" (https://github.com/snudatalab/SVGA)
  - GNN-AC: "Heterogeneous Graph Neural Network via Attribute Completion" (https://github.com/liangchundong/HGNN-AC)
  - ITR: "Initializing Then Refning: A Simple Graph Attribute Imputation Network" (https://github.com/WxTu/ITR)
  - SGC: "Simplifying Graph Convolutional Networks" (https://github.com/Tiiiger/SGC)
  - AGE: "Adaptive Graph Encoder for Attributed Graph Embedding" (https://github.com/thunlp/AGE)




