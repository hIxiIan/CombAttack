# datasets [cora ...] and [bc]
# method 1:
CUDA_VISIBLE_DEVICES=1 python test_cluster.py --dataset=cora --edge_flips=false --times=5 --direct_attack=true



# method 2:
# get the edge_flips_timestamp, such as 2022_01_22_00_18_43
CUDA_VISIBLE_DEVICES=1 python test_cluster.py --dataset=cora --edge_flips=true --times=5 --direct_attack=true

CUDA_VISIBLE_DEVICES=1 python test_edges.py --dataset=cora --times=5 --timestamp=2022_01_22_00_18_43