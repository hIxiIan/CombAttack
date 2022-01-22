#  get edge_flips_timestamp, such as 2022_01_20_23_57_24
CUDA_VISIBLE_DEVICES=1 python test_cluster.py --dataset=bc1 --times=5 -ef=true --bmbc_mode=true

# get attack results according to edge_flips_timestamp
CUDA_VISIBLE_DEVICES=1 python test_bm_edges.py --dataset=bc1 --times=5 -eft=2022_01_20_23_57_24

# mappings edge_flips_timestamp (.npy), result_timestamp (.csv)
""