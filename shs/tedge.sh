# get features and timestamp, such as feature_timestamp=2022_01_22_00_56_04
python test_tedge.py --times=5

# get edge_flips, such as edge_flips_timestamp=2022_01_22_01_21_23
CUDA_VISIBLE_DEVICES=1 python test_cluster.py --dataset=tedge --times=5 --feature_timestamp=2022_01_22_00_56_04

# get attack results according to feature_timestamp and edge_flips_timestamp
CUDA_VISIBLE_DEVICES=1 python test_bc_edges.py --dataset=tedge --times=5 -tt=2022_01_22_00_56_04 -eft=2022_01_22_01_21_23


# mappings features_timestamp (.csv), edge_flips_timestamp (.npy), result_timestamp (.csv)
"2022_01_22_00_56_04, 2022_01_22_01_21_23, 2022_01_22_01_21_23"
