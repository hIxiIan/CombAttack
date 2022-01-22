# get features and timestamp, such as feature_timestamp=2022_01_22_00_18_43
python test_trans2vec.py --times=5

# get edge_flips, such as edge_flips_timestamp=
CUDA_VISIBLE_DEVICES=1 python test_cluster.py --dataset=trans2vec --times=5 --feature_timestamp=2022_01_22_00_18_43

# get attack results according to feature_timestamp and edge_flips_timestamp
CUDA_VISIBLE_DEVICES=1 python test_bc_edges.py --dataset=tedge --times=5 -tt=2022_01_22_00_18_43 -eft=


# mappings features_timestamp (.csv), edge_flips_timestamp (.npy), result_timestamp (.csv)
"2022_01_22_00_18_43, , "
