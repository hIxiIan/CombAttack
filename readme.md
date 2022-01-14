#trans2vec
## trans2vec.dan_walk
最容易理解的实现（效率也最低），部分使用了numba优化，但没有gf_walk的numba并行优化效率高

## trans2vec.gf_walk
模式1：

args.gf=True and args.gf_alias_mode=True

采用alias table实现随机游走

模式2：

args.gf=True and args.gf_alias_mode=False

采用基于邻居权重的偏置随机游走