# swa_demo
This is a stochastic weight averaging demo

About SWA construction : https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging

简单点讲就是把每个epoch训练的ckpt做平均，对batchnorm这种特殊的norm结构，需要对moving做整个数据的均值方差的平均化处理
