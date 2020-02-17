# ResNet50 on ImageNet

This example trains a ResNet50 model on the ImageNet2012 dataset.

This is not a state of the art result on ImageNet2012 (see for example
https://arxiv.org/abs/1911.04252v2) but it is a useful example of a well
configured ResNet50 training script that reliably achieves 76.2-76.4% top-1 at
batch size 1024.

Batch size | Configuration               | Train Time | Top-1
---------- | --------------------------- | ---------- | -----
256        | 8 NVidia V100 (16GB) GPUs   | 1016m      | 76.7%
1024       | 32 Google TPUv3 cores       | 130m       | 76.5%
4096       | 128 Google TPUv3 cores      | 33m        | 76.2%
32768      | 1024 Google TPUv3 cores     | 5m         | 71.7%
