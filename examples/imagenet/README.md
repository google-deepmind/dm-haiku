# ResNet50 on ImageNet

This example trains a ResNet50 model on the ImageNet2012 dataset.

This is not a state of the art result on ImageNet2012 (see for example
https://arxiv.org/abs/1911.04252v2) but it is a useful example of a well
configured ResNet50 training script that out of the box achieves ~77% top-1
accuracy at batch size 4096 within ~90 minutes on 32 TPUv3 cores.
