# ResNet50 on ImageNet

This example trains a ResNet50 model on the ImageNet2012 dataset.

This is not a state of the art result on ImageNet2012 (see for example
https://arxiv.org/abs/2102.06171) but it is a useful example of a well
configured ResNet50 training script that out of the box achieves ~77% top-1
accuracy at batch size 4096 within ~90 minutes on 32 TPUv3 cores.

## Mixed precision training

By default this example runs in full (f32) precision on all platforms. Training
the model this way using 32 NVIDIA V100 GPUs leads to a top-1 accuracy of ~77%
within about 200 minutes.

On many GPUs you can often get a significant performance boost from running
certain parts of your model using half precision (F16). This is further
accelerated by the presence of special hardware that has hardware support for
common deep learning operations (such as 4x4 fused multiply-add in f16).

Haiku integrates with [JMP] (a mixed precision library for JAX) to easily enable
mixed precision training. Using mixed precision you can achieve a 2x speedup in
training time with only a small drop in top-1 accuracy. We trained this model
using the flags below on 32 NVIDIA V100 GPUs to 76.7% top-1 in 100 minutes:

    --train_device_batch_size=256
    --mp_policy=params=f32,compute=f16,output=f32
    --mp_scale_type=Static

TPUs have hardware support for an alternative 16bit floating point format called
[bfloat16]. Because [bfloat16] has a higher dynamic range you can avoid loss
scaling and for this model you can keep batch norm in bfloat16 as well. Using
the hyper parameters below we were able to train the model to 76.8% top-1 on
32 Google TPUv3 cores to 76.8% top-1 in 59 minutes:

    --train_device_batch_size=256
    --mp_policy=params=f32,compute=bf16
    --mp_bn_policy=params=f32,compute=bf16

[JMP]: https://github.com/deepmind/jmp
[bfloat16]: https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
