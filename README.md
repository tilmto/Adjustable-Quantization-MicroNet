# Adjustable-Quantization-MicroNet
### This is the submission for MicroNet Challenge hosted at NIPS 2019.

#### Our solution is called Adjustable Quantization, a finegrained mix-precision quantization scheme which is extremely fast to reach convergence started from a pretrained float32 model.

#### Main Idea
To enable a finegrained mix-precision quantization on a light-weight model, we aim at refining the traditional quantization to be more adjustable according to the efficiency limitation. Based on the previous [Quantization Aware Training](https://arxiv.org/abs/1712.05877), we introduce channel-wise trainable scale factors for both quantization range and precision. To eliminate the effect of the [outliners](https://arxiv.org/abs/1803.08607), we introduce scale factor <img src="https://latex.codecogs.com/svg.latex?\Large&space;x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" title="\Large \alpha" />
