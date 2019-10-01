# Adjustable-Quantization-MicroNet
### This is the submission for MicroNet Challenge hosted at NIPS 2019.

#### Our solution is called Adjustable Quantization, a finegrained mix-precision quantization scheme which is extremely fast to reach convergence started from a pretrained float32 model.

#### Main Idea
To enable a finegrained mix-precision quantization on a light-weight model, we aim at refining the traditional quantization to be more adjustable according to the efficiency limitation. Based on the previous [Quantization Aware Training (QAT)](https://arxiv.org/abs/1712.05877), we introduce channel-wise trainable scale factors for both quantization range and precision. To eliminate the effect of the [outliners](https://arxiv.org/abs/1803.08607), we introduce a trainable scale factor *alpha* for each channel to make the quantization range tp adjust itself during the quantization process. In addition, to further explore the compression potential, we give a channel-wise scale factor *beta* to the quantization precision so that each channel in different layers can learn to adjust its precision. Since number of parameters and FLOPS are all related with the channel precision settings, we explicitly introduce the #params and #FLOPS into the loss function, controlling the trade-off between accuracy and efficiency. The quantization parameter in the [QAT paper](https://arxiv.org/abs/1712.05877) can be rewritten as:  

<div align="center">
<img src="https://latex.codecogs.com/svg.latex?\Large&space;S=\frac{\alpha*T_{range}}{2^{[\beta*n]}-1}" title="Method" style="text-align: center"/>
</div>
