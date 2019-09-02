## I. INTRUCTION

### 1.1 Advantages and Disadvantages of MRI
1. MRI是在K-space域获取的而不是图像域，高质量的图像获取需要在k空间域一条一条的采集（64-512），由于扫描速度和病人以及活跃器官的运动会使得图像不清晰。
2. 对于需要造影剂的采集场景，长时间的采样会导致造影剂的稀释（例如。。。）
3. 长时间的采样也会是病人不舒服

### 1.2 Acceleration
1. fully sampled：Nyquist-Shannon sampling criteria[[1]](#q1)
2. PFI[[2]](#q2): undersampled technique, speed limited < 2
3. parallel imaging: using multiple receiver, 重建方法有SENSE、GRAPPA。速度受到线圈数的影响（fastmri中的速度提及[[3]](#q3)），会提升制造成本
4. Compressed Sensing[[5]](#q5)：条件苛刻，迭代重建算法时间长，CS超参数调优问题[[4]](#q4)

## II. RELATED WORK AND OUR CONTRIBUTIONS

### A. Compressed Sensing
1. undersampling scheme: 研究已经寻求最佳欠采样方案，该方案应尽可能随机地产生非相干欠采样伪像，以便可以应用适当的非线性重建来抑制类似噪声的伪影而不会降低重建的图像质量。欠采样的设置还应该考虑到物理的易实现性。目前大多数用的是在2D空间中应用1D的高斯分布
2. 一般情况下，MRI获得的医学图像是自然可压缩的（图像像素或原始数据点主要是零值或即可压缩的）
3. 非线性的重建方法（迭代算法，时间较长）[[4]](#q4)
4. 尽管CS有希望应用于临床，但是目前大多数诊所采用的是Cartesian sequences全采样或者Parallel Imaging[[6]](#q6)
5. CS主要受一下因素限制：（1）满足CS-MRI要求的不一致标准[[7]](#q7)（2）广泛应用的稀疏变换可能过于简单而不能捕获与生物组织的细微差异相关的复杂图像细节，例如，基于TV的稀疏变换惩罚重建图像中的局部变化，但可能引入阶梯伪影，并且小波变换实施点奇异性和各向同性特征，但正交小波可能导致块状伪影。（3）非线性优化求解器通常涉及迭代计算，这可能导致相对较长的重建时间[[7]](#q7)。（4）在当前的CS-MRI方法中预测的不适当的超参数可能会导致过度规则化，这将产生过于平滑和不自然的重建或具有残余欠采样伪影的图像[[7]](#q7)。由于以上限制，CS的加速率在[2x, 6x]之间[[6]](#q6)。

### B. Deep Learning-BasedCS-MRI
1. 深度学习在计算机视觉中大展身手，在许多视觉任务上都表现优异。[[8]](#q8)综述了深度学习在深度学习方面的应用，并且[[9]](#q9)将深度学习应用到图像重建上。特别的CS-MRI解决了一般的反转问题如图像超分、降噪等。尤其是CNN表现优异。
2. 最近出现了一些将深度学习应用于CS-MRI的研究，可以分为两类，模拟单线圈CS-MRI[[6]](#q6)与多线圈CS-MRI[[4]](#q4)。
3. [[10]](#q10)将深度学习引入CS-MRI，尽管初步的定性可视化显示出了一些希望，但这种方法对于CS-MRI重建的适用性还有待于详细的定量评估。
4. ADMM-Net[[11]](#q11)， 该方法获得了与经典CS-MRI重建方法相似的重建结果，但显着减少了重建时间。
5. Hammernik[[4]](#q4)训练了一个变分网络来求解CS-MRI,和Lee[[12]](#q12)将基于CNN的CS-MRI与并行成像相结合来估计和去除混叠伪影。
6. 尽管深度学习在解决CS-MRI方面显示出了巨大的潜力，重建速度更快，但到目前为止没有发现改善与经典CS-MRI所能达到的有明显不同[[6]](#q6)。此外，与其他深度学习应用程序类似，定义网络体系结构并不是轻而易举的事情，除非执行全面的参数调整，否则深度网络训练的收敛可能难以实现。

### B. Contributions
1. 提出一种新的网络结构（新在何处）
2. 精细的训练方法（如GAN以及维持GAN有效训练的方法）
3. new loss
4. Frequency domain限制来保持数据的一致性（应用在loss中）
5. 对比试验：将重建的图像应用于分割等任务中

## II. Method
### A. General CS-MRI
#### 1) forward model
图像恢复或重建的观测或数据采集正演模型可以近似为离散的线性系统[[13]](#q13),

$$y = Fx + \epsilon$$

其中$x \in \Complex^N$表示需要重建的图像，它包括$\sqrt{N} \times \sqrt{N}$个像素点。观测数据标注为$y \in \Complex^M$。该前向模型可以用一个线性运算符$F: \Complex^N \mapsto \Complex^M$来表示。$F \in \Complex^{M \times N}$表示用于去模糊的各种图像恢复或重构卷积算子，用于SR的过滤子采样算子，以及用于CS-MRI重建的k空间随机欠采样算子[[13]](#q13)

#### 2) Inverse Model
方程1的逆估计通常是不适定的，因为问题通常是欠确定的，$M \ll N$。此外，由于数值病态算子F和噪声的存在(ε)，逆模型是不稳定的[[11]](#q11)。

#### 3) Classic Model-Based CS-MRI
为了解决这个欠确定和不适定的cs-mri系统，人们必须利用x的先验知识，这可以被表达为一个无约束的优化问题，即：
$$ min_x \frac{}{}12\parallel F_ux - y \parallel_{2}^{2} + \lambda R(x)$$ 
其中$\parallel F_ux - y \parallel_{2}^{2}$是数据保真项，$F_u \in \Complex^{M \times N}$是欠采样的傅里叶变换编码矩阵。$R$是$x$的正则化项，$\lambda$是正则化项系数.正则化项$R$通常为$l_q-norms (0\leq q \leq 1)$在x的稀疏域中。

#### 4) 基于深度学习的CS-MRI
基于深度学习的研究[[10]](#q10)[[14]](#q14)将CNN与CS-MRI结合，公式如下：
$$min_x \frac{1}{2} \parallel F_ux-y \parallel_{2}^{2}+\lambda R(x) + \zeta \parallel x - f_{cnn}(x_u|\hat{\theta}) \parallel_2^2 $$
其中$f_{cnn}$是数据前向传播通过参数$\theta$来参数化，$\zeta$是正则化系数，经过CNN产生的图像作为推理图像加入到正则化项中，$\hat{\theta}$是经过训练的CNN参数。另外$x_u=F_u^H$是从用零值填充的欠采样k-space获取的，其中$H$表示Hermitian的转置运算。

MRI数据以复数格式对幅度和相位信息进行自然编码。对于基于深度学习的处理复数的方法，至少有两种策略：（1）可以映射函数$Re*: \reals^N \mapsto \Complex^N$将实值函数映射到复数空间，例如$Re*(x)=x+0i$。因此MRI前向运算符可以被表示为$F: \reals^N \mapsto \Complex^N \mapsto \it{f}\Complex^N \mapsto u\Complex^M$，那么$F_u$包含了傅里叶变换$\it{f}$与下采样运算符$u$[[15]](#q15)。（2）复数可以分为两个独立的通道进行学习($\Complex^N \mapsto \reals^{2N}$)[[14]](#q14)。

### B. Method 2
### C. Propose Method
### D. Innovation point ...
### E. Evaluation Methods
1. PSNR
2. SSIM

## III. Experiments Settings and Results
### A. Experiments Settings
#### 1) Datasets
1. fastMri[[3]](#q3):
2. VNData[[11]](#q4):
#### 2) Mask
random Cartesian sequences, x4, x8
#### 3) Network Variations
#### 4) Comparison Methods
[[3]](#q3), [[11]](#q11), Total Variation regularization



<span id='q1'></span>
[1] H. Nyquist, “Certain topics in telegraph transmission theory,” Trans. Amer. Inst. Electr. Engineers, vol. 47, no. 2, pp. 617–644, Apr. 1928.

<span id='q2'></span>
[2] G. McGibney, M. R. Smith, S. T. Nichols, and A. Crawley, “Quantitative evaluation of several partial Fourier reconstruction algorithms used in MRI,” Magn. Reson. Med., vol. 30, no. 1, pp. 51–59, 1993.

<span id='q3'></span>
[3] fastMRI: An Open Dataset and Benchmarks for Accelerated MRI

<span id='q4'></span>
[4] Learning a variational network for reconstruction of accelerated MRI data

<span id='q5'></span>
[5] D. L. Donoho, “Compressed sensing,” IEEE Trans. Inf. Theory, vol. 52, no. 4, pp. 1289–1306, Apr. 2006.

<span id='q6'></span>
[6] DAGAN: Deep De-Aliasing GenerativeAdversarial Networks for Fast Compressed Sensing MRI Reconstruction

<span id='q7'></span>
[7] K. G. Hollingsworth, “Reducing acquisition time in clinical MRI by data undersampling and compressed sensing reconstruction,” Phys. Med. Biol., vol. 60, no. 21, pp. R297–R322, 2015.

<span id='q8'></span>
[8] D. Shen, G. Wu, and H.-I. Suk, “Deep learning in medical image analysis,” Annu. Rev. Biomed. Eng., vol. 19, pp. 221–248, Jun. 2017.

<span id='q9'></span>
[9] G. Wang, “A perspective on deep imaging,” IEEE Access,vol.4, pp. 8914–8924, 2016.

<span id='q10'></span>
[10] S. Wang et al., “Accelerating magnetic resonance imaging via deep learning,” in Proc. ISBI, 2016, pp. 514–517.

<span id='q11'></span>
[11] Y. Liu et al., “Balanced sparse model for tight frames in compressed sensing magnetic resonance imaging,” PLoS ONE, vol. 10, no. 4, p. e0119584, 2015.

<span id='q12'></span>
[12] D. Lee, J. Yoo, and J. C. Ye. (2017). “Deep artifact learning for compressed sensing and parallel MRI.” [Online]. Available: https://arxiv.org/abs/1703.01120

<span id='q13'></span>
[13] E. M. Eksioglu, “Decoupled algorithm for MRI reconstruction using nonlocal block matching model: BM3D-MRI,” J. Math. Imag. Vis., vol. 56, no. 3, pp. 430–440, 2016.

<span id='q14'></span>
[14] A deep cascade of convolutional neural networks for dynamic MR image reconstruction

<span id='q15'></span>
[15] Multicontrast MRI reconstruction with structure-guided total variation