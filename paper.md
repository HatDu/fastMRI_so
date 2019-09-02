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
5. CS主要受一下因素限制：（1）满足CS-MRI要求的不一致标准[[7]](#q7)（2）广泛应用的稀疏变换可能过于简单而不能捕获与生物组织的细微差异相关的复杂图像细节，例如，基于TV的稀疏变换惩罚重建图像中的局部变化，但可能引入阶梯伪影，并且小波变换实施点奇异性和各向同性特征，但正交小波可能导致块状伪影。（3）非线性优化求解器通常涉及迭代计算，这可能导致相对较长的重建时间[[7]](#q7)。（4）在当前的CS-MRI方法中预测的不适当的超参数可能会导致过度规则化，这将产生过于平滑和不自然的重建或具有残余欠采样伪影的图像[[7]](#q7)。


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
