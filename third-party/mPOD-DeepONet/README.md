# POD-DeepONet for multiple outputs
[![ICLR 2026](https://img.shields.io/badge/ICLR%202026-AI%20%26%20PDE%20Workshop-blue.svg)](https://openreview.net/forum?id=6cR1KU4inu)

This repository contains the official implementation of our **ICLR 2026 AI & PDE Workshop** paper: [**mPOD-DeepONet: POD-DeepONet for Multiple Outputs**](https://openreview.net/forum?id=6cR1KU4inu).

Our framework extends Proper Orthogonal Decomposition (POD) DeepONets to handle complex multiphysics scenarios where multiple output variables must be predicted simultaneously. We propose two primary configurations: 

* **cPOD-DeepONet**, which performs independent POD per field for *channel-wise* coefficient prediction. 
* **mPOD-DeepONet**, which uses multivariate functional PCA to extract a *joint*, shared basis across all fields, significantly reducing model size.

We also include **mKPCA-DeepONet**, a non-linear kernel PCA extension that maintains the compact footprint of mPOD-DeepONet.

## Datasets
Our experiments utilize the Multiphysics-Bench, covering diverse physical phenomena: Thermo-Fluid, Electro-Thermal, Electro-Fluid, and Magneto-Hydrodynamics.

### Data Acquisition & Preprocessing
- **Original Source**: Access the full raw data at the [xie-lab-ml/multiphysics-bench](https://github.com/xie-lab-ml/multiphysics-bench.git) repository.
- **Processing**: We provide utility scripts in the `merge_data/` directory to transform the raw benchmark data into `.pt` files ready for training and testing.

## Models and Benchmarks
We evaluate our proposed POD-based architectures (cPOD-DeepONet, mPOD-DeepONet, and the non-linear mKPCA-DeepONet) against state-of-the-art operator learning frameworks. Detailed usage examples and training configurations are available in the `example/` directory.

### Baseline Comparisons & Implemented Models
- **FNO (Fourier Neural Operator)**: Implementation via the [NeuralOperator library](https://neuraloperator.github.io/dev/index.html).
- **cDeepONet**: Located in `models/DeepONet.py` as *MultiTrunkDeepONet*.
- **mDeepONet**: Located in `models/DeepONet.py` as *ShareTrunkDeepONet*.
- **cPOD-DeepONet (Ours)**: Located in `models/MFPCA.py` as *ChannelPodONet*.
- **mPOD-DeepONet (Ours)**: Located in `models/MFPCA.py` as *MPodONet*.
- **mKPCA-DeepONet (Ours)**: Located in `models/MFPCA.py` as *MKPodONet*.

## Citation
If you find our work or code useful in your research, please cite our paper:

```bibtex
@inproceedings{
chou2026mpoddeeponet,
title={m{POD}-Deep{ON}et: {POD}-Deep{ON}et for Multiple Outputs},
author={Chieh-An Chou and Lu-Hung Chen},
booktitle={AI{\&}PDE: ICLR 2026 Workshop on AI and Partial Differential Equations},
year={2026},
url={[https://openreview.net/forum?id=6cR1KU4inu](https://openreview.net/forum?id=6cR1KU4inu)}
}