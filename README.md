# ANTHROPOS-V: Benchmarking the Novel Task of Crowd Volume Estimation

![teaser](STEERER-V/imgs/teaser.png)




Luca Collorone, Stefano D'Arrigo, Massimiliano Pappa, Guido Maria D'Amely di Melendugno, Giovanni Ficarra, Fabio Galasso



## Abstract

We introduce the novel task of Crowd Volume Estimation (CVE), defined as the process of estimating the collective body volume of crowds using only RGB images. Besides event management and public safety, CVE can be instrumental in approximating body weight, unlocking weight sensitive applications such as infrastructure stress assessment, and assuring even weight balance. We propose the first benchmark for CVE, comprising ANTHROPOS-V, a synthetic photorealistic video dataset featuring crowds in diverse urban environments. Its annotations include each person's volume, SMPL shape parameters, and keypoints. Also, we explore metrics pertinent to CVE, define baseline models adapted from Human Mesh Recovery and Crowd Counting domains, and propose a CVE specific methodology that surpasses baselines. Although synthetic, the weights and heights of individuals are aligned with the real-world population distribution across genders, and they transfer to the downstream task of CVE from real images.



## Table of Contents

- [Dataset](#dataset)
- [Benchmark](#benchmark)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)



## Dataset

The ANTHROPOS-V dataset is a synthetic, photorealistic video dataset containing crowds in diverse urban environments. 
The dataset is designed to reflect realistic weight and height distributions across genders, ensuring transferability to real-world scenarios.



## Benchmark

This work includes:
- Baseline models adapted from Human Mesh Recovery and Crowd Counting.
- A CVE-specific methodology that outperforms the baselines, namely STEERER-V.
- Evaluation metrics pertinent to the Crowd Volume Estimation (CVE) task.

For more details, please refer to the [paper](https://arxiv.org/abs/2501.01877) and the supplementary material provided.



## Installation

This repository uses a Conda environment to manage dependencies. Follow the steps below to set up your environment.

### Prerequisites

- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your machine.
- Git installed on your system.

### Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/colloroneluca/Crowd-Volume-Estimation.git
   cd Crowd-Volume-Estimation

2. **Install Environment**
    ```bash
    conda env create -f environment.yml
    conda activate STEERERV

3. **Download Dataset**
    ```bash
    cd Crowd-Volume-Estimation
    mkdir ProcessedData
    ```
    Download the dataset from [this link](https://drive.google.com/file/d/1IWvC4QQnwK2xL15Fxlz0To_JRb01nETT/view?usp=sharing) and place it under the folder `ProcessedData`.

    ```bash
    mkdir PretrainedModels
    ```
    Download the pretrained weights from [this link](https://drive.google.com/file/d/10PJEH9yI_0n0t1Tqwf7Y9Og3_f9saQRq/view?usp=sharing) and place it under `PretrainedModels`.


### Usage

1. **Train** 
    ```bash
    cd STEERER-V 
    python tools/train_cc.py --cfg configs/STEERER_V.py
2. **Test**
    ```bash
    cd STEERER-V 
    python tools/test_cc.py --cfg configs/STEERER_V_Test.py --checkpoint=../PretrainedModels/Anthropos-STEERER-V.pth
### Citation
If you find this work useful, please consider citing our paper:
```
@InProceedings{Collorone_2025_WACV,
    author    = {Collorone, Luca and Darrigo, Stefano and Pappa, Massimiliano and di Melendugno, Guido M. Damely and Ficarra, Giovanni and Galasso, Fabio},
    title     = {ANTHROPOS-V: Benchmarking the Novel Task of Crowd Volume Estimation},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {5284-5294}
}
```

### Acknowledgements

This code is primarily based on the work from the [STEERER](https://github.com/taohan10200/STEERER) original repository. We extend our gratitude to Matteo Fabbri for providing the materials essential for the initial setup of the dataset generation.