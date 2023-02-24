# AF-Net
**Attention Augmented Feature Fusion based Nuclei Instance Segmentation and Type Classification in Histology Images**
# Introduction

Nuclei instance segmentation and type classification are tasks in medical image analysis that aim to identify and separate individual nuclei in an image and classify them based on their type.
In nuclei instance segmentation, the goal is to identify the location and boundaries of each individual nucleus in an image. This is usually done using techniques such as image segmentation, object detection, or semantic segmentation. The output is typically a mask or a binary image where each nucleus is assigned a unique label or color.
![Alt text](Figures/ff1_7.jpg?raw=true "Title")
![Alt text](Figures/ff1_9.png?raw=true "Title")

# Abstract
The precise identification and examination of nuclei play a critical role in the detection and examination of cancer. Despite their significance, the task of accurately dividing and categorizing nuclei instances is a challenging one, often complicated by the presence of overlapping and congested nuclei with indistinct edges. Existing methods concentrate on area proposal techniques and feature encoding frameworks, yet often fall short in accurately pinpointing instances. In this research, we introduce a straightforward yet effective model that can recognize instance borders with precision and tackle the issue of class imbalance. By incorporating nuclei pixel positional information and a unique loss function, our model delivers accurate class information for each nucleus. Our network comprises a compact, attention-aware feature fusion architecture with dedicated instance probability, shape radial estimator, and classification heads. A combined classification loss function is employed to minimize loss by assigning weighted loss to each class based on their frequency of occurrence, resolving the class imbalance issues frequently encountered in publicly accessible nuclei datasets. 
## Highlights
- **Dual Segmentation & Classification** A **new method** for combining the characteristics of detected objects on an individual basis, utilizing an **attention mechanism** for **feature fusion** to tackle the diverse shape and texture variations of **nuclei**.
- **Radial Distance based Shape Estimation** Estimating the shape of nuclei accurately by using **radial distances** that measure the distance of each pixel in the nucleus from the **contour** and giving greater importance to **pixels** closer to the boundary.
-**State-of-the-art performance.** The proposed model has shown **exceptional performance** compared to other leading methods when tested on **eight** different publicly available datasets, as demonstrated by the experimental results.
![Alt text](Figures/ff1_5.png?raw=true "Title")
# Base Model
  Models used in the original paper
  ![Alt text](Figures/nurisc.png?raw=true "Title")
## Requirements
-   python  3.6.10
-   scikit-learn 0.23.1
-   scikit-image 0.16.2
-   opencv-python 4.1.2.32
-   Tensorflow 1.12
-   
## Installation

This package is compatible with Python 3.6 - 3.10.

For using AF-Net please follow the following steps.

1. Please first [install TensorFlow](https://www.tensorflow.org/install)
(either TensorFlow 1 or 2) by following the official instructions.
For [GPU support](https://www.tensorflow.org/install/gpu), it is very
important to install the specific versions of CUDA and cuDNN that are
compatible with the respective version of TensorFlow. (If you need help and can use `conda`, take a look at [this](https://github.com/CSBDeep/CSBDeep/tree/master/extras#conda-environment).)



#### Notes

- Depending on your Python installation, you may need to use `pip3` instead of `pip`.
- You can find out which version of TensorFlow is installed via `pip show tensorflow`.
- You need to install [gputools](https://github.com/maweigert/gputools) if you want to use OpenCL-based computations on the GPU to speed up training.



## Usage

We provide example workflows for each dataset in Jupyter [notebooks](https://github.com/eshasadia/AF-Net/blob/master/Notebooks/) that illustrate how this model can be used.

### Pretrained Models

Currently we provide pretrained models for the following 8 datasets that can be utilized for further analysis or training:


| Dataset | Modality (Staining) | Image format | Example Image    | Description  |
| :-- | :-: | :-:| :-:| :-- |
| [`CPM-15`](https://drive.google.com/drive/folders/11ko-GcDsPpA9GBHuCtl_jNzWQl6qY_-I)| H&E | RGB| <img src="https://github.com/eshasadia/AF-Net/blob/master/Figures/cpm15_git.PNG" title="example CPM-15 image " width="120px" align="center">       |*CPM-15  dataset using AF-Net model*  trained for [nuclei instance segmentation](https://drive.google.com/drive/folders/1M_RuhS03SytNasZHsjd2xIKy8yRQsZiY?usp=share_link) |
| [`CPM-17`](https://drive.google.com/drive/folders/1sJ4nmkif6j4s2FOGj8j6i_Ye7z9w0TfA)| H&E | RGB| <img src="https://github.com/eshasadia/AF-Net/blob/master/Figures/cpm17_git.PNG" title="example CPM-17 image " width="120px" align="center">       |*CPM-17  dataset using AF-Net model*  trained for [nuclei instance segmentation](https://drive.google.com/drive/folders/1VFC4IYs5OCkqlWM1UKAG8HumSKjrZyOz?usp=share_link) |
| [`CoNSeP`](https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/)| H&E | RGB| <img src="https://github.com/eshasadia/AF-Net/blob/master/Figures/con2.PNG" title="example CoNSeP image " width="120px" align="center">       |*CoNSeP dataset using AF-Net model*  trained for [nuclei instance segmentation and classification](https://drive.google.com/drive/folders/1lQ6ZWK3opvbRC2-ft6C8X7omELyZIMSY?usp=share_link) |
| [`Kumar`](https://drive.google.com/drive/folders/1bI3RyshWej9c4YoRW-_q7lh7FOFDFUrJ)| H&E | RGB| <img src="https://github.com/eshasadia/AF-Net/blob/master/Figures/kumar_git.PNG" title="example Kumar image " width="120px" align="center">       |*Kumar  dataset using AF-Net model*  trained for [nuclei instance segmentation](https://drive.google.com/drive/folders/1x5_Bt6i9FlZBxo7EnDNzYCm0Z5eIBD51?usp=share_link) |
| [`TNBC`](https://drive.google.com/drive/folders/1taB8boGyycjV4X1a2vCIAV9fwMxFSS41)| H&E | RGB| <img src="https://github.com/eshasadia/AF-Net/blob/master/Figures/tnbc_git.PNG" title="example TNBC image " width="120px" align="center">       |*TNBC  dataset using AF-Net model*  trained for [nuclei instance segmentation](https://drive.google.com/drive/folders/1OW0FlsI5cK47ZDNnHAIL-rfWsjfg3X-R?usp=share_link) |
| [`CryoNuSeg`](https://www.kaggle.com/datasets/ipateam/segmentation-of-nuclei-in-cryosectioned-he-images)| H&E | RGB| <img src="https://github.com/eshasadia/AF-Net/blob/master/Figures/cryonuseg_git.PNG" title="example CryoNuSeg image " width="120px" align="center">       |*CryoNuSeg  dataset using AF-Net model*  trained for [nuclei instance segmentation](https://drive.google.com/drive/folders/1TrXM0sSK2ynjT6HigyJ29twdU-KBc4SZ?usp=share_link) |
| [`Lizard`](https://warwick.ac.uk/fac/cross_fac/tia/data/lizard)| H&E | RGB| <img src="https://github.com/eshasadia/AF-Net/blob/master/Figures/lizard_img.PNG" title="example Lizard image " width="120px" align="center">       |*Lizard  dataset using AF-Net model*  trained for [nuclei instance segmentation](https://drive.google.com/drive/folders/1U7FAGMPynbnSEPbd6Xjq2PIr0JY1YmSM?usp=share_link) |
| [`PanNuke`](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke)| H&E | RGB| <img src="https://github.com/eshasadia/AF-Net/blob/master/Figures/pannuke_git.PNG" title="example PanNuke image " width="120px" align="center">       |*PanNuke  dataset using AF-Net model*  trained for [nuclei instance segmentation and classification](https://drive.google.com/drive/folders/1ssrGPj65SKdpqNXiL4dwfR7zPv2FEYPk?usp=share_link) |


## Steps To run Notebooks
  * Define path for training and test images 
  * For training from scratch use the following configuration:
    `model = nurisc2D(config=conf, name='define folder for saving model weights', basedir='define path for model weights')`
  * For loading pre-trained model use this configuration:
   `model = nurisc2D(config=None, name='pretrained model folder', basedir='pretained model path')`

## Performance



### Segmentation

AFNet model has been trained for instance segmentation for the following 8 datasets including CPM-17, CPM-15, Kumar,  PanNuke, TNBC, CoNSeP and Lizard datasets.
![Alt text](Figures/segmentation.PNG?raw=true "Title")

![Alt text](Figures/resultstable.PNG?raw=true "Title")
### Classification
AFNet model has been trained for classification for PanNuke and CoNSeP datasets, i.e. each found nuclei instance is additionally  classified into  its specific tumor class(e.g. cell types):

![Alt text](Figures/con_class.PNG?raw=true "Title")

## Instance segmentation and classification Evaluation Metrics

We have evaluated the models performance on the following evaluation metrics.

* `tp`, `fp`, `fn`
* `precision`, `recall`, `accuracy`, `f1`
* `panoptic_quality`
* `mean_true_score`, `mean_matched_score`

which are computed by matching ground-truth/prediction objects if their IoU exceeds a threshold (by default 50%). 



# How to Cite
If you find our work useful in your research, please consider citing:
```bibtex
 @inproceedings{nurisc,
  title={NuRiSC: Nuclei Radial Instance Segmentation and Type Classification},
  author={Nasir, Esha and Fraz, Muhammad},
  booktitle={Medical Imaging and Computer-Aided Diagnosis, MICAD2022},
  year={2022},
  organization={Springer}
}


@article{article,
author = {Nasir, Esha and Parvaiz, Arshi and Fraz, Muhammad},
year = {2022},
month = {12},
pages = {1-56},
title = {Nuclei and glands instance segmentation in histology images: a narrative review},
journal = {Artificial Intelligence Review},
doi = {10.1007/s10462-022-10372-5}
}
```

