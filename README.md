# adversarial-yolov3-cowc

This repository is an adaptation of https://gitlab.com/EAVISE/adversarial-yolo

This work is based on the paper: https://openaccess.thecvf.com/content/WACV2022/html/Du_Physical_Adversarial_Attacks_on_an_Aerial_Imagery_Object_Detector_WACV_2022_paper.html

```
@InProceedings{Du_2022_WACV,
    author    = {Du, Andrew and Chen, Bo and Chin, Tat-Jun and Law, Yee Wei and Sasdelli, Michele and Rajasegaran, Ramesh and Campbell, Dillon},
    title     = {Physical Adversarial Attacks on an Aerial Imagery Object Detector},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2022},
    pages     = {1796-1806}
}

```
## Getting started

### Installations
We used Python 3.7 to run our code and Anaconda to install the following libraries by:

```
* conda install opencv
* conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
* pip install tensorboardX tensorboard
* conda install matplotlib
* conda install -c conda-forge tqdm
```

### Pretrained weights
You can download the YOLOv3 COWC pretrained weights at: 

### Datasets
You can download the sidestreet and carpark datasets at: 

We also provide data from our physical-world test at:

NOTE: Make sure you add the pretrained weights and datasets into the main directory. 

### Optimising an adversarial patch
There are three main phython scripts used to optimise an adversarial patch:

* 


### Digitally testing the patch


### Physically testing the patch





