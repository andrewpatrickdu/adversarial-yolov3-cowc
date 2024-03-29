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
We used Python 3.7 to write/run our code and Anaconda to install the following libraries:

```
conda install opencv
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
pip install tensorboardX tensorboard
conda install matplotlib
conda install -c conda-forge tqdm
```

### Pretrained weights and datasets
You can download the YOLOv3 COWC pretrained weights (```weights``` folder), sidestreet and carpark datasets (```#sidestreet``` and ```#carpark``` folders), and data from our physical-world test (```physical_test``` folder) at: https://universityofadelaide.box.com/v/adversarial-cowc

NOTE: Make sure you add the four folders into the main directory (```adversarial-yolov3-cowc```). 

## Optimising an adversarial patch
There are three main python scripts used to optimise an adversarial patch:

* ```patch_config.py``` - contains the training parameters of the patch such as what scene to attack, to apply weather transformations or not, patch size, number of patches, number of training epochs, learning rate, etc.
* ```load_data.py``` - defines important functions used in the patch optimisation process such as extracting the maximum objectness score from an image, calculating NPS and TV, and patch transformations (geometric and colour space).
* ```train_patch.py``` - used to optimise an adversarial patch. 

You can generate a patch by running the command:

```
python train_patch.py {experiment} {folder_name}
```
E.g., 

```
python train_patch.py sidestreet experiment01
```

## Digitally testing the patch
You can digitally test your patch with ```digital_test.py```. Open the script and adjust the 'attack settings' accordingly. More details are provided in the script.

## Physically testing the patch
We provide some data that comprises of images of our patches in the scene of attack (sidestreet and carpark). You can evaluate these images with ```physical_test.py```. Open the script and adjust the 'attack settings' accordingly. More details are provided in the script.

## Help and updates
When time permits, I will update the repository based on the issues others may have at setting up and/or running the code.



