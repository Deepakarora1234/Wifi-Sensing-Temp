
# DeepFi: A Comparative Study of Deep-Learning Techniques in WiFi Sensing
## Introduction
DeepFi is an open-source benchmark and library for WiFi CSI human sensing, implemented by PyTorch. This study expands upon already established benchmark datasets and evaluation methodologies, offering valuable insights into essential performance metrics. Complex deep learning models like VGG-16, VGG-64,DenseNet and GoogLeNet are evaluated on publicly available dataset UT-HAR. The details are illustrated in our paper: [DeepFi: A Comparative Study of Deep-Learning Techniques in WiFi Sensing.](https://drive.google.com/drive/folders/1kcfN4SksUlWrKkKWNg8vmzOVOdeNbN2w?usp=drive_link)




## Requirements

1. Install `pytorch` and `torchvision` (we use `pytorch==1.12.0` and `torchvision==0.13.0`).
2. `pip install -r requirements.txt`

**Note that the project runs perfectly in Linux OS (`Ubuntu`). If you plan to use `Windows` to run the codes, you need to modify the all the `/` to `\\` in the code regarding the dataset directory for the CSI data loading.**

## Run
### Download Processed Data
Please download and organize the [processed datasets](https://drive.google.com/drive/folders/1kcfN4SksUlWrKkKWNg8vmzOVOdeNbN2w?usp=drive_link) in this structure:
```
Benchmark
├── Data
    ├── UT_HAR
    │   ├── data
    │   ├── label
```




To run models with supervised learning (train & test):  
Run: `python run.py --model [model name] --dataset UT_HAR_data`  

You can choose [model name] from the model list below
- VGG16
- VGG64
- DenseNet
- GoogLeNet





 


## Model Zoo
### VGG16
- Composed of 16 weight layers: 13 convolutional and 3 fully connected layers.
- Uses small 3x3 convolutional kernels and max-pooling layers for robust feature extraction.
- Renowned for its effectiveness in image classification tasks despite its computational intensity.
### VGG64
- An extension of VGG16 with 64 weight layers, offering greater feature extraction capacity.
- Employs the same 3x3 convolutional kernels and max-pooling layers as VGG16 but with increased depth.
- Requires significant computational resources, potentially enhancing performance on complex tasks.
### DenseNet
- Known for its dense connectivity, with each layer connected to every other layer within the same block.
- Enhances feature reuse and gradient flow, improving model performance and parameter efficiency.
- Efficient at capturing complex spatial and temporal information, suitable for activity recognition in Wi-Fi sensing.

### GoogLeNet
- Features 22 layers and inception modules, allowing for multi-scale feature extraction.
- Combines convolutions of varying sizes within a single layer for efficient local and global information acquisition.
- Computationally efficient with fewer parameters compared to traditional CNNs, excelling in image classification tasks.



## Dataset
#### UT-HAR
[*A Survey on Behavior Recognition Using WiFi Channel State Information*](https://ieeexplore.ieee.org/document/8067693) [[Github]](https://github.com/ermongroup/Wifi_Activity_Recognition)  
- **CSI size** : 1 x 250 x 90
- **number of classes** : 7
- **classes** : lie down, fall, walk, pickup, run, sit down, stand up
- **train number** : 3977
- **test number** : 996  

