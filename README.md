# oukinkaikousyo

Here are the codes on some basic cv models and the application of these models on the human face recognization dataset.

The dataset is available at http://pan.baidu.com/s/1ge9iKXP.

## The Dataset

Below are some samples of the dataset,
<div align = 'center'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/oukinkaikousyo/master/samples/Face1.png" width='200px'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/oukinkaikousyo/master/samples/Face2.png" width='200px'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/oukinkaikousyo/master/samples/Face3.png" width='200px'>
</div>

The dataset consists of 62928 pictures with 3 channels, which are of the size 31*31. It is divided into a training set with a size of 41952 and a test set with a size of 20976. The dataset is derived from 1311 persons and the task is to classify the given photo into the right person.

## Models
I have implemented the following models:

 * A DIY model (I don't give the code of this model since it is written by my classmate and I'm not sure whether he would be happy if I did so)
 * Network in Network
 * Vgg16
 * Vgg19
 * ResNet
 * DenseNet
 * ResNext
 * Xception
 * InceptionV3
 * SENet


## Experiment results

| model | training accuracy | training loss | test accuracy | test loss | trained model
| :----: | :-----: | :----: | :----: | :----: | :----: |
| DIY Model1 | 99.97% | 0.9100 | 78.81% | No Record | https://pan.baidu.com/s/1o7NhJ1w |
| DIY Model2 | 100.00% | 0.6700 | 74.70% | No Record | https://pan.baidu.com/s/1eRUjFBc |
| LeCun Net | 58.48% | 1.6422 | 53.69% | 2.1783 | https://pan.baidu.com/s/1qXKvQPm |
| NiN | 99.80% | 0.1481 | 93.54% | 0.4095 | https://pan.baidu.com/s/1c1DYCZA |
| Vgg16 | 96.56% | 0.6324 | 88.89% | 1.1115 | https://pan.baidu.com/s/1i5KQyLf |
| Vgg19 | 99.75% | 0.6149 | 91.75% | 1.1425 | https://pan.baidu.com/s/1bVDB34 |
| ResNet | 99.99% | 0.1433 | 96.67% | 0.3012 | https://pan.baidu.com/s/1dEBoKZR |
| DenseNet | 99.99% | 0.0913 | 98.03% | 0.1904 | https://pan.baidu.com/s/1cAVmJc |
| ResNext | 99.99% | 0.0734 | 98.42% | 0.1585 | https://pan.baidu.com/s/1misZTOW |
| Xception | 99.98% | 0.0904 | 93.32% | 0.4085 | https://pan.baidu.com/s/1hrZ5meC |
| InceptionV3 | uncompatible | uncompatible | uncompatible | uncompatible |  |
| SENet | 99.98% | 0.0989 | 98.38% | 0.1782 | https://pan.baidu.com/s/1mikmAZA |


## Training Process

Below is the data derived from my training process.

### LeNet
<div align = 'center'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/oukinkaikousyo/master/images/LeNet.png" width='600px'>
</div>

### NiN
<div align = 'center'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/oukinkaikousyo/master/images/NiN.png" width='600px'>
</div>

### Vgg16
<div align = 'center'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/oukinkaikousyo/master/images/Vgg16.png" width='600px'>
</div>

### Vgg19
<div align = 'center'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/oukinkaikousyo/master/images/Vgg19.png" width='600px'>
</div>

### ResNet
<div align = 'center'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/oukinkaikousyo/master/images/ResNet.png" width='600px'>
</div>

### DenseNet
<div align = 'center'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/oukinkaikousyo/master/images/DenseNet.png" width='600px'>
</div>

### ResNext
<div align = 'center'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/oukinkaikousyo/master/images/ResNext.png" width='600px'>
</div>

### Xception
<div align = 'center'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/oukinkaikousyo/master/images/Xception.png" width='600px'>
</div>

### SENet
<div align = 'center'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/oukinkaikousyo/master/images/SENet.png" width='600px'>
</div>

# How to Reproduce
You should first download the dataset from the link given above. The dataset is stored in the binary form, but it really takes some time to process the data. So I implement a numpy version for data loading. When you first run the code, make sure that you set "first_run" to be True. For example, you can run the following code to train a ResNet Model,

```
python main.py --model ResNet --first_run True
```

After the first run, some .npy formed files will be created. When you run the model for the second time or more, simply use the following code,

```
python main.py --model ResNet
```
