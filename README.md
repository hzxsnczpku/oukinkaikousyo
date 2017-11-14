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
 * Vgg19
 * ResNet
 * DenseNet
 * ResNext
 * Xception


## Experiment results

| model | training accuracy | training loss | test accuracy | test loss | trained model
| :----: | :-----: | :----: | :----: | :----: | :----: |
| The diy model | 99.97% | 0.9100 | 78.81% | Not Recorded | https://pan.baidu.com/s/1o7NhJ1w |
| Vgg19 | 99.75% | 0.6149 | 91.75% | 1.1425 | https://pan.baidu.com/s/1bVDB34 |
| ResNet | 99.99% | 0.1433 | 96.67% | 0.3012 | https://pan.baidu.com/s/1dEBoKZR |
| DenseNet | under experiment | under experiment | under experiment | under experiment | pass |
| ResNext | under experiment | under experiment | under experiment | under experiment | pass |

## Training Process

Below is the data derived from my training process.

### Vgg19
<div align = 'center'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/oukinkaikousyo/master/images/Vgg19.png" width='600px'>
</div>

### ResNet
<div align = 'center'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/oukinkaikousyo/master/images/ResNet.png" width='600px'>
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
