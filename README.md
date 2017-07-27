# Neural Architecture Search for ScatNet
## Dataset

The following model was trained on the CIFAR-10 dataset. To get the data and data-prep. related filed contact me.

## Working
This application is an implementation of Neural Architecture Search which uses a recurrent neural network to generate the hyperparameters. We use a softmax layer, and let the network "choose" between multiple choices which we provide (hard-coded), and construct a architecture (currently a single conv-layered CNN) which is trained, and the validation accuracy is calculated. The validation accuracy is used as the reward signal, and the goal is to maximize using it. For this, we use Gradient Ascent, and calculate the gradients using the REINFORCE algorithm. 

## Installation
This script was built and tested on python3, so make sure you use pip3!
Install Tensorflow from the official website, that should install all the other dependencies too, hopefully.

## Running 
```shell
cd ./model/src
python3 __main__.py
```
