# Neural Architecture Search with Reinforcement Learning
> `model_lenet` and `model_dummy` have different code. Check out `model_lenet` for better explanation.
## Dataset

The following model was trained on the CIFAR-10 dataset. To get the data and data-prep. related filed contact me.

## Working
This application is an implementation of Neural Architecture Search which uses a recurrent neural network to generate the hyperparameters. We use a softmax layer, and let the network "choose" between multiple choices which we provide (hard-coded), and construct a architecture which is trained, and the validation accuracy is calculated. The validation accuracy is used as the reward signal, and the goal is to maximize using it. For this, we use Gradient Ascent, and calculate the gradients using the REINFORCE algorithm. 

I have tried out 2 implementations (in seperate folders). This is a minimal implementation of the algorithm and it's highly likely that it might not work.

## Installation
This script was built and tested on python3, so make sure you use pip3!
Install Tensorflow from the official website, that should install all the other dependencies too, hopefully.
