# Bidirectional Learning for Robust Neural Networks
This repository contains the complete project for:

Sidney Pontes-Filho and Marcus Liwicki. "[Bidirectional Learning for Robust Neural Networks](https://ieeexplore.ieee.org/document/8852120)". 2019 International Joint Conference on Neural Networks (IJCNN), Budapest, Hungary, 2019, pp. 1-8.

[arXiv Preprint](https://arxiv.org/abs/1805.08006)

Dependencies used:
* Python 3.6.4
* TensorFlow 1.7
* keras 2.1.5
* cleverhans 2.0.0

Installing cleverhans:
```
git clone https://github.com/tensorflow/cleverhans.git
export PYTHONPATH="/path/to/cleverhans":$PYTHONPATH
```
Download MNIST dataset:
```
bash download_mnist.sh
```
Training and testing model:
```
python <dataset>_<model>.py <learning>
```
\<dataset\>:
* mnist
* cifar

\<model\>:
* nn_no_hidden: Bidirectional propagation of errors on fully connected neural network without hidden layer
* nn_one_hidden: Bidirectional propagation of errors on fully connected neural network with one hidden layer
* nn_two_hidden: Bidirectional propagation of errors on fully connected neural network with two hidden layers
* nn_four_hidden: Bidirectional propagation of errors on fully connected neural network with four hidden layers
* cnn_three_conv: Bidirectional propagation of errors on convolutional neural network with three convolutional layers
* gan_cnn_nn_one_hidden: Hybrid adversarial networks on fully connected neural network with one hidden layer
* gan_cnn_two_conv: Hybrid adversarial networks on convolutional neural network with two convolutional layers ([infoGAN](https://arxiv.org/abs/1606.03657) architecture for MNIST)

\<learning\>:

0. Backpropagation
1. Bidirectional learning
2. 1st half Bidirectional learning then 2nd half Backpropagation
3. Backpropagation NO BIAS
4. Bidirectional learning NO BIAS
5. 1st half Bidirectional learning then 2nd half Backpropagation NO BIAS

There are three supporting shell scripts for running several Python scripts:
```
run_all.sh
run_mnist.sh
run_cifar.sh
```
Creating CSV and plots after training and testing:
```
python utils_csv.py
```

For CIFAR-100 dataset, just replace the following lines of code of the scripts for CIFAR-10 dataset:

```
from keras.datasets import cifar10
```
to
```
from keras.datasets import cifar100 as cifar10
```

and

```
num_classes = 10
```
to
```
num_classes = 100
```

Code references for this repository:

https://github.com/martin-gorner/tensorflow-mnist-tutorial

https://github.com/hwalsuklee/tensorflow-generative-model-collections

https://github.com/wiseodd/generative-models

## Citation

```
@INPROCEEDINGS{8852120,
author={S. {Pontes-Filho} and M. {Liwicki}},
booktitle={2019 International Joint Conference on Neural Networks (IJCNN)},
title={Bidirectional Learning for Robust Neural Networks},
year={2019},
volume={},
number={},
pages={1-8},
keywords={adversarial example defense;noise defense;bidirectional learning;hybrid neural network;Hebbian theory},
doi={10.1109/IJCNN.2019.8852120},
ISSN={},
month={July},}
```
