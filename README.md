# Bidirectional Learning for Robust Neural Networks
This repository contains the complete project for:

Sidney Pontes-Filho and Marcus Liwicki. "[Bidirectional Learning for Robust Neural Networks](https://arxiv.org/abs/1805.08006)". arXiv. 2018.

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
* nn_one_hidden: Bidirectional propagation of errors on fully connected neural network without hidden layer
* nn_two_hidden: Bidirectional propagation of errors on fully connected neural network without hidden layer
* nn_four_hidden: Bidirectional propagation of errors on fully connected neural network without hidden layer
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

Code references for this repository:

https://github.com/martin-gorner/tensorflow-mnist-tutorial

https://github.com/hwalsuklee/tensorflow-generative-model-collections

https://github.com/wiseodd/generative-models
