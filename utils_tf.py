"""
Based on:
https://github.com/shekkizh/EBGAN.tensorflow/blob/master/TensorflowUtils.py
"""

# Utils used with tensorflow implemetation
import tensorflow as tf
import numpy as np
import scipy.misc as misc
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def weight_variable(shape, stddev=0.02, name=None):
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)

def weight_variable_xavier_initialized(shape, name=None):
    # https://github.com/wiseodd/generative-models/blob/master/GAN/vanilla_gan/gan_tensorflow.py
    in_dim = shape[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)    
    return weight_variable(shape, stddev=xavier_stddev, name=name)

def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)

def bn(x, is_training, scope):
    # https://github.com/hwalsuklee/tensorflow-generative-model-collections/blob/master/ops.py
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        updates_collections=None,
                                        epsilon=1e-5,
                                        scale=True,
                                        is_training=is_training,
                                        scope=scope)

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)
