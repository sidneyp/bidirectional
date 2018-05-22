import tensorflow as tf
import keras
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys
import csv
import utils_csv
import utils_tf as utils
from cleverhans.utils_tf import model_train, model_eval
from cleverhans.attacks import FastGradientMethod
from cleverhans.model import Model
print("Tensorflow version " + tf.__version__)

config_num = int(sys.argv[1]) if len(sys.argv) > 1 else 1  # Choose type of learning technique according to config_dict
config_dict = {0: "backprop", 1: "biprop", 2: "halfbiprop", 3: "nobias_backprop", 4: "nobias_biprop", 5: "nobias_halfbiprop"}

num_classes = 10

model_name = sys.argv[0].replace(".py", "") + "_" + config_dict[config_num]
print("Model name: " + model_name)

# load data
# https://github.com/BIGBALLON/cifar-10-cnn/blob/master/1_Lecun_Network/LeNet_keras.py
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# for reproducibility
np.random.seed(0)
tf.set_random_seed(0)

sess = tf.InteractiveSession()

# one hidden layer and its number of neurons
K = 128

Z_dim = num_classes # Random layer

with tf.name_scope("input"):
    # input X & output GX_: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
    X = tf.placeholder(tf.float32, [None, 32, 32, 3])
    X_noisy = tf.placeholder(tf.float32, [None, 32, 32, 3])
    X_adv = tf.placeholder(tf.float32, [None, 32, 32, 3])

    # output Y_ & input GY: labels for classification and generation
    Y_ = tf.placeholder(tf.float32, [None, num_classes])

    # random input for Generator
    Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

    input_test_sum = tf.summary.image("input", X, num_classes)
    input_noisy_sum = tf.summary.image("input-noisy", X_noisy, num_classes)
    input_adv_sum = tf.summary.image("input-adv", X_adv, num_classes)

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

with tf.name_scope("classifier-generator"):
    # Weights for classifier and generator
    C_W1 = utils.weight_variable_xavier_initialized([32*32*3, K], name="C_W1")
    C_W2 = utils.weight_variable_xavier_initialized([K, num_classes], name="C_W2")

def classifier(x, reuse=None):
    with tf.variable_scope("classifier", reuse=reuse) as scope_c:
        # Variables for classifier
        C_B1 = utils.bias_variable([K], name="C_B1")
        C_B2 = utils.bias_variable([Z_dim], name="C_B2")

        XX = tf.reshape(x, [-1, 32*32*3])
        H1 = tf.nn.relu(tf.matmul(XX, C_W1) + C_B1)
        Ylogits = tf.matmul(H1, C_W2) + C_B2

        Ysigmoid = tf.nn.sigmoid(Ylogits)
        Ysoftmax = tf.nn.softmax(Ylogits)

        return Ysoftmax, Ysigmoid, Ylogits


class ClassifierModel(Model):
    def get_logits(self, x):
        Ysoftmax, Ysigmoid, Ylogits = classifier(x, reuse=True)
        return Ylogits

# Generator of random input reuses weights of classifier
def generator(z, reuse=None):
    with tf.variable_scope("generator", reuse=reuse) as scope_g:
        # Variables for classifier
        G_B1 = utils.bias_variable([32*32*3], name="G_B1")
        G_B2 = utils.bias_variable([K], name="G_B2")

        GH1 = tf.nn.relu(tf.matmul(z, tf.transpose(C_W2)) + G_B2)
        GX = tf.matmul(GH1, tf.transpose(C_W1)) + G_B1
        GXlogits = tf.reshape(GX, [-1, 32, 32, 3])
        GXsigmoid = tf.nn.sigmoid(GXlogits)

        return GXsigmoid, GXlogits

def discriminator(x, reuse=None):
    with tf.variable_scope("discriminator", reuse=reuse) as scope:
        # Variables for classifier
        D_W1 = utils.weight_variable_xavier_initialized([32*32*3, K], name="D_W1")
        D_B1 = utils.bias_variable([K], name="D_B1")
        D_W2 = utils.weight_variable_xavier_initialized([K, 1], name="D_W2")
        D_B2 = utils.bias_variable([1], name="D_B2")

        XX = tf.reshape(x, [-1, 32*32*3])
        H1 = tf.nn.relu(tf.matmul(XX, D_W1) + D_B1)
        Ylogits = tf.matmul(H1, D_W2) + D_B2

        Ysigmoid = tf.nn.sigmoid(Ylogits)
        Ysoftmax = tf.nn.softmax(Ylogits)

        return Ysoftmax, Ysigmoid, Ylogits

def plot_generator(samples, figsize=[5,5]):
    fig = plt.figure(figsize=(figsize[0], figsize[1]))
    gs = gridspec.GridSpec(figsize[1], figsize[0])
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape((32,32,3)), cmap='gray')

    return fig

GXsigmoid, GXlogits = generator(Z)
GXsigmoid_test, GXlogits_test = generator(Z, reuse=True)

Ysoftmax, Ysigmoid, Ylogits = classifier(X)
model_classifier = ClassifierModel()

Ysoftmax_noisy, Ysigmoid_noisy, Ylogits_noisy = classifier(X_noisy, reuse=True)
Ysoftmax_adv, Ysigmoid_adv, Ylogits_adv = classifier(X_adv, reuse=True)

Ysoftmax_real, Ysigmoid_real, Ylogits_real = discriminator(X)
Ysoftmax_fake, Ysigmoid_fake, Ylogits_fake = discriminator(GXsigmoid, reuse=True)

with tf.name_scope("loss"):
    c_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_))

    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Ylogits_real, labels=tf.ones_like(Ylogits_real)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Ylogits_fake, labels=tf.zeros_like(Ylogits_fake)))
    d_loss = d_loss_real + d_loss_fake

    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Ylogits_fake, labels=tf.ones_like(Ylogits_fake)))

    """ Summary """
    d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
    d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
    d_loss_sum = tf.summary.scalar("d_loss", d_loss)
    g_loss_sum = tf.summary.scalar("g_loss", g_loss)
    c_loss_sum = tf.summary.scalar("c_loss", c_loss)

# accuracy of the trained model, between 0 (worst) and 1 (best)
with tf.name_scope("accuracy"):
    with tf.name_scope("correct_prediction"):
        correct_prediction = tf.equal(tf.argmax(Ysoftmax, 1), tf.argmax(Y_, 1))
    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.name_scope("correct_prediction_noisy"):
        correct_prediction_noisy = tf.equal(tf.argmax(Ysoftmax_noisy, 1), tf.argmax(Y_, 1))
    with tf.name_scope("accuracy_noisy"):
        accuracy_noisy = tf.reduce_mean(tf.cast(correct_prediction_noisy, tf.float32))
    with tf.name_scope("correct_prediction_adv"):
        correct_prediction_adv = tf.equal(tf.argmax(Ysoftmax_adv, 1), tf.argmax(Y_, 1))
    with tf.name_scope("accuracy_adv"):
        accuracy_adv = tf.reduce_mean(tf.cast(correct_prediction_adv, tf.float32))

    """ Summary """
    accuracy_sum = tf.summary.scalar("accuracy", accuracy)
    accuracy_noisy_sum = tf.summary.scalar("accuracy_noisy", accuracy_noisy)
    accuracy_adv_sum = tf.summary.scalar("accuracy_adv", accuracy_adv)

with tf.name_scope("max_output"):
    with tf.name_scope("max_output_test"):
        max_output_sigmoid_test = tf.reduce_max(Ysigmoid)
        max_output_softmax_test = tf.reduce_max(Ysoftmax)
    with tf.name_scope("max_output_noise"):
        max_output_sigmoid_noise = tf.reduce_max(Ysigmoid_noisy)
        max_output_softmax_noise = tf.reduce_max(Ysoftmax_noisy)
    with tf.name_scope("max_output_adv"):
        max_output_sigmoid_adv = tf.reduce_max(Ysigmoid_adv)
        max_output_softmax_adv = tf.reduce_max(Ysoftmax_adv)

    """ Summary """
    max_output_sigmoid_test_sum = tf.summary.scalar("max_output_sigmoid_test", max_output_sigmoid_test)
    max_output_softmax_test_sum = tf.summary.scalar("max_output_softmax_test", max_output_softmax_test)
    max_output_sigmoid_noise_sum = tf.summary.scalar("max_output_sigmoid_noise", max_output_sigmoid_noise)
    max_output_softmax_noise_sum = tf.summary.scalar("max_output_softmax_noise", max_output_softmax_noise)
    max_output_sigmoid_adv_sum = tf.summary.scalar("max_output_sigmoid_adv", max_output_sigmoid_adv)
    max_output_softmax_adv_sum = tf.summary.scalar("max_output_softmax_adv", max_output_softmax_adv)

utils.show_all_variables()
t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'D_' in var.name]
c_vars = [var for var in t_vars if 'C_' in var.name]\
    if config_num < 3 else [var for var in t_vars if 'C_W' in var.name]
g_vars = [var for var in t_vars if 'C_W' in var.name or 'G_' in var.name]\
    if config_num < 3 else c_vars

# training step
learning_rate_dis = 0.0002
learning_rate_gen = 0.001
beta1 = 0.5

with tf.name_scope("train"):
    c_train = tf.train.AdamOptimizer(learning_rate_dis, beta1=beta1).minimize(c_loss, var_list=c_vars)
    d_train = tf.train.AdamOptimizer(learning_rate_dis, beta1=beta1).minimize(d_loss, var_list=d_vars)
    g_train = tf.train.AdamOptimizer(learning_rate_gen, beta1=beta1).minimize(g_loss, var_list=g_vars)

# final summary operations
g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])
c_sum = tf.summary.merge([input_test_sum, accuracy_sum, c_loss_sum, max_output_sigmoid_test_sum, max_output_softmax_test_sum])
noise_sum = tf.summary.merge([max_output_sigmoid_noise_sum, max_output_softmax_noise_sum])
noisy_sum = tf.summary.merge([input_noisy_sum, accuracy_noisy_sum])
adv_sum = tf.summary.merge([input_adv_sum, accuracy_adv_sum, max_output_sigmoid_adv_sum, max_output_softmax_adv_sum])

folder_out = 'out/' + model_name + '/'
if not os.path.exists(folder_out):
    os.makedirs(folder_out)

folder_csv = 'csv/' + model_name + '/'
if not os.path.exists(folder_csv):
    os.makedirs(folder_csv)

folder_logs = 'logs/' + model_name
if not os.path.exists(folder_csv):
    os.makedirs(folder_logs)

writer = tf.summary.FileWriter(folder_logs, sess.graph)

batch_size = 100
num_train_images = x_train.shape[0]
num_batches =  num_train_images // batch_size
all_classes = np.eye(num_classes)

counter = 0

fgsm_params = {'eps': 0.03,
               'clip_min': 0.,
               'clip_max': 1.}

random_noise = np.random.random_sample(x_test.shape)
test_image_with_noise = np.clip(x_test + 0.1*random_noise, 0., 1.)

accuracy_list = []
sigmoid_list = []
softmax_list = []

# initialize all variables
tf.global_variables_initializer().run()

for i in range(500001):
    if i % num_batches == 0:
        idx_train = np.arange(x_train.shape[0])
        np.random.shuffle(idx_train)
        x_train, y_train = x_train[idx_train], y_train[idx_train]
    
    idx = i % num_batches
    batch_X = x_train[idx*batch_size:(idx+1)*batch_size]
    batch_Y = y_train[idx*batch_size:(idx+1)*batch_size]

    if i % 5000 == 0 or i == 500000:
        counter += 1
        # Saves generated images
        samples = sess.run(GXsigmoid_test, feed_dict={Z: sample_Z(25, Z_dim)})
        fig = plot_generator(samples)
        plt.savefig(folder_out+"gen_"+str(i).zfill(6)+'.png', bbox_inches='tight')
        plt.close(fig)

        attack_fgsm = FastGradientMethod(model_classifier, sess=sess)
        adv_x_np = attack_fgsm.generate_np(x_test, **fgsm_params)
        fig = plot_generator(adv_x_np[:25])
        plt.savefig(folder_out+"adv_"+str(i).zfill(6)+'.png', bbox_inches='tight')
        plt.close(fig)

        accu_test, c_loss_test, sigmoid_test, softmax_test, sum_c = sess.run([accuracy, c_loss, max_output_sigmoid_test, max_output_softmax_test, c_sum], {X: x_test, Y_: y_test})
        writer.add_summary(sum_c, i)
        d_loss_test, sum_d = sess.run([d_loss, d_sum], {X: batch_X, Z: sample_Z(batch_size, Z_dim)})
        writer.add_summary(sum_d, i)
        g_loss_test, sum_g = sess.run([g_loss, g_sum], {Z: sample_Z(batch_size, Z_dim)})
        writer.add_summary(sum_g, i)

        print(str(i) + ": epoch " + str(i*batch_size//x_train.shape[0]+1)\
            + " - test loss class: " + str(c_loss_test) + " test loss gen: " + str(g_loss_test) + " test loss dis: " + str(d_loss_test))
        print("Real test images     - Sigmoid: " + str(sigmoid_test) + "\tSoftmax: " + str(softmax_test) + "\taccuracy: "+ str(accu_test))

        sigmoid_random, softmax_random, sum_random = sess.run([max_output_sigmoid_noise, max_output_softmax_noise, noise_sum], {X_noisy: random_noise})
        writer.add_summary(sum_random, i)
        accu_random, sum_noisy = sess.run([accuracy_noisy, noisy_sum], {X_noisy: test_image_with_noise, Y_: y_test})
        writer.add_summary(sum_noisy, i)
        print("Random noise images  - Sigmoid: " + str(sigmoid_random) + "\tSoftmax: " + str(softmax_random) + "\taccuracy: "+ str(accu_random))

        accu_adv, sigmoid_adv, softmax_adv, sum_adv = sess.run([accuracy_adv, max_output_sigmoid_adv, max_output_softmax_adv, adv_sum], {X_adv: adv_x_np, Y_: y_test})
        writer.add_summary(sum_adv, i)
        print("Adversarial examples - Sigmoid: " + str(sigmoid_adv) + "\tSoftmax: " + str(softmax_adv) + "\taccuracy: "+ str(accu_adv))
        print()
        accuracy_list.append([i, accu_test, accu_random, accu_adv, counter])
        sigmoid_list.append([i, sigmoid_test, sigmoid_random, sigmoid_adv, counter])
        softmax_list.append([i, softmax_test, softmax_random, softmax_adv, counter])

    sess.run(c_train, {X: batch_X, Y_: batch_Y})
    if config_num == 1 or (config_num == 2 and i < 250000) or\
        config_num == 4 or (config_num == 5 and i < 250000):
        sess.run(d_train, {X: batch_X, Z: sample_Z(batch_size, Z_dim)})
        sess.run(g_train, {Z: sample_Z(batch_size, Z_dim)})
writer.close()

# Save data in csv
with open(folder_csv+"accuracy.csv", "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(accuracy_list)

with open(folder_csv+"sigmoid.csv", "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(sigmoid_list)

with open(folder_csv+"softmax.csv", "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(softmax_list)

# Load data in csv
accu_data = utils_csv.get_data_csv_file(folder_csv+"accuracy.csv")
sigmoid_data = utils_csv.get_data_csv_file(folder_csv+"sigmoid.csv")
softmax_data = utils_csv.get_data_csv_file(folder_csv+"softmax.csv")

# Print best values
utils_csv.print_best(accu_data, sigmoid_data, softmax_data, folder_csv+"summary.txt")
