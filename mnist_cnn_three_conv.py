import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
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

model_name = sys.argv[0].replace(".py", "") + "_" + config_dict[config_num]
print("Model name: " + model_name)

# for reproducibility
np.random.seed(0)
tf.set_random_seed(0)

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = input_data.read_data_sets("data/mnist", one_hot=True, reshape=False, validation_size=0)

sess = tf.InteractiveSession()

# three convolutional layers with their channel counts, and a
# fully connected layer (tha last layer has 10 softmax neurons)
K = 4  # first convolutional layer output depth
L = 8  # second convolutional layer output depth
M = 12  # third convolutional layer
N = 200  # fully connected layer

with tf.name_scope("input"):
    # input X & output GX_: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
    X = tf.placeholder(tf.float32, [None, 28, 28, 1])
    X_noisy = tf.placeholder(tf.float32, [None, 28, 28, 1])
    X_adv = tf.placeholder(tf.float32, [None, 28, 28, 1])

    GX_ = tf.placeholder(tf.float32, [None, 28, 28, 1])

    # output Y_ & input GY: labels for classification and generation
    Y_ = tf.placeholder(tf.float32, [None, 10])
    GY = tf.placeholder(tf.float32, [None, 10])

    # variable learning rate
    lr = tf.placeholder(tf.float32)

    # variable batch size
    BS = tf.placeholder(tf.int32)

    input_test_sum = tf.summary.image("input", X, 10)
    input_noisy_sum = tf.summary.image("input-noisy", X_noisy, 10)
    input_adv_sum = tf.summary.image("input-adv", X_adv, 10)

with tf.name_scope("classifier-generator"):
    C_W1 = utils.weight_variable([5, 5, 1, K], stddev=0.1, name="C_W1")
    C_W2 = utils.weight_variable([5, 5, K, L], stddev=0.1, name="C_W2")
    C_W3 = utils.weight_variable([4, 4, L, M], stddev=0.1, name="C_W3")

    C_W4 = utils.weight_variable([7 * 7 * M, N], stddev=0.1, name="C_W4")
    C_W5 = utils.weight_variable([N, 10], stddev=0.1, name="C_W5")

def classifier(x, reuse=None):
    with tf.variable_scope("classifier", reuse=reuse) as scope_c:
        # Variables for classifier
        C_B1 = utils.bias_variable([K], name="C_B1")
        C_B2 = utils.bias_variable([L], name="C_B2")
        C_B3 = utils.bias_variable([M], name="C_B3")
        C_B4 = utils.bias_variable([N], name="C_B4")
        C_B5 = utils.bias_variable([10], name="C_B5")

        stride = 1  # output is 28x28
        H1 = tf.nn.relu(tf.nn.conv2d(x, C_W1, strides=[1, stride, stride, 1], padding='SAME') + C_B1)
        stride = 2  # output is 14x14
        H2 = tf.nn.relu(tf.nn.conv2d(H1, C_W2, strides=[1, stride, stride, 1], padding='SAME') + C_B2)
        stride = 2  # output is 7x7
        H3 = tf.nn.relu(tf.nn.conv2d(H2, C_W3, strides=[1, stride, stride, 1], padding='SAME') + C_B3)

        # reshape the output from the third convolution for the fully connected layer
        HH3 = tf.reshape(H3, shape=[-1, 7 * 7 * M])

        H4 = tf.nn.relu(tf.matmul(HH3, C_W4) + C_B4)
        Ylogits = tf.matmul(H4, C_W5) + C_B5

        Ysigmoid = tf.nn.sigmoid(Ylogits)
        Ysoftmax = tf.nn.softmax(Ylogits)

        return Ysoftmax, Ysigmoid, Ylogits

class ClassifierModel(Model):
    def get_logits(self, x):
        Ysoftmax, Ysigmoid, Ylogits = classifier(x, reuse=True)
        return Ylogits

# Generator of random input reuses weights of classifier
def generator(y, bs, reuse=None):
    with tf.variable_scope("generator", reuse=reuse) as scope_g:
        # Variables for classifier
        G_B1 = utils.bias_variable([1], name="G_B1")
        G_B2 = utils.bias_variable([K], name="G_B2")
        G_B3 = utils.bias_variable([L], name="G_B3")
        G_B4 = utils.bias_variable([M*7*7], name="G_B4")
        G_B5 = utils.bias_variable([N], name="G_B5")

        GH4 = tf.nn.relu(tf.matmul(y, tf.transpose(C_W5)) + G_B5)
        GH3 = tf.nn.relu(tf.matmul(GH4, tf.transpose(C_W4)) + G_B4)
        GHH3 = tf.reshape(GH3, shape=[-1, 7, 7, M])
        stride = 2  # output is 14x14
        GH2 = tf.nn.relu(tf.nn.conv2d_transpose(GHH3, C_W3, output_shape=[bs, 14, 14, L], strides=[1, stride, stride, 1]) + G_B3)  #deconv2 W3
        stride = 2  # output is 28x28
        GH1 = tf.nn.relu(tf.nn.conv2d_transpose(GH2, C_W2, output_shape=[bs, 28, 28, K], strides=[1, stride, stride, 1]) + G_B2)#deconv2 W2
        stride = 1  # output is 28x28
        GXlogits = tf.nn.conv2d_transpose(GH1, C_W1, output_shape=[bs, 28, 28, 1], strides=[1, stride, stride, 1]) + G_B1#deconv2 W1
        GXsigmoid = tf.nn.sigmoid(GXlogits)

        return GXsigmoid, GXlogits

def plot_generator(samples):
    fig = plt.figure(figsize=(5, 2))
    gs = gridspec.GridSpec(2, 5)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape((28,28)), cmap='gray')

    return fig

GXsigmoid, GXlogits = generator(GY, BS)
GXsigmoid_test, GXlogits_test = generator(GY, BS, reuse=True)

Ysoftmax, Ysigmoid, Ylogits = classifier(X)
model_classifier = ClassifierModel()

Ysoftmax_noisy, Ysigmoid_noisy, Ylogits_noisy = classifier(X_noisy, reuse=True)
Ysoftmax_adv, Ysigmoid_adv, Ylogits_adv = classifier(X_adv, reuse=True)

with tf.name_scope("loss"):
    c_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_))

    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=GXlogits, labels=GX_))

    """ Summary """
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
c_vars = [var for var in t_vars if 'C_' in var.name]\
    if config_num < 3 else [var for var in t_vars if 'C_W' in var.name]
g_vars = [var for var in t_vars if 'C_W' in var.name or 'G_' in var.name]\
    if config_num < 3 else c_vars

# training step
learning_rate_dis = lr
learning_rate_gen = 10*lr

with tf.name_scope("train"):
    c_train = tf.train.AdamOptimizer(learning_rate_dis).minimize(c_loss, var_list=c_vars)
    g_train = tf.train.AdamOptimizer(learning_rate_gen).minimize(g_loss, var_list=g_vars)

# final summary operations
g_sum = tf.summary.merge([g_loss_sum])
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
num_train_images = mnist.train.images.shape[0]
num_batches =  num_train_images // batch_size
all_classes = np.eye(10)

counter = 0

fgsm_params = {'eps': 0.3,
               'clip_min': 0.,
               'clip_max': 1.}

random_noise = np.random.random_sample(mnist.test.images.shape)
test_image_with_noise = np.clip(mnist.test.images + 0.1*random_noise, 0., 1.)

accuracy_list = []
sigmoid_list = []
softmax_list = []

# initialize all variables
tf.global_variables_initializer().run()

for i in range(50001):
    batch_X, batch_Y = mnist.train.next_batch(batch_size)

    # learning rate decay
    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 2000.0
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * np.exp(-i/decay_speed)

    if i % 500 == 0 or i == 50000:
        counter += 1
        # Saves generated images
        samples = sess.run(GXsigmoid_test, feed_dict={GY: all_classes, BS: 10})
        fig = plot_generator(samples)
        plt.savefig(folder_out+"gen_"+str(i).zfill(6)+'.png', bbox_inches='tight')
        plt.close(fig)

        attack_fgsm = FastGradientMethod(model_classifier, sess=sess)
        adv_x_np = attack_fgsm.generate_np(mnist.test.images, **fgsm_params)
        fig = plot_generator(adv_x_np[:10])
        plt.savefig(folder_out+"adv_"+str(i).zfill(6)+'.png', bbox_inches='tight')
        plt.close(fig)

        accu_test, c_loss_test, sigmoid_test, softmax_test, sum_c = sess.run([accuracy, c_loss, max_output_sigmoid_test, max_output_softmax_test, c_sum], {X: mnist.test.images, Y_: mnist.test.labels})
        writer.add_summary(sum_c, i)
        g_loss_test, sum_g = sess.run([g_loss, g_sum], {GY: batch_Y, GX_: batch_X, BS: batch_size})
        writer.add_summary(sum_g, i)

        print(str(i) + ": epoch " + str(i*batch_size//mnist.train.images.shape[0]+1)\
            + " - test loss class: " + str(c_loss_test) + " test loss gen: " + str(g_loss_test))
        print("Real test images     - Sigmoid: " + str(sigmoid_test) + "\tSoftmax: " + str(softmax_test) + "\taccuracy: "+ str(accu_test))

        sigmoid_random, softmax_random, sum_random = sess.run([max_output_sigmoid_noise, max_output_softmax_noise, noise_sum], {X_noisy: random_noise})
        writer.add_summary(sum_random, i)
        accu_random, sum_noisy = sess.run([accuracy_noisy, noisy_sum], {X_noisy: test_image_with_noise, Y_: mnist.test.labels})
        writer.add_summary(sum_noisy, i)
        print("Random noise images  - Sigmoid: " + str(sigmoid_random) + "\tSoftmax: " + str(softmax_random) + "\taccuracy: "+ str(accu_random))

        accu_adv, sigmoid_adv, softmax_adv, sum_adv = sess.run([accuracy_adv, max_output_sigmoid_adv, max_output_softmax_adv, adv_sum], {X_adv: adv_x_np, Y_: mnist.test.labels})
        writer.add_summary(sum_adv, i)
        print("Adversarial examples - Sigmoid: " + str(sigmoid_adv) + "\tSoftmax: " + str(softmax_adv) + "\taccuracy: "+ str(accu_adv))
        print()
        accuracy_list.append([i, accu_test, accu_random, accu_adv, counter])
        sigmoid_list.append([i, sigmoid_test, sigmoid_random, sigmoid_adv, counter])
        softmax_list.append([i, softmax_test, softmax_random, softmax_adv, counter])

    sess.run(c_train, {X: batch_X, Y_: batch_Y, lr: learning_rate})
    if config_num == 1 or (config_num == 2 and i < 25000) or\
        config_num == 4 or (config_num == 5 and i < 25000):
        sess.run(g_train, {GY: batch_Y, GX_: batch_X, lr: learning_rate, BS: batch_size})

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
