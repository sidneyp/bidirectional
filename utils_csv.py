import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import csv
import sys
import argparse
import glob
import datetime

def get_data_csv_file(data_file):
    data_x = []
    data_test = []
    data_random = []
    data_adv = []
    data_counter = []
    with open(data_file) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data_x.append(int(row[0]))
            data_test.append(float(row[1]))
            data_random.append(float(row[2]))
            data_adv.append(float(row[3]))
            data_counter.append(int(row[4]))
    return (data_x, data_test,data_random, data_adv, data_counter)

def plot_accuracy(accu_x, accu_test,accu_random, accu_adv):
    plt.plot(accu_x,accu_test,label='Test data')
    plt.plot(accu_x,accu_random,label='Test data with noise')
    plt.plot(accu_x,accu_adv,label='Test data with FGSM')

    plt.legend(loc=0)
    plt.grid(True)
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.xlim([0, 10000])
    plt.ylim([0., 1.1])

    plt.show()

def plot_sigmoid(sigmoid_x, sigmoid_test,sigmoid_random, sigmoid_adv):
    plt.plot(sigmoid_x,sigmoid_test,label='Test data')
    plt.plot(sigmoid_x,sigmoid_random,label='White noise')
    plt.plot(sigmoid_x,sigmoid_adv,label='Test data with FGSM')

    plt.legend(loc=0)
    plt.grid(True)
    plt.xlabel("Iteration")
    plt.ylabel("Maximum sigmoid output")
    plt.xlim([0, 10000])
    plt.ylim([0., 1.1])
    plt.show()

def plot_softmax(softmax_x, softmax_test,softmax_random, softmax_adv):
    plt.plot(softmax_x,softmax_test,label='Test data')
    plt.plot(softmax_x,softmax_random,label='White noise')
    plt.plot(softmax_x,softmax_adv,label='Test data with FGSM')

    plt.legend(loc=0)
    plt.grid(True)
    plt.xlabel("Iteration")
    plt.ylabel("Maximum softmax output")
    plt.xlim([0, 10000])
    plt.ylim([0., 1.1])
    plt.show()

def print_best(accu_data, sigmoid_data, softmax_data, fileout):
    x, accu_test,accu_random, accu_adv, counter = \
        accu_data[0], accu_data[1], accu_data[2], accu_data[3], accu_data[4]
    sigmoid_test,sigmoid_random, sigmoid_adv = \
        sigmoid_data[1], sigmoid_data[2], sigmoid_data[3]
    softmax_test,softmax_random, softmax_adv = \
        softmax_data[1], softmax_data[2], softmax_data[3]

    best_accu_test_idx = np.argmax(accu_test)
    best_accu_random_idx = np.argmax(accu_random)
    best_accu_adv_idx = np.argmax(accu_adv)

    sigmoid_rate = np.array(sigmoid_random) / np.array(sigmoid_test)
    best_sigmoid_rate_idx = np.argmin(sigmoid_rate)

    softmax_rate = np.array(softmax_random) / np.array(softmax_test)
    best_softmax_rate_idx = np.argmin(softmax_rate)

    print_txt = ""

    print_txt = print_txt +"Best accuracy test data:\n"
    print_txt = print_txt +"Index: " + str(best_accu_test_idx) +\
        " Iteration: " + str(x[best_accu_test_idx]) +\
        " Epoch: " + str(counter[best_accu_test_idx]) + "\n"
    print_txt = print_txt +"Test data: " + str(accu_test[best_accu_test_idx]) + "\n"
    print_txt = print_txt +"Test data with noise: " + str(accu_random[best_accu_test_idx]) + "\n"
    print_txt = print_txt +"Test data with FGSM: " + str(accu_adv[best_accu_test_idx]) + "\n"
    print_txt = print_txt +"Sigmoid rate of maximum output of noise over real: "+\
        str(sigmoid_rate[best_accu_test_idx]) + "\n"
    print_txt = print_txt +"Softmax rate of maximum output of noise over real: "+\
        str(softmax_rate[best_accu_test_idx]) + "\n"
    print_txt = print_txt + "\n"
    print_txt = print_txt +"Best accuracy test data with noise:\n"
    print_txt = print_txt +"Index: " + str(best_accu_random_idx) +\
        " Iteration: " + str(x[best_accu_random_idx]) +\
        " Epoch: " + str(counter[best_accu_random_idx]) + "\n"
    print_txt = print_txt +"Test data: " + str(accu_test[best_accu_random_idx]) + "\n"
    print_txt = print_txt +"Test data with noise: " + str(accu_random[best_accu_random_idx]) + "\n"
    print_txt = print_txt +"Test data with FGSM: " + str(accu_adv[best_accu_random_idx]) + "\n"
    print_txt = print_txt +"Sigmoid rate of maximum output of noise over real: "+\
        str(sigmoid_rate[best_accu_random_idx]) + "\n"
    print_txt = print_txt +"Softmax rate of maximum output of noise over real: "+\
        str(softmax_rate[best_accu_random_idx])+ "\n"
    print_txt = print_txt + "\n"
    print_txt = print_txt +"Best accuracy test data with FGSM:\n"
    print_txt = print_txt +"Index: " + str(best_accu_adv_idx) +\
        " Iteration: " + str(x[best_accu_adv_idx]) +\
        " Epoch: " + str(counter[best_accu_adv_idx]) + "\n"
    print_txt = print_txt +"Test data: " + str(accu_test[best_accu_adv_idx])+ "\n"
    print_txt = print_txt +"Test data with noise: " + str(accu_random[best_accu_adv_idx])+ "\n"
    print_txt = print_txt +"Test data with FGSM: " + str(accu_adv[best_accu_adv_idx])+ "\n"
    print_txt = print_txt +"Sigmoid rate of maximum output of noise over real: "+\
        str(sigmoid_rate[best_accu_adv_idx])+ "\n"
    print_txt = print_txt +"Softmax rate of maximum output of noise over real: "+\
        str(softmax_rate[best_accu_adv_idx])+ "\n"
    print_txt = print_txt + "\n"
    print_txt = print_txt +"Best sigmoid rate of maximum output of noise over real: "\
            + str(sigmoid_rate[best_sigmoid_rate_idx])+ "\n"
    print_txt = print_txt +"Index: " + str(best_sigmoid_rate_idx) +\
        " Iteration: " + str(x[best_sigmoid_rate_idx]) +\
        " Epoch: " + str(counter[best_sigmoid_rate_idx]) + "\n"
    print_txt = print_txt +"Accuracy:\n"
    print_txt = print_txt +"Test data: " + str(accu_test[best_sigmoid_rate_idx])+ "\n"
    print_txt = print_txt +"Test data with noise: " + str(accu_random[best_sigmoid_rate_idx])+ "\n"
    print_txt = print_txt +"Test data with FGSM: " + str(accu_adv[best_sigmoid_rate_idx])+ "\n"
    print_txt = print_txt + "\n"
    print_txt = print_txt +"Best softmax rate of maximum output of noise over real: "\
            + str(softmax_rate[best_softmax_rate_idx])+ "\n"
    print_txt = print_txt +"Index: " + str(best_softmax_rate_idx) +\
        " Iteration: " + str(x[best_softmax_rate_idx]) +\
        " Epoch: " + str(counter[best_softmax_rate_idx]) + "\n"
    print_txt = print_txt +"Accuracy:\n"
    print_txt = print_txt +"Test data: " + str(accu_test[best_softmax_rate_idx])+ "\n"
    print_txt = print_txt +"Test data with noise: " + str(accu_random[best_softmax_rate_idx])+ "\n"
    print_txt = print_txt +"Test data with FGSM: " + str(accu_adv[best_softmax_rate_idx])+ "\n"

    print(print_txt)

    with open(fileout, "w") as fout:
        fout.write(print_txt)

    return print_txt

def process_csv_best(csv_folder = "csv"):
    time_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    fileout = os.path.join(csv_folder, 'summary_'+time_str+'.csv')

    header_best_accu_test = ["best_accu_test_iter","best_accu_test_test",\
        "best_accu_test_noise", "best_accu_test_adv", "best_accu_test_sigmoid",\
        "best_accu_test_softmax"]
    header_best_accu_noise = ["best_accu_noise_iter","best_accu_noise_test",\
        "best_accu_noise_noise", "best_accu_noise_adv", "best_accu_noise_sigmoid",\
        "best_accu_noise_softmax"]
    header_best_accu_adv = ["best_accu_adv_iter","best_accu_adv_test",\
        "best_accu_adv_noise", "best_accu_adv_adv", "best_accu_adv_sigmoid",\
        "best_accu_adv_softmax"]
    header_best_sigmoid = ["best_sigmoid_iter","best_sigmoid_test",\
        "best_sigmoid_noise", "best_sigmoid_adv", "best_sigmoid_sigmoid",\
        "best_sigmoid_softmax"]
    header_best_softmax = ["best_softmax_iter","best_softmax_test",\
        "best_softmax_noise", "best_softmax_adv", "best_softmax_sigmoid",\
        "best_softmax_softmax"]

    header_list = ["model"] + header_best_accu_test + header_best_accu_noise\
        + header_best_accu_adv + header_best_sigmoid + header_best_softmax
    summary_list = [header_list]
    for path in sorted(glob.glob( os.path.join(csv_folder, '*') )):
        if os.path.isdir(path):
            accu_csv = os.path.join(path, 'accuracy.csv')
            sigmoid_csv = os.path.join(path, 'sigmoid.csv')
            softmax_csv = os.path.join(path, 'softmax.csv')

            if os.path.exists(accu_csv) and os.path.exists(sigmoid_csv)\
                and os.path.exists(softmax_csv):
                model_name = path.split("/")[-1]
                print(model_name)

                data_list = [model_name]
                accu_data = get_data_csv_file(accu_csv)
                sigmoid_data = get_data_csv_file(sigmoid_csv)
                softmax_data = get_data_csv_file(softmax_csv)

                x, accu_test,accu_random, accu_adv, counter = \
                    accu_data[0], accu_data[1], accu_data[2], accu_data[3], accu_data[4]
                sigmoid_test,sigmoid_random, sigmoid_adv = \
                    sigmoid_data[1], sigmoid_data[2], sigmoid_data[3]
                softmax_test,softmax_random, softmax_adv = \
                    softmax_data[1], softmax_data[2], softmax_data[3]

                best_accu_test_idx = np.argmax(accu_test)
                best_accu_random_idx = np.argmax(accu_random)
                best_accu_adv_idx = np.argmax(accu_adv)

                sigmoid_rate = np.array(sigmoid_random) / np.array(sigmoid_test)
                best_sigmoid_rate_idx = np.argmin(sigmoid_rate)

                softmax_rate = np.array(softmax_random) / np.array(softmax_test)
                best_softmax_rate_idx = np.argmin(softmax_rate)

                data_list.append(x[best_accu_test_idx])
                data_list.append(accu_test[best_accu_test_idx])
                data_list.append(accu_random[best_accu_test_idx])
                data_list.append(accu_adv[best_accu_test_idx])
                data_list.append(sigmoid_rate[best_accu_test_idx])
                data_list.append(softmax_rate[best_accu_test_idx])

                data_list.append(x[best_accu_random_idx])
                data_list.append(accu_test[best_accu_random_idx])
                data_list.append(accu_random[best_accu_random_idx])
                data_list.append(accu_adv[best_accu_random_idx])
                data_list.append(sigmoid_rate[best_accu_random_idx])
                data_list.append(softmax_rate[best_accu_random_idx])

                data_list.append(x[best_accu_adv_idx])
                data_list.append(accu_test[best_accu_adv_idx])
                data_list.append(accu_random[best_accu_adv_idx])
                data_list.append(accu_adv[best_accu_adv_idx])
                data_list.append(sigmoid_rate[best_accu_adv_idx])
                data_list.append(softmax_rate[best_accu_adv_idx])

                data_list.append(x[best_sigmoid_rate_idx])
                data_list.append(accu_test[best_sigmoid_rate_idx])
                data_list.append(accu_random[best_sigmoid_rate_idx])
                data_list.append(accu_adv[best_sigmoid_rate_idx])
                data_list.append(sigmoid_rate[best_sigmoid_rate_idx])
                data_list.append(softmax_rate[best_sigmoid_rate_idx])

                data_list.append(x[best_softmax_rate_idx])
                data_list.append(accu_test[best_softmax_rate_idx])
                data_list.append(accu_random[best_softmax_rate_idx])
                data_list.append(accu_adv[best_softmax_rate_idx])
                data_list.append(sigmoid_rate[best_softmax_rate_idx])
                data_list.append(softmax_rate[best_softmax_rate_idx])

                summary_list.append(data_list)

    with open(fileout, 'w') as fout:
        writer = csv.writer(fout, lineterminator='\n')
        writer.writerows(summary_list)

def plot_csv(csv_folder = "csv", plot_folder="plots"):
    time_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    dict_data = {}
    for path in sorted(glob.glob( os.path.join(csv_folder, '*') )):
        if os.path.isdir(path):
            accu_csv = os.path.join(path, 'accuracy.csv')
            sigmoid_csv = os.path.join(path, 'sigmoid.csv')
            softmax_csv = os.path.join(path, 'softmax.csv')

            if os.path.exists(accu_csv) and os.path.exists(sigmoid_csv)\
                and os.path.exists(softmax_csv):
                model_name = path.split("/")[-1]
                print(model_name)

                model_basename = model_name.replace("_backprop","")\
                    .replace("_biprop","").replace("_halfbiprop","")\
                    .replace("_nobias","")

                data_list = [model_name]
                accu_data = get_data_csv_file(accu_csv)
                sigmoid_data = get_data_csv_file(sigmoid_csv)
                softmax_data = get_data_csv_file(softmax_csv)

                x, accu_test,accu_random, accu_adv, counter = \
                    accu_data[0], accu_data[1], accu_data[2], accu_data[3], accu_data[4]
                sigmoid_test,sigmoid_random, sigmoid_adv = \
                    sigmoid_data[1], sigmoid_data[2], sigmoid_data[3]
                softmax_test,softmax_random, softmax_adv = \
                    softmax_data[1], softmax_data[2], softmax_data[3]

                if not model_basename in dict_data.keys():
                    dict_data[model_basename] = {model_name: [x,accu_test,accu_random,accu_adv,sigmoid_test,sigmoid_random,sigmoid_adv,softmax_test,softmax_random,softmax_adv]}
                else:
                    (dict_data[model_basename])[model_name] = [x,accu_test,accu_random,accu_adv,sigmoid_test,sigmoid_random,sigmoid_adv,softmax_test,softmax_random,softmax_adv]

    for key, value in sorted(dict_data.items()):
        print(key)
        f, axarr = plt.subplots(2,3,sharey='row',figsize=(20,10))

        axarr[0,0].set_title('Accuracy for test data')
        axarr[0,0].set_xlabel('Iteration')
        axarr[0,0].set_ylabel('Accuracy')
        
        axarr[0,1].set_title('Accuracy for test data with noise')
        axarr[0,1].set_xlabel('Iteration')
        
        axarr[0,2].set_title('Accuracy for adversarial test data')
        axarr[0,2].set_xlabel('Iteration')
        
        axarr[1,0].set_title('Sigmoid maximum output rate')
        axarr[1,0].set_xlabel('Iteration')
        axarr[1,0].set_ylabel('Rate')
        
        axarr[1,1].set_title('Softmax maximum output rate')
        axarr[1,1].set_xlabel('Iteration')

        for k, v in value.items():
            label="None"
            if "_backprop" in k and not "nobias" in k:
                label = "Backpropagation"
            elif "_biprop" in k and not "nobias" in k:
                label = "Bidirectional learning"
            elif "_halfbiprop" in k and not "nobias" in k:
                label = "Half bidirectional learning"
            elif "_backprop" in k and "nobias" in k:
                label = "Backpropagation no bias"
            elif "_biprop" in k and "nobias" in k:
                label = "Bidirectional learning no bias"
            elif "_halfbiprop" in k and "nobias" in k:
                label = "Half bidirectional learning no bias"

            sigmoid_rate = np.array(v[5]) / np.array(v[4])
            softmax_rate = np.array(v[8]) / np.array(v[7])

            axarr[0,0].plot(v[0], v[1], label=label)
            axarr[0,0].grid(True)
            axarr[0,0].set_ylim([0., 1.1])
            axarr[0,1].plot(v[0], v[2], label=label)
            axarr[0,1].grid(True)
            axarr[0,1].set_ylim([0., 1.1])
            axarr[0,2].plot(v[0], v[3], label=label)
            axarr[0,2].grid(True)
            axarr[0,2].set_ylim([0., 1.1])
            axarr[1,0].plot(v[0], sigmoid_rate, label=label)
            axarr[1,0].grid(True)
            axarr[1,0].set_ylim([0., 1.1])
            axarr[1,1].plot(v[0], softmax_rate, label=label)
            axarr[1,1].grid(True)
            axarr[0,1].set_ylim([0., 1.1])
            axarr[1,2].plot(0.5, 0.5, label=label)
            axarr[1,2].set_ylim([0., 1.1])
            axarr[1,2].axis('off')
            axarr[1,2].legend(loc='upper left', prop={'size': 14})

        plt.savefig(os.path.join(plot_folder, key+".png"), bbox_inches='tight')
        plt.close(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot CSV files in folder')
    parser.add_argument('csv_folder', nargs='?', default="")
    args = parser.parse_args()

    if args is None:
        exit()
    elif not args.csv_folder:
        print("process_csv_best()")
        process_csv_best()
        print("plot_csv()")
        plot_csv()
        exit()

    accu_data = get_data_csv_file(os.path.join(args.csv_folder, 'accuracy.csv'))
    sigmoid_data = get_data_csv_file(os.path.join(args.csv_folder, 'sigmoid.csv'))
    softmax_data = get_data_csv_file(os.path.join(args.csv_folder, 'softmax.csv'))

    print_best(accu_data, sigmoid_data, softmax_data, os.path.join(args.csv_folder, 'summary.txt'))

    plot_accuracy(accu_data[0], accu_data[1], accu_data[2], accu_data[3])
    plot_sigmoid(sigmoid_data[0], sigmoid_data[1], sigmoid_data[2], sigmoid_data[3])
    plot_softmax(softmax_data[0], softmax_data[1], softmax_data[2], softmax_data[3])
