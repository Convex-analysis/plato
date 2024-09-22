import pandas as pd
from matplotlib import pyplot as plt
import os
from scipy.ndimage import gaussian_filter1d

def plot_acc_comparison(filename):
    linestyles = ['-', '--', '-.', ':', 'solid', 'dashed']
    marks = ['.', '.', '^', '<', '>']
    #根据CSV文件绘制准确率对比图
    # Read the CSV file as dataframe with four columns
    df = pd.read_csv(filename,sep='\t')
    #plot the accuracy comparison
    plt.figure()
    plt.plot(df['round'], df['Scheme1'], label='Scheme1', marker=marks[0], linestyle=linestyles[0])
    plt.plot(df['round'], df['Scheme2'], label='Scheme2', marker=marks[1], linestyle=linestyles[1])
    plt.plot(df['round'], df['AVFL'], label='AVFL', marker=marks[2], linestyle=linestyles[2])
    
    plt.xlabel('round',fontdict={'size': 18})
    plt.ylabel('accuracy',fontdict={'size': 18})
    plt.tick_params(labelsize=13)
    plt.legend(fontsize=15)
    plt.savefig(filename + ".png")

def impact_on_noniid(filename):
    linestyles = ['-.', ':', 'solid', 'dashed']
    marks = ['.', '^', '<', '>']
    data = pd.read_csv(filename)
    aixs_x = data['round']
    aixs_y = []
    for column in data.columns:
        if column != 'round':
            aixs_y.append(data[column])
    plt.figure()
    for i in range(len(aixs_y)):
        column = "noniid degree="+ data.columns[i+1]
        plt.plot(aixs_x, aixs_y[i], label=column, marker=marks[i], linestyle=linestyles[i])
    plt.xlabel('round')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig(filename + ".png")

def impact_of_noniid_on_shceme(filename):
    linestyles = ['-', '--', '-.', ':', 'solid', 'dashed']
    marks = ['.', '.', '^', '<', '>','+']
    data = pd.read_csv(filename)
    aixs_x = data['round']
    aixs_y = []
    for column in data.columns:
        if column != 'round':
            aixs_y.append(data[column])
    plt.figure()
    for i in range(len(aixs_y)):
        column = data.columns[i+1]
        #aixs_y[i] = gaussian_filter1d(aixs_y[i], sigma = 2)
        plt.plot(aixs_x, aixs_y[i], label=column, marker=marks[i], markevery=2,linestyle=linestyles[i])
    plt.xlabel('round',fontdict={'size': 18})
    plt.ylabel('accuracy',fontdict={'size': 18})
    plt.tick_params(labelsize=13)
    plt.legend(fontsize=15)
    plt.savefig(filename + ".png")

def impact_on_mobility(filename):
    linestyles = ['-.', ':', 'solid', 'dashed']
    marks = ['.', '^', '<', '>']
    data = pd.read_csv(filename)
    aixs_x = data['round']
    aixs_y = []
    for column in data.columns:
        if column != 'round':
            aixs_y.append(data[column])
    plt.figure()
    for i in range(len(aixs_y)):
        aixs_y[i] = gaussian_filter1d(aixs_y[i], sigma = 1)
        plt.plot(aixs_x, aixs_y[i], label=data.columns[i+1], marker=marks[i], markevery=5, linestyle=linestyles[i])
    plt.xlabel('round',fontdict={'size': 18})
    plt.ylabel('accuracy',fontdict={'size': 18})
    plt.tick_params(labelsize=13)
    plt.legend(fontsize=15)
    filename = filename.split(".")[0]
    plt.savefig(filename + ".png")

if __name__ == "__main__":
    """
    plot_acc_comparison("FashionCNN.csv")
    plot_acc_comparison("EMNISTlenet.csv")
    impact_on_noniid("EMNIST_noniid.csv")
    impact_on_noniid("Fashion_noniid.csv")
    impact_on_mobility("mobilityFashion.csv")
    impact_on_mobility("mobilityEMNIST.csv")
    
    plot_acc_comparison("D:/EXP/plato/examples/magni_fed/log/FashionCNN.csv")
    plot_acc_comparison("D:/EXP/plato/examples/magni_fed/log/EMNISTlenet.csv")
    impact_on_mobility("D:/EXP/plato/examples/magni_fed/log/mobilityFashion.csv")
    impact_on_mobility("D:/EXP/plato/examples/magni_fed/log/mobilityEMNIST.csv") 
    impact_of_noniid_on_shceme("D:/EXP/plato/examples/magni_fed/log/EMNIST_noniid_comparision.csv")
    impact_of_noniid_on_shceme("D:/EXP/plato/examples/magni_fed/log/Fashion_noniid_comparision.csv")

    impact_of_noniid_on_shceme("D:/EXP/plato/examples/magni_fed/log/FashionMINSTnoniid100R.csv")
    impact_of_noniid_on_shceme("D:/EXP/plato/examples/magni_fed/log/EMINSTnoniid100R.csv")
    impact_on_mobility("examples\magni_fed\log\FashionMINSTmobi100R.csv")
    """
    defalut_path = "./"
    for file in os.listdir(defalut_path):
        filename = file.split(".")[0] + ".png"
        if file.endswith(".csv") and os.path.exists(defalut_path + "/" + filename) == False:
            print(file)
            impact_on_mobility(defalut_path + "/" + file)
           