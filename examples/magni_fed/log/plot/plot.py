import pandas as pd
from matplotlib import pyplot as plt


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
    plt.plot(df['round'], df['Scheme3'], label='Scheme3', marker=marks[2], linestyle=linestyles[2])

    plt.xlabel('round',fontdict={'size': 18})
    plt.ylabel('accuracy',fontdict={'size': 18})
    plt.legend(fontsize=15)
    plt.tick_params(labelsize=13)
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
    plt.xlabel('round',fontdict={'size': 18})
    plt.ylabel('accuracy',fontdict={'size': 18})
    plt.legend(fontsize=15)
    plt.tick_params(labelsize=13)
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
        plt.plot(aixs_x, aixs_y[i], label=column, marker=marks[i], linestyle=linestyles[i] )
    plt.xlabel('round',fontdict={'size': 18})
    plt.ylabel('accuracy',fontdict={'size': 18})
    plt.tick_params(labelsize=13)
    plt.legend(fontsize=15)
    plt.savefig(filename + ".png")

if __name__ == "__main__":
    """
    impact_on_noniid("EMNIST_noniid.csv")
    impact_on_noniid("Fashion_noniid.csv")
    """
    #对比算法
    plot_acc_comparison("FashionCNN.csv")
    plot_acc_comparison("EMNISTlenet.csv")
    #对比不同程度的noniid
    impact_of_noniid_on_shceme("EMNIST_noniid_comparision.csv")
    impact_of_noniid_on_shceme("Fashion_noniid_comparision.csv")

