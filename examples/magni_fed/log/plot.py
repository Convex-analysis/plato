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
    linestyles = ['-.', ':', 'solid', 'dashed', '-', '--']
    marks = ['.', '^', '<', '>', '+', 'x']
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
    filename = filename.split("/")[-1]
    filename = filename.split(".")[0]
    save_path = "./examples/magni_fed/log/" + filename + ".png"
    plt.savefig(save_path)
    
def scalablity_exp(save_path):
    mamba_result = {
        "50": 0.72,
        "100": 1.60,
        "150": 2.73,
        "200": 4.11,
        "250": 5.74,
        "300": 7.61,
        "500": 17.64,
        "700": 31.67,
        "1000": 60.25
    }
    transformer_result = {
        "50": 0.72,
        "100": 2.10,
        "150": 4.99,
        "200": 10.12,
        "250": 18.27,
        "300": 30.17,
        "500": 130.46,
        "700": 350.86,
        "1000": 1011.97
    }
    LSTM_result = {
        "50": 0.73,
        "100": 1.85,
        "150": 3.67,
        "200": 6.35,
        "250": 10.03,
        "300": 14.82,
        "500": 47.13,
        "700": 104.62,
        "1000": 247.92
    }
    Baseline_result = {
        "50": 0.66,
        "100": 1.35,
        "150": 2.17,
        "200": 3.10,
        "250": 4.17,
        "300": 5.36,
        "500": 11.37,
        "700": 19.40,
        "1000": 35.18
    }
    
    plt.figure(figsize=(6.4, 4.8))  # Increased figure size for better readability
    
    # Convert string keys to integers for x-axis values
    x_values = [int(k) for k in mamba_result.keys()]
    
    # Create the plots with logarithmic scales
    plt.semilogx(x_values, list(mamba_result.values()), label='Mamba', 
                 marker='o', linestyle='-', linewidth=2, markersize=8)
    plt.semilogx(x_values, list(transformer_result.values()), label='Transformer', 
                 marker='s', linestyle='--', linewidth=2, markersize=8)
    plt.semilogx(x_values, list(LSTM_result.values()), label='LSTM', 
                 marker='^', linestyle='-.', linewidth=2, markersize=8)
    plt.semilogx(x_values, list(Baseline_result.values()), label='Baseline', 
                 marker='d', linestyle=':', linewidth=2, markersize=8)
    
    # Customize x-axis ticks to show all data points
    plt.xticks(x_values, x_values, rotation=45)
    
    plt.xlabel('Number of Vehicles', fontdict={'size': 18})
    plt.ylabel('Time (ms)', fontdict={'size': 18})
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tick_params(labelsize=12)
    
    # Adjust legend position and style
    plt.legend(fontsize=15, loc='upper left', bbox_to_anchor=(0.02, 0.98))
    
    # Add minor gridlines for better readability
    plt.grid(True, which='minor', linestyle=':', alpha=0.3)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save with higher DPI for better quality
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
def adaptability_exp(save_path):
    results = {
        "Mamba": {
            "average_reward": 22567.84,
            "average_performance": 0.0504,
            "decision_time": 50.37,
            "phase_performances": {
                "phase1 (High Mobility)": 0.0180*18,
                "phase2 (Poor Connectivity)": 0.0378*18,
                "phase3 (Resource Constraint)": 0.0504*18
            },
            "adaptability_score": 0.9163
        },
        "Transformer": {
            "average_reward": 21584.56,
            "average_performance": 0.0274,
            "decision_time": 7.97,
            "phase_performances": {
                "phase1 (High Mobility)": 0.0130*18,
                "phase2 (Poor Connectivity)": 0.0206*18,
                "phase3 (Resource Constraint)": 0.0274*18
            },
            "adaptability_score": 0.9139
        },
        "LSTM": {
            "average_reward": 21656.90,
            "average_performance": 0.0385,
            "decision_time": 5.05,
            "phase_performances": {
                "phase1 (High Mobility)": 0.0173*18,
                "phase2 (Poor Connectivity)": 0.0297*18,
                "phase3 (Resource Constraint)": 0.0385*18
            },
            "adaptability_score": 0.9000
        },
        "Random": {
            "average_reward": 22209.89,
            "average_performance": 0.0328,
            "decision_time": 0.02,
            "phase_performances": {
                "phase1 (High Mobility)": 0.0206*18,
                "phase2 (Poor Connectivity)": 0.0283*18,
                "phase3 (Resource Constraint)": 0.0328*18
            },
            "adaptability_score": 0.7909
        }
    }
    
    # Now you can easily access the data, for example:
    # results["Mamba"]["average_reward"]
    # results["Transformer"]["phase_performances"]["phase1"]
    
    # TODO: Add visualization code here using the reorganized data structure
    plt.figure()
    bar_width = 0.2
    x = list(range(len(results["Mamba"]["phase_performances"])))
    
    for i, (key, value) in enumerate(results.items()):
        plt.bar([pos + bar_width*i for pos in x], list(value["phase_performances"].values()), 
                width=bar_width, label=key)
    
    plt.xlabel('Phases', fontdict={'size': 18})
    plt.ylabel('Performance', fontdict={'size': 18})
    plt.xticks([pos + bar_width*(len(results)-1)/2 for pos in x], 
               list(results["Mamba"]["phase_performances"].keys()),
               rotation=15)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path + "_phases.png", dpi=300, bbox_inches='tight')
    
    # Adaptability score plot
    plt.figure()
    schedulers = list(results.keys())
    scores = [value["adaptability_score"] for value in results.values()]
    plt.bar(schedulers, scores, color=['blue', 'orange', 'green', 'red'])
    
    plt.xlabel('Schedulers', fontdict={'size': 18})
    plt.ylabel('Adaptability Score', fontdict={'size': 18})
    plt.xticks(rotation=0)
    plt.ylim(0.77, 0.93)  # Adjust y-axis range for better visualization
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path + "_adaptability.png", dpi=300, bbox_inches='tight')
    
    plt.show()
    

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
    defalut_path = "./examples/magni_fed/log"
    for file in os.listdir(defalut_path):
        filename = file.split(".")[0] + ".png"
        if file.endswith(".csv") and os.path.exists(defalut_path + "/" + filename) == False:
            print(file)
            impact_on_mobility(defalut_path + "/" + file)
    save_path = "./examples/magni_fed/log/scalablity.png"
    scalablity_exp(save_path)
    adaptability_exp("./examples/magni_fed/log/adaptability")
