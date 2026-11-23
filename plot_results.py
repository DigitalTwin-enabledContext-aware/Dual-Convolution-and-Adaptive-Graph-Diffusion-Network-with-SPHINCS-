import os
from itertools import cycle
import numpy as np
from matplotlib import pyplot as plt, patches, pylab
from matplotlib.patches import Polygon
from prettytable import PrettyTable
from sklearn import metrics

from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score, roc_curve

No_of_dataset = 1



#
# GTO-DCAGDN-NCRL
# BCO-DCAGDN-NCRL
# SFOA-DCAGDN-NCRL
# OOA-DCAGDN-NCRL
# HOOA-DCAGDN-NCRL
#
#
# LSTM
# RNN
# Bi-GRU
# GDN
# HOOA-DCAGDN-NCRL
#
# ECC
# BAQE
# PQC-PUF
# LQRA
# SPHINCS+



def Table1():
    eval = np.load('Eval_all_plot.npy', allow_pickle=True)
    Algorithm = ['BatchSize', 'GTO-DCAGDN-NCRL', 'BCO-DCAGDN-NCRL', 'SFOA-DCAGDN-NCRL', 'OOA-DCAGDN-NCRL', 'HOOA-DCAGDN-NCRL']
    Classifier = ['BatchSize', 'LSTM', 'RNN', 'Bi-GRU', ' GDN', 'HOOA-DCAGDN-NCRL']
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = np.array([0, 4,5, 9, 15]).astype(int)
    Table_Terms = [4,5,  9, 15]
    table_terms = [Terms[i] for i in Table_Terms]
    Batchsize = ['4', '8', '16', '32', '48']
    for i in range(3):
        for k in range(len(Table_Terms)):
            value = eval[i, :, :, 4:]

            Table = PrettyTable()
            Table.add_column(Algorithm[0], Batchsize)
            for j in range(len(Algorithm) - 1):
                Table.add_column(Algorithm[j + 1], value[:, j, Graph_Terms[k]])
            print('-------------------------------------classification Dataset - ', i + 1, table_terms[k], '  Algorithm Comparison',
                  '---------------------------------------')
            print(Table)

            Table = PrettyTable()
            Table.add_column(Classifier[0], Batchsize)
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[:, len(Algorithm) + j - 1, Graph_Terms[k]])
            print('--------------------------------------- classification Dataset - ', i + 1, table_terms[k], '  Classifier Comparison',
                  '---------------------------------------')
            print(Table)


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])

def plotConvResults():

    Algorithm = ['TERMS','GTO-DCAGDN-NCRL', 'BCO-DCAGDN-NCRL', 'SFOA-DCAGDN-NCRL', 'OOA-DCAGDN-NCRL', 'HOOA-DCAGDN-NCRL']
    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    for i in range(3):
        Fitness = np.load('Fitness_' + str(i + 1) + '.npy', allow_pickle=True)
        Conv_Graph = np.zeros((len(Algorithm) - 1, len(Terms)))
        for j in range(len(Algorithm) - 1):
            Conv_Graph[j, :] = Statistical(Fitness[j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print(
            f'-------------------------------------------------- Statistical Analysis {i + 1} --------------------------------------------------')
        print(Table)

        length = np.arange(Fitness.shape[1])
        fig = plt.figure()
        fig.canvas.manager.set_window_title('Dataset-' + str(i + 1) + ' Convergence Curve')
        plt.plot(length, Fitness[0, :], color='#94d3ac', linewidth=3, marker='*', markerfacecolor='red',
                 markersize=12, label=Algorithm[1])
        plt.plot(length, Fitness[1, :], color='#7b5de1', linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12, label=Algorithm[2])
        plt.plot(length, Fitness[2, :], color='#ef5350', linewidth=3, marker='*', markerfacecolor='blue',
                 markersize=12, label=Algorithm[3])
        plt.plot(length, Fitness[3, :], color='#94d3ac', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12, label=Algorithm[4])
        plt.plot(length, Fitness[4, :], color='#ef5350', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12, label=Algorithm[5])
        plt.xlabel('No. of Iteration', fontname="Arial", fontsize=15, fontweight='bold', color='k')
        plt.ylabel('Cost Function', fontname="Arial", fontsize=15, fontweight='bold', color='k')
        plt.legend(loc=1, fontsize=15)
        plt.savefig("./Results/Conv_%s.png" % (i + 1))
        plt.show()




# def Plot_ROC_Curve():
#     cls = ['CNN', 'VGG-19', 'Resnet', ' ERV-SEGP', 'PUEHO-AERV-SEGP']
#     for a in range(3):  # For 5 Datasets
#         Actual = np.load('Target_' + str(a + 1) + '.npy', allow_pickle=True)
#         lenper = round(Actual.shape[0] * 0.75)
#         fig = plt.figure()
#         fig.canvas.manager.set_window_title('Dataset-' + str(a + 1) + ' ROC Curve')
#         colors = cycle(["blue", "darkorange", "limegreen", "deeppink", "black"])
#         for i, color in zip(range(len(cls)), colors):
#             Predicted = np.load('Y_Score_' + str(a + 1) + '.npy', allow_pickle=True)[i]
#             false_positive_rate, true_positive_rate, _ = roc_curve(Actual.ravel(), Predicted.ravel())
#             roc_auc = roc_auc_score(Actual.ravel(), Predicted.ravel())
#             roc_auc = roc_auc * 100
#
#             plt.plot(
#                 false_positive_rate,
#                 true_positive_rate,
#                 color=color,
#                 lw=2,
#                 label=f'{cls[i]} (AUC = {roc_auc:.2f} %)')
#
#         plt.plot([0, 1], [0, 1], "k--", lw=2)
#         plt.xlim([0.0, 1.0])
#         plt.ylim([0.0, 1.05])
#         plt.title('Accuracy')
#         plt.xlabel("False Positive Rate", fontsize=12)
#         plt.ylabel("True Positive Rate", fontsize=12)
#         plt.title("ROC Curve")
#         plt.legend(loc="lower right", fontsize=12)
#         path = "./Results/Dataset_%s_ROC.png" % (a + 1)
#         plt.savefig(path)
#         plt.show()

def Plot_ROC_Curve():
    cls = ['LSTM', 'RNN', 'Bi-GRU', ' GDN', 'HOOA-DCAGDN-NCRL']
    for a in range(3):  # For 5 Datasets
        Actual = np.load('Target_' + str(a + 1) + '.npy', allow_pickle=True)
        lenper = round(Actual.shape[0] * 0.75)
        fig = plt.figure()
        fig.canvas.manager.set_window_title('Dataset-' + str(a + 1) + ' ROC Curve')
        colors = cycle(["blue", "darkorange", "limegreen", "deeppink", "black"])
        for i, color in zip(range(len(cls)), colors):
            Predicted = np.load('Y_Score_' + str(a + 1) + '.npy', allow_pickle=True)[i]
            false_positive_rate, true_positive_rate, _ = roc_curve(Actual.ravel(), Predicted.ravel())
            roc_auc = roc_auc_score(Actual.ravel(), Predicted.ravel())
            roc_auc = roc_auc * 100

            plt.plot(
                false_positive_rate,
                true_positive_rate,
                color=color,
                lw=2,
                label=f'{cls[i]} (AUC = {roc_auc:.2f} %)')

        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Accuracy')
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("ROC Curve")
        plt.legend(loc="lower right", fontsize=12)
        path = "./Results/Dataset_%s_ROC.png" % (a + 1)
        plt.savefig(path)
        plt.show()





def Plot_Proposed_Results():
    eval = np.load('Evaluate_prop.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']

    Epoch = ['10', '20', '30',
             '40', '50', '60',
             '70', '80', '90',
             '100']

    Graph_Terms = [0,   4, 5]  # Metrics to visualize

    colors = ['#FFB74D', '#E57373', '#BA68C8', '#64B5F6', '#4DB6AC',
              '#81C784', '#FFD54F', '#F06292', '#9575CD', '#4FC3F7']

    for n in range(2):
        for j in range(len(Graph_Terms)):
            Graph = eval[n, :, 4, Graph_Terms[j]+4]  # shape (10, 10)
            X = np.arange(len(Epoch))
            val = Graph
            nos = len(val)
            bW = 0.50
            Max = np.max(val)
            Min=np.min(val)

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_axes([0.12, 0.25, 0.8, 0.65])
            fig.canvas.manager.set_window_title('3D Styled Bar Graph')

            for idx, value in enumerate(val):
                base_color = colors[idx % len(colors)]

                poly_front = Polygon([[idx - bW / 2, Max / 20], [idx - bW / 2, value],
                                      [idx, -Max / 20 + value], [idx, 0]],
                                     closed=True, facecolor=base_color, alpha=0.7, zorder=1)
                ax.add_patch(poly_front)

                poly_side = Polygon([[idx, -Max / 20 + value], [idx, 0], [idx + bW / 2, Max / 20],
                                     [idx + bW / 2, value]],
                                    closed=True, facecolor=base_color, alpha=0.9, zorder=1)
                ax.add_patch(poly_side)
                poly_top = Polygon([[idx - bW / 2, value], [idx, -Max / 20 + value],
                                    [idx + bW / 2, value], [idx, Max /20 + value]],
                                   closed=True, facecolor=base_color, alpha=0.7, zorder=1)
                ax.add_patch(poly_top)

            ax.set_xlim(-0.4, nos)
            ax.set_ylim(0, np.max(val) + np.max(val) * 0.05)
            ax.set_ylabel(Terms[Graph_Terms[j]], fontname="Arial",
                          fontsize=14, fontweight='bold', color='k')
            ax.set_xlabel('No of Epoch', fontname="Arial",
                          fontsize=14, fontweight='bold', color='k',labelpad= 45)
            ax.set_xticks([])
            ax.set_xticklabels('')
            ax.spines['bottom'].set_visible(True)
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            ax.set_axisbelow(True)
            for idx, ep in enumerate(Epoch):
                ax.text(idx, -(np.max(val) * 0.10), ep, ha='center', va='top',
                        fontsize=9, fontweight='bold', color='black', fontname='Arial')
                ax.scatter(idx, -(np.max(val) * 0.05), s=120, color=colors[idx % len(colors)], marker='s', clip_on=False)


            plt.tight_layout(rect=[0, 0.1, 1, 1])
            path = f"./Results/Dataset{n+1}_{Terms[Graph_Terms[j]]}_Mod.png"
            plt.savefig(path, dpi=350, bbox_inches='tight')
            plt.show()

# class_1 = ['Ripple', 'Fast Ripple','Ripple and Fast Ripple']


def Confusion():
    for n in range(3):
        Actual = np.load('Actual_' + str(n + 1) + '.npy', allow_pickle=True).astype(np.int32)
        Predict = np.load('Predict_' + str(n + 1) + '.npy', allow_pickle=True).astype(np.int32)
        class_1 = ['Yes', 'No']
        class_2 = ['Yes', 'No']
        class_3 = ['Malignant ', 'Benign']
        classes = [class_1, class_2,class_3]
        confusion_matrix = metrics.confusion_matrix(Actual, Predict)
        fig = plt.figure(figsize=(10, 7))
        fig.canvas.manager.set_window_title('Confusion Matrix')
        ax = fig.add_axes([0.12, 0.18, 0.7, 0.7])
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=classes[n])
        cm_display.plot(ax=ax, cmap='Blues')
        for labels in cm_display.text_.ravel():
            labels.set_fontsize(16)  # Set desired font size
        path = "./Results/Confusion_%s.png" % (n + 1)
        plt.title("Confusion Matrix")
        plt.ylabel('Actual', fontname="Arial", fontsize=15, fontweight='bold', color='k')
        plt.xlabel('Predicted', fontname="Arial", fontsize=15, fontweight='bold', color='k')
        plt.xticks(fontname="Arial", fontsize=13, fontweight='bold', color='k')
        plt.yticks(fontname="Arial", fontsize=13, fontweight='bold', color='k')
        plt.savefig(path)
        plt.show()

def Plot_Proposed():
    eval = np.load('Eval_all_plot.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = [0,1,2,3]
    Act_fun = ['Linear', 'ReLU', 'TanH', 'Softmax', 'Sigmoid']
    colors = ['#8c2f39', '#2a9d8f', 'm', '#6a0572', '#3c096c']
    for n in range(3):  # For each dataset
        for j in range(len(Graph_Terms)):  # For each metric
            Graph = eval[n, :, 4, Graph_Terms[j] + 4]
            val = Graph
            nos = len(val)
            bW = 0.30
            Max = np.max(val)
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_axes([0.12, 0.25, 0.8, 0.65])
            fig.canvas.manager.set_window_title('Activation Function vs Proposed')
            for idx, value in enumerate(val):
                base_color = colors[idx % len(colors)]

                # front face
                poly_front = Polygon([[idx - bW / 2, Max / 20], [idx - bW / 2, value],
                                      [idx, -Max / 80 + value], [idx, 0]],
                                     closed=True, facecolor=base_color, alpha=0.7, zorder=1)
                ax.add_patch(poly_front)

                # side face
                poly_side = Polygon([[idx, -Max / 80 + value], [idx, 0], [idx + bW / 2, Max / 20],
                                     [idx + bW / 2, value]],
                                    closed=True, facecolor=base_color, alpha=0.9, zorder=1)
                ax.add_patch(poly_side)

                # top face
                poly_top = Polygon([[idx - bW / 2, value], [idx, -Max / 80 + value],
                                    [idx + bW / 2, value], [idx, Max / 80 + value]],
                                   closed=True, facecolor=base_color, alpha=0.7, zorder=1)
                ax.add_patch(poly_top)
            ax.set_xlim(-0.4, nos)
            ax.set_ylim(0, np.max(val) + np.max(val) * 0.05)
            plt.yticks(fontsize=12, fontweight='bold', color='black', fontname='Arial')
            ax.set_ylabel(Terms[Graph_Terms[j]], fontname="Arial",
                          fontsize=14, fontweight='bold', color='k')
            ax.set_xlabel(
                'Activation Function',
                fontname="Arial",
                fontsize=14,
                fontweight='bold',
                color='k',
                labelpad= 45
            )

            # Clean x-axis
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.spines['bottom'].set_visible(True)
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            ax.set_axisbelow(True)

            # Epoch markers
            for idx, ep in enumerate(Act_fun):
                ax.scatter(idx, -(np.max(val) * 0.05), s=125, color=colors[idx % len(colors)], marker='s',
                           clip_on=False)
                ax.text(idx, -(np.max(val) * 0.10), ep, ha='center', va='top',
                        fontsize=12, fontweight='bold', color='black', fontname='Arial')

            plt.tight_layout(rect=[0, 0.1, 1, 1])
            path = "./Results/Dataset_%s_%s_Proposed_bar.png" % (n+1, Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()




def plot_block(Graph, labels, title_name, save_path, Terms, term_idx, Kfold):

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_axes([0.15, 0.19, 0.8, 0.7])

    X = np.arange(len(Kfold))
    num_items = len(labels)

    positions = np.linspace(0, 0.7, num_items)
    colors = plt.cm.tab10.colors[:num_items]

    for idx in range(num_items):
        for k in range(len(Kfold)):

            x_pos = X[k] + positions[idx]
            y_big = Graph[k, idx]
            y_small = y_big * 0.9

            ax.plot([x_pos, x_pos], [y_small, y_big],
                    linestyle='--', linewidth = 1,
                    color=colors[idx])

            ax.plot([x_pos, x_pos], [0, y_big],
                    linestyle='-', linewidth=2,
                    color=colors[idx], alpha=0.7)

            ax.scatter(x_pos, y_big,
                       color=colors[idx],
                       edgecolor='black',
                       s=70, marker='d')
            ax.text(x_pos, y_big / 2,
                    f'{y_big:.2f}',
                    ha='center', va='center',
                    fontsize=9,
                    fontweight='bold',
                    rotation=90)

    center_offset = np.mean(positions)

    plt.xticks(X + center_offset,
               ['100', '200', '300', '400', '500'],
               fontsize=13, fontweight='bold')

    plt.xlabel('Steps per Epoch', fontsize=14, fontweight='bold')
    plt.ylabel(Terms[term_idx], fontsize=14, fontweight='bold')
    legend_handles = [
        Line2D([0], [0],
               marker='d',
               linestyle='',
               markerfacecolor=colors[i],
               markeredgecolor='black',
               markersize=10,
               label=labels[i])
        for i in range(num_items)
    ]

    plt.legend(handles=legend_handles,
               loc='upper center',
               bbox_to_anchor=(0.5, 1.15),
               ncol=3,
               frameon=False,
               fontsize=14)

    ax.set_ylim(0, np.max(Graph) * 1.177)
    plt.title(title_name, fontsize=14, fontweight='bold')
    plt.savefig(save_path, dpi=300)
    plt.show(block=False)
    plt.pause(1)
    plt.close()


# Algorithm = ['BatchSize', 'GTO-DCAGDN-NCRL', 'BCO-DCAGDN-NCRL', 'SFOA-DCAGDN-NCRL', 'OOA-DCAGDN-NCRL', 'HOOA-DCAGDN-NCRL']
# Classifier = ['BatchSize', 'LSTM', 'RNN', 'Bi-GRU', ' GDN', 'HOOA-DCAGDN-NCRL']


def Plots_Results1():

    eval_data = np.load('Eval_all_plot.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = [0,1,2,3]
    Algorithm = ['GTO-DCAGDN-NCRL', 'BCO-DCAGDN-NCRL', 'SFOA-DCAGDN-NCRL', 'OOA-DCAGDN-NCRL', 'HOOA-DCAGDN-NCRL']
    Cls = ['LSTM', 'RNN', 'Bi-GRU', ' GDN', 'HOOA-DCAGDN-NCRL']
    Kfold = [1, 2, 3, 4, 5]
    for i in range(3):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval_data.shape[1:3])
            for k in range(eval_data.shape[1]):
                for m in range(eval_data.shape[2]):
                    Graph[k, m] = eval_data[i, k, m, Graph_Terms[j] + 4]

            path1 = f"./Results/classification_Dataset_{i + 1}_{Terms[Graph_Terms[j]]}_Algorithm.png"

            plot_block(Graph,
                       Algorithm, "",
                       path1,
                       Terms,
                       Graph_Terms[j],
                       Kfold)

            path2 = f"./Results/classification_Dataset_{i + 1}_{Terms[Graph_Terms[j]]}_Classifier.png"

            plot_block(Graph + 3,
                       Cls,
                       "",
                       path2,
                       Terms,
                       Graph_Terms[j],
                       Kfold)


def Table1_attention():
    eval = np.load('Evaluate_table_new.npy', allow_pickle=True)
    Algorithm = ['Loss', 'GTO-DCAGDN-NCRL', 'BCO-DCAGDN-NCRL', 'SFOA-DCAGDN-NCRL', 'OOA-DCAGDN-NCRL',
                 'HOOA-DCAGDN-NCRL']
    Classifier = ['Loss', 'LSTM', 'RNN', 'Bi-GRU', ' GDN', 'HOOA-DCAGDN-NCRL']
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = np.array([0, 4,5, 9, 15]).astype(int)
    Table_Terms = [0, 4,5,  9, 15]
    table_terms = [Terms[i] for i in Table_Terms]
    Batchsize = ['Attention guided loss', 'Hybrid Loss', 'Margin based loss', 'Triplet loss', 'Label smoothing loss']
    for i in range(3):
        for k in range(len(Table_Terms)):
            value = eval[i, :, :, 4:]
            Table = PrettyTable()
            Table.add_column(Classifier[0], Batchsize)
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[:, len(Algorithm) + j - 1, Graph_Terms[k]])
            print('--------------------------------------- Loss Dataset - ', i + 1, table_terms[k], '  Classifier Comparison',
                  '---------------------------------------')
            print(Table)

            Table = PrettyTable()
            Table.add_column(Classifier[0], Batchsize)
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[:, len(Algorithm) + j - 1, Graph_Terms[k]])
            print('--------------------------------------- Loss Dataset - ', i + 1, table_terms[k],
                  '  Classifier Comparison',
                  '---------------------------------------')
            print(Table)


# ECC
# BAQE
# PQC-PUF
# LQRA
# SPHINCS+

def Cls_Bar(file_name, ylabel, title):

    if not os.path.exists(file_name):
        return
    Graph = np.load(file_name, allow_pickle=True)[0]  # shape: (num_cases, num_algorithms)
    Algorithm = ['ECC', 'BAQE', 'PQC-PUF', 'LQRA',
                 'SPHINCS+']
    colors = ['#B2BEB5', '#fee440', '#ef5350', '#94d3ac', 'b']
    num_cases = Graph.shape[0]

    X = np.arange(num_cases)
    bar_width = 0.14
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
    for i in range(len(Algorithm)):
        ax.barh(X + i*bar_width, Graph[:, i], color=colors[i], edgecolor='w',
                height=bar_width, label=Algorithm[i])
    plt.yticks(X + bar_width*2, ['16', '32', '64', '128', '256'], fontsize=12,
               fontname="Arial", fontweight='bold', color='k')

    plt.xlabel(ylabel, fontname="Arial", fontsize=15, fontweight='bold', color='k')
    plt.ylabel('Key Length', fontsize=15, fontname="Arial", fontweight='bold', color='k')
    plt.xticks(fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    dot_markers = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color, markersize=10)
                   for color in colors]
    plt.legend(dot_markers, Algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.10),
               fontsize=13, frameon=False, ncol= 3)
    plt.grid(axis='y', linestyle='-', alpha=0.7)
    plt.grid(axis='x')
    os.makedirs("./Results", exist_ok=True)
    path = f"./Results/{title}.png"
    plt.savefig(path)
    plt.show()



def Key_Cls(file_name, ylabel, title):

    Eval = np.load(file_name, allow_pickle=True)[0]

    Classifier = ['LSTM', 'RNN', 'Bi-GRU', ' GDN', 'HOOA-DCAGDN-NCRL']
    colors = ['#ff1654', '#8ac926', '#9e2a2b', '#197278', 'k']
    Configurations = np.arange(Eval.shape[0])
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
    width = 0.15
    for i in range(len(Classifier)):
        ax.bar(Configurations + i * width, Eval[:, i], color=colors[i], width=width, label=Classifier[i])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    dot_markers = [plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color, markersize=10)
                   for color in colors]
    plt.legend(dot_markers, Classifier, loc='upper center', bbox_to_anchor=(0.5, 1.08),
               fontsize=9, frameon=False, ncol=3)
    plt.xticks(Configurations + 0.3, [str(i + 1) for i in range(Eval.shape[0])],
               fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
    plt.xlabel('Cases', fontname="Arial", fontsize=12, fontweight='bold', color='k')
    plt.ylabel(ylabel, fontname="Arial", fontsize=12, fontweight='bold', color='k')
    plt.ylim(0)
    path = f"./Results/{title}.png"
    os.makedirs("./Results", exist_ok=True)
    plt.savefig(path)
    plt.show()

def Results():
    Cls_Bar('Encryption_time.npy', 'Encryption time (S)', 'Encryption_time')
    Cls_Bar('Decryption_time.npy', 'Decryption time (S)', 'Decryptiontime')
    Cls_Bar('Total Time.npy', ' Total Computational time (S)', 'Time')
    Cls_Bar('Memory_Size.npy', 'Memory Size (KB)', 'Memory_size')




if __name__ == "__main__":
    # Plot_ROC_Curve()
    # Plots_Results1()
    # Plot_Proposed()
    # Table1_attention()
    # Confusion()
    # plotConvResults()
    # Table1()
    Results()
