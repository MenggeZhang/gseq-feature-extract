from __future__ import division
import sys
import os
import os.path
from os.path import basename
import argparse
import csv
import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def extend_method(method_abbreviation):
    if method_abbreviation == 'DL':
        return 'Deep Learning'
    elif method_abbreviation == 'Logistic':
        return 'Logistic Regression'
    elif method_abbreviation == 'RF':
        return 'Random Forest'
    else:
        return method_abbreviation


def main():
    numCmdArgs = len(sys.argv)
    if(numCmdArgs!=2):
        print('Correct Usage:')
        print('python3 BarPlot_Performance.py Method')
        sys.exit(0)

    Method = sys.argv[1]
    mc_list = ['0','1']
    kmer_list = ['3','6','9']
    feat_list = ['1','2','3','4']

    kmer_3_filedirect = '/Users/menggezhang/Documents/colorectal-cancer-project/CRC_ClusterDown/Average_AUC_scores/k3/'
    kmer_6_filedirect = '/Users/menggezhang/Documents/colorectal-cancer-project/CRC_ClusterDown/Average_AUC_scores/k6/'
    kmer_9_filedirect = '/Users/menggezhang/Documents/colorectal-cancer-project/CRC_ClusterDown/Average_AUC_scores/k9/'

    AUC_FD = [kmer_3_filedirect, kmer_6_filedirect, kmer_9_filedirect]

    for mc in mc_list:
        AUC_scores = []
        for k in range(0,3):
            kmer = kmer_list[k]
            auc_directory = AUC_FD[k]
            AUC_Fname = auc_directory+Method+'_k'+kmer+'_m'+mc+'_AUC.csv'
            file_reader = csv.reader(open(AUC_Fname),delimiter=',')
            AUC_scores_perK = [[0 for i in range(4)] for j in range(12)]
            tr_te_com=0
            for row in file_reader:
                AUC_scores_perK[tr_te_com][0] = float([str(x) for x in row][0])
                AUC_scores_perK[tr_te_com][1] = float([str(x) for x in row][1])
                AUC_scores_perK[tr_te_com][2] = float([str(x) for x in row][2])
                AUC_scores_perK[tr_te_com][3] = float([str(x) for x in row][3])
                tr_te_com = tr_te_com+1
            
            # AUC_scores dimension: 3*12*4  kmer*12*feat
            AUC_scores.append(AUC_scores_perK)

        Train_Test_Com = [[0,1,2],[3,4,5,6,7,8],[9,10,11]]
        BarPlotNames = ['Self_Split','One_TrainSet','Two_TrainSets']
        
        figsizes = [(10,4),(20,4),(10,4)]
        for tr_te in range(3):
            train_test = Train_Test_Com[tr_te]
            barplotname = Method+'_'+mc+'_'+BarPlotNames[tr_te]+'_AUC.png'
            
            fig = plt.figure(figsize=figsizes[tr_te])
            ax = fig.add_subplot(111)

            N = len(feat_list)*len(train_test) + len(train_test)

            Feat_mat = [[0 for j in range(N)] for i in range(len(kmer_list)*len(train_test))]

            for feat in range(0,4):
                for i in range(len(kmer_list)*len(train_test)):
                    Feat_mat[i][feat+((i//len(kmer_list))*(1+len(feat_list)))]=AUC_scores[i%len(kmer_list)][train_test[i//len(train_test)]][feat]


            ind = np.arange(N)
            width = 0.3

            colors = ['#CC0066','#009999','#FFCC33']
            hatches = ['\\','.','x']

            rects = []
            for i in range(len(kmer_list)*len(train_test)):
                rect = ax.bar(ind+(i%len(kmer_list))*width, Feat_mat[i], width, color=colors[i%len(kmer_list)], hatch=hatches[i%len(kmer_list)],error_kw=dict(elinewidth=2,ecolor='#696969'))
                rects.append(rect)

            ax.set_xlim(-width,len(ind)+width)
            ax.set_ylim(0,1.0)
            ax.set_ylabel('AUC scores')
            ax.set_title(extend_method(Method)+' AUC-Scores  mc-order: '+mc+'  '+BarPlotNames[tr_te])

            xTickMarks = ['F-1','F-2','F-3','F-4',' ','F-1','F-2','F-3','F-4',' ','F-1','F-2','F-3','F-4']

            if tr_te==1:
                xTickMarks = ['F-1','F-2','F-3','F-4',' ','F-1','F-2','F-3','F-4',' ','F-1','F-2','F-3','F-4',' ','F-1','F-2','F-3','F-4',' ','F-1','F-2','F-3','F-4',' ','F-1','F-2','F-3','F-4']

            ax.set_yticks(np.arange(0, 1, step=0.05))
            ax.set_xticks(ind+2*width)
            xtickNames = ax.set_xticklabels(xTickMarks)
            plt.setp(xtickNames, rotation=45, fontsize=6)

            if tr_te==0:
                plt.text(0, 0.95, r'Train:NC / Test:NC',fontsize=10,weight='bold')
                plt.text(5, 0.95, r'Train:PO / Test:PO',fontsize=10,weight='bold')
                plt.text(10, 0.95, r'Train:MSB / Test:MSB',fontsize=10,weight='bold')
            elif tr_te==1:
                plt.text(0, 0.95, r'Train:NC / Test:PO',fontsize=10,weight='bold')
                plt.text(5, 0.95, r'Train:NC / Test:MSB',fontsize=10,weight='bold')
                plt.text(10, 0.95, r'Train:PO / Test:NC',fontsize=10,weight='bold')
                plt.text(15, 0.95, r'Train:PO / Test:MSB',fontsize=10,weight='bold')
                plt.text(20, 0.95, r'Train:MSB / Test:NC',fontsize=10,weight='bold')
                plt.text(25, 0.95, r'Train:MSB / Test:PO',fontsize=10,weight='bold')
            else:
                plt.text(0, 0.95, r'Train:NC&PO / Test:MSB',fontsize=10,weight='bold')
                plt.text(5, 0.95, r'Train:NC&MSB / Test:PO',fontsize=10,weight='bold')
                plt.text(10, 0.95, r'Train:PO&MSB / Test:NC',fontsize=10,weight='bold')
                

            ax.legend( (rects[0][0], rects[1][0], rects[2][0]), ('k = 3', 'k = 6', 'k = 9') ,prop={'size':6})
            plt.savefig(barplotname)


if __name__ == "__main__":
    main()