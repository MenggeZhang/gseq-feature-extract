import sys
import csv
import os
import numpy as np
from os.path import basename
from statistics import mean, stdev
import random

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import plotly.plotly as py
import plotly.figure_factory as ff
import plotly.graph_objs as go


from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn import mixture
from sklearn import linear_model
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn import ensemble
from sklearn.metrics import roc_curve, auc

def select_Pos(Data_Info_Mat):
    # Epi is chosen from 2-Age,3-Gender,4-BMI,5-Country,6-Diagnosis
    Valid_Pos_With_Info = []
    position = 0
    patient_label = Data_Info_Mat[0][1]
    while position in range(0,len(Data_Info_Mat)-1):
        valid_Pos_positions_per_patient = []
        while position in range(0,len(Data_Info_Mat)-1) and Data_Info_Mat[position][1]==patient_label:
            if Data_Info_Mat[position][6]!='Large adenoma' and Data_Info_Mat[position][2]!='NA' and Data_Info_Mat[position][3]!='NA' and Data_Info_Mat[position][4]!='NA': 
                valid_Pos_positions_per_patient.append(position)
            position = position+1

        if len(valid_Pos_positions_per_patient)>0:
            Valid_Pos_With_Info.append(random.choice(valid_Pos_positions_per_patient))
            # Valid_Pos_With_Info.append(Data_Info_Mat[random.choice(valid_Pos_positions_per_patient)])
        valid_Pos_positions_per_patient = []
        patient_label = Data_Info_Mat[position][1]

    return Valid_Pos_With_Info

def rescale_mat(Mat):
    horizontal_lenth = len(Mat)
    if horizontal_lenth==0:
        return Mat
    else:
        vertical_lenth = len(Mat[0])
        Mat_ret = [[0 for i in range(vertical_lenth)] for j in range(horizontal_lenth)]
        for i in range(0,vertical_lenth):
            x_vec = []
            for j in range(0,horizontal_lenth):
                x_vec.append(Mat[j][i])
                
            x_vec_new = []
            for j in range(0,horizontal_lenth):
                if max(x_vec)==min(x_vec):
                    x_vec_new.append(x_vec[j])
                else:
                    x_vec_new.append((x_vec[j]-min(x_vec))/(max(x_vec)-min(x_vec)))
            
            for j in range(0,horizontal_lenth):
                Mat_ret[j][i] = x_vec_new[j]
        return Mat_ret


def map_epi_value(y,epi):
    if epi==3:
        # F is 0, M is 1
        if y=='F':
            return 0
        else:
            return 1
    elif epi==5:
        if y=='France':
            return 0
        elif y=='Germany':
            return 1
        elif y=='Denmark':
            return 2
        elif y=='Austria':
            return 3
        else:
            return 4

    elif epi==6:
        # Normal and Small adenoma is 0, Cancer is 1
        if y=='Cancer':
            return 1
        else:
            return 0
    else:
        return float(y)

def map_X_value(Data_Info_Mat,pos_vec):
    Mat_ret = []
    # epi_vec = [2,3,4,5,6]
    epi_vec = [2,4,5]
    for i in range(len(pos_vec)):
        vec = []
        for epi in epi_vec:
            vec.append(map_epi_value(Data_Info_Mat[pos_vec[i]][epi],epi))
        Mat_ret.append(vec)
    return Mat_ret

def map_y_value(Data_Info_Mat,pos_vec):
    y_ret = []
    # epi_vec = [2,3,4,5,6]
    # epi_vec = [6]
    for i in range(len(pos_vec)):
        y_ret.append(map_epi_value(Data_Info_Mat[pos_vec[i]][6],6))
        
    return y_ret


def main():
    numCmdArgs = len(sys.argv)
    if (numCmdArgs != 1):
        print('Correct Usage:')
        print('python3 Pheno_Kmeans_Clust.py')
        sys.exit(0)

    Complete_Data_Info_Fname = '/Users/menggezhang/Documents/colorectal-cancer-project/CRC_ClusterDown/Data_Info/Formed_Data_Info.csv'

    Complete_Data_Info_Reader = csv.reader(open(Complete_Data_Info_Fname),delimiter=',')
    Complete_Data_Info = []
    next(Complete_Data_Info_Reader)

    for row in Complete_Data_Info_Reader:
        Complete_Data_Info.append([str(x) for x in row])

    Nature_Data = []
    PlosOne_Data = []
    MolSysBio_Data = []

    for i in range(0,1212):
        MolSysBio_Data.append(Complete_Data_Info[i])
    for i in range(1212,1368):
        Nature_Data.append(Complete_Data_Info[i])
    for i in range(1368,1813):
        PlosOne_Data.append(Complete_Data_Info[i])

    Nature_Data_Valid = select_Pos(Nature_Data)
    PlosOne_Data_Valid = select_Pos(PlosOne_Data)
    MolSysBio_Data_Valid = select_Pos(MolSysBio_Data)
    Complete_Data_Valid = select_Pos(Complete_Data_Info)

    # print(len(Complete_Data_Valid))

    random.shuffle(Nature_Data_Valid)
    random.shuffle(PlosOne_Data_Valid)
    random.shuffle(MolSysBio_Data_Valid)
    random.shuffle(Complete_Data_Valid)

    # Epi is chosen from 2-Age,3-Gender,4-BMI,5-Country,6-Diagnosis
    
    Complete_X = map_X_value(Complete_Data_Info,Complete_Data_Valid)
    Nature_X = map_X_value(Nature_Data,Nature_Data_Valid)
    PlosOne_X = map_X_value(PlosOne_Data,PlosOne_Data_Valid)
    MolSysBio_X = map_X_value(MolSysBio_Data,MolSysBio_Data_Valid)

    Complete_y = map_y_value(Complete_Data_Info,Complete_Data_Valid)
    Nature_y = map_y_value(Nature_Data,Nature_Data_Valid)
    PlosOne_y = map_y_value(PlosOne_Data,PlosOne_Data_Valid)
    MolSysBio_y = map_y_value(MolSysBio_Data,MolSysBio_Data_Valid)

    Complete_X_pos = []
    Complete_X_neg = []
    for i in range(0,len(Complete_y)):
        if Complete_y[i]==1:
            Complete_X_pos.append(Complete_X[i])
        else:
            Complete_X_neg.append(Complete_X[i])
    Nature_X_pos = []
    Nature_X_neg = []
    for i in range(0,len(Nature_y)):
        if Nature_y[i]==1:
            Nature_X_pos.append(Nature_X[i])
        else:
            Nature_X_neg.append(Nature_X[i])
    PlosOne_X_pos = []
    PlosOne_X_neg = []
    for i in range(0,len(PlosOne_y)):
        if PlosOne_y[i]==1:
            PlosOne_X_pos.append(PlosOne_X[i])
        else:
            PlosOne_X_neg.append(PlosOne_X[i])

    MolSysBio_X_pos = []
    MolSysBio_X_neg = []
    for i in range(0,len(MolSysBio_y)):
        if MolSysBio_y[i]==1:
            MolSysBio_X_pos.append(MolSysBio_X[i])
        else:
            MolSysBio_X_neg.append(MolSysBio_X[i])




    # print(Complete_X)
    res_Complete_X = rescale_mat(Complete_X)
    res_Nature_X = rescale_mat(Nature_X)
    res_PlosOne_X = rescale_mat(PlosOne_X)
    res_MolSysBio_X = rescale_mat(MolSysBio_X)



    factor_vec = ['Rescaled Age', 'Rescaled BMI', 'Country']
    titles = ['Complete_Data','NatureCom_Data','PlosOne_Data','MolSysBio_Data']

    estimators = [('Complete_Data', res_Complete_X, Complete_y, KMeans(n_clusters=2)),('NatureCom_Data', res_Nature_X, Nature_y,KMeans(n_clusters=2)),('PlosOne_Data', res_PlosOne_X, PlosOne_y,KMeans(n_clusters=2)),('MolSysBio_Data', res_MolSysBio_X, MolSysBio_y, KMeans(n_clusters=2))]

    fignum=1
    for name,X,y,est in estimators:
        fig = plt.figure(fignum, figsize=(10, 9))
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=40, azim=134)
        est.fit(X)
        X_array=np.asarray(X)
        labels = est.labels_
        # print(X)
        ax.scatter(X_array[:, 0], X_array[:, 1], X_array[:, 2],c=labels.astype(np.float), edgecolor='k')

        

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel(factor_vec[0])
        ax.set_ylabel(factor_vec[1])
        ax.set_zlabel(factor_vec[2])
        ax.set_title(titles[fignum - 1])
        ax.dist = 12
        
        plt.savefig('Pheno_Kmeans_'+titles[fignum - 1]+'.jpg')
        fignum = fignum + 1

    
    

    # Draw the age histogram
    colors_age = ['rgb(255, 0, 0)', 'rgb(13, 13, 13)']
    Complete_Age_hist_data = [np.asarray(Complete_X_pos)[:, 0],np.asarray(Complete_X_neg)[:, 0]]
    Complete_Age_group_labels = ['Cancer_CompleteData', 'Normal_CompleteData']
    Complete_Age_fig = ff.create_distplot(Complete_Age_hist_data, Complete_Age_group_labels, bin_size=1.3, colors=colors_age)
    Complete_Age_fig['layout'].update(title='Age Distribution of Complete Datasets')
    py.iplot(Complete_Age_fig, filename='Complete_Distplot_of_Age')

    Nature_Age_hist_data = [np.asarray(Nature_X_pos)[:, 0],np.asarray(Nature_X_neg)[:, 0]]
    Nature_Age_group_labels = ['Cancer_NatureComData', 'Normal_NatureComData']
    Nature_Age_fig = ff.create_distplot(Nature_Age_hist_data, Nature_Age_group_labels, bin_size=1.3, colors=colors_age)
    Nature_Age_fig['layout'].update(title='Age Distribution of NatureCom Dataset')
    py.iplot(Nature_Age_fig, filename='Nature_Distplot_of_Age')

    PlosOne_Age_hist_data = [np.asarray(PlosOne_X_pos)[:, 0],np.asarray(PlosOne_X_neg)[:, 0]]
    PlosOne_Age_group_labels = ['Cancer_PlosOneData', 'Normal_PlosOneData']
    PlosOne_Age_fig = ff.create_distplot(PlosOne_Age_hist_data, PlosOne_Age_group_labels, bin_size=1.3, colors=colors_age)
    PlosOne_Age_fig['layout'].update(title='Age Distribution of PlosOne Dataset')
    py.iplot(PlosOne_Age_fig, filename='PlosOne_Distplot_of_Age')

    MolSysBio_Age_hist_data = [np.asarray(MolSysBio_X_pos)[:, 0],np.asarray(MolSysBio_X_neg)[:, 0]]
    MolSysBio_Age_group_labels = ['Cancer_MolSysBioData', 'Normal_MolSysBioData']
    MolSysBio_Age_fig = ff.create_distplot(MolSysBio_Age_hist_data, MolSysBio_Age_group_labels, bin_size=1.3, colors=colors_age)
    MolSysBio_Age_fig['layout'].update(title='Age Distribution of MolSysBio Dataset')
    py.iplot(MolSysBio_Age_fig, filename='MolSysBio_Distplot_of_Age')

    # Draw the BMI histogram
    colors_bmi = ['rgb(255, 102, 0)', 'rgb(0, 0, 255)']

    Complete_BMI_hist_data = [np.asarray(Complete_X_pos)[:, 1],np.asarray(Complete_X_neg)[:, 1]]
    Complete_BMI_group_labels = ['Cancer_CompleteData', 'Normal_CompleteData']
    Complete_BMI_fig = ff.create_distplot(Complete_BMI_hist_data, Complete_BMI_group_labels, bin_size=0.7, colors=colors_bmi)
    Complete_BMI_fig['layout'].update(title='BMI Distribution of Complete Datasets')
    py.iplot(Complete_BMI_fig, filename='Complete_Distplot_of_BMI')

    Nature_BMI_hist_data = [np.asarray(Nature_X_pos)[:, 1],np.asarray(Nature_X_neg)[:, 1]]
    Nature_BMI_group_labels = ['Cancer_NatureComData', 'Normal_NatureComData']
    Nature_BMI_fig = ff.create_distplot(Nature_BMI_hist_data, Nature_BMI_group_labels, bin_size=0.7, colors=colors_bmi)
    Nature_BMI_fig['layout'].update(title='BMI Distribution of NatureCom Dataset')
    py.iplot(Nature_BMI_fig, filename='Nature_Distplot_of_BMI')

    PlosOne_BMI_hist_data = [np.asarray(PlosOne_X_pos)[:, 1],np.asarray(PlosOne_X_neg)[:, 1]]
    PlosOne_BMI_group_labels = ['Cancer_PlosOneData', 'Normal_PlosOneData']
    PlosOne_BMI_fig = ff.create_distplot(PlosOne_BMI_hist_data, PlosOne_BMI_group_labels, bin_size=0.7, colors=colors_bmi)
    PlosOne_BMI_fig['layout'].update(title='BMI Distribution of PlosOne Dataset')
    py.iplot(PlosOne_BMI_fig, filename='PlosOne_Distplot_of_BMI')

    MolSysBio_BMI_hist_data = [np.asarray(MolSysBio_X_pos)[:, 1],np.asarray(MolSysBio_X_neg)[:, 1]]
    MolSysBio_BMI_group_labels = ['Cancer_MolSysBioData', 'Normal_MolSysBioData']
    MolSysBio_BMI_fig = ff.create_distplot(MolSysBio_BMI_hist_data, MolSysBio_BMI_group_labels, bin_size=0.7, colors=colors_bmi)
    MolSysBio_BMI_fig['layout'].update(title='BMI Distribution of MolSysBio Dataset')
    py.iplot(MolSysBio_BMI_fig, filename='MolSysBio_Distplot_of_BMI')
    

if __name__ == "__main__":
    main()