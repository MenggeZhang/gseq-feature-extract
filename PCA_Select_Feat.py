import sys
import csv
import os
import numpy as np
from os.path import basename
from statistics import mean, stdev
import random

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def select_ERR(Data_Info_Mat, epi):
    # Epi is chosen from 2-Age,3-Gender,4-BMI,5-Country,6-Diagnosis
    Valid_ERR_With_Info = []
    if epi==6:
        position = 0
        patient_label = Data_Info_Mat[0][1]
        while position in range(0,len(Data_Info_Mat)-1):
            valid_ERR_positions_per_patient = []
            while position in range(0,len(Data_Info_Mat)-1) and Data_Info_Mat[position][1]==patient_label:
                if Data_Info_Mat[position][epi]!='Large adenoma': 
                    valid_ERR_positions_per_patient.append(position)
                position = position+1
            if len(valid_ERR_positions_per_patient)>0:
                Valid_ERR_With_Info.append(Data_Info_Mat[random.choice(valid_ERR_positions_per_patient)])
            valid_ERR_positions_per_patient = []
            patient_label = Data_Info_Mat[position][1]
    else:
        position = 0
        patient_label = Data_Info_Mat[0][1]
        while position in range(0,len(Data_Info_Mat)-1):
            valid_ERR_positions_per_patient = []
            while position in range(0,len(Data_Info_Mat)-1) and Data_Info_Mat[position][1]==patient_label:
                if Data_Info_Mat[position][epi]!='NA': 
                    valid_ERR_positions_per_patient.append(position)
                position = position+1
            
            if len(valid_ERR_positions_per_patient)>0:
                Valid_ERR_With_Info.append(Data_Info_Mat[random.choice(valid_ERR_positions_per_patient)])
            valid_ERR_positions_per_patient = []
            patient_label = Data_Info_Mat[position][1]

    return Valid_ERR_With_Info

def map_y_value(y,epi):
    if epi==3:
        # F is 0, M is 1
        if y=='F':
            return 0
        else:
            return 1
    elif epi==6:
        # Normal and Small adenoma is 0, Cancer is 1
        if y=='Cancer':
            return 1
        else:
            return 0
    else:
        return float(y)


def main():
    numCmdArgs = len(sys.argv)
    if (numCmdArgs != 3):
        print('Correct Usage:')
        print('python3 CRC_RF.py mc kmer')
        sys.exit(0)

    mc = sys.argv[1]
    kmer = sys.argv[2]
    feat_vec = [1,2,3,4]
    
    FeatVecPwdPrefix = '/auto/cmb-12/fs3/menggezh/Colorectal_Feature_k'+kmer+'/Feat_k'+kmer+'_m'+mc+'/'
    Complete_Data_Info_Fname = '/auto/cmb-12/fs3/menggezh/Colorectal_Feature_k9/Data_Info/Formed_Data_Info.csv'

    ERR_List = []
    Complete_Data_Info_Reader = csv.reader(open(Complete_Data_Info_Fname),delimiter=',')
    next(Complete_Data_Info_Reader)

    for row in Complete_Data_Info_Reader:
        ERR_List.append([str(x) for x in row][0])

    Nature_ERR_List = []
    PlosOne_ERR_List = []
    MolSysBio_ERR_List = []

    


    for feat in feat_vec:
        Feat_Mat_All = []
        Feat_Mat_Nature = []
        Feat_Mat_PlosOne = []
        Feat_Mat_MolSysBio = []
        Feat_Mat_Nature_PlosOne = []
        Feat_Mat_Nature_MolSysBio = []
        Feat_Mat_PlosOne_MolSysBio = []

        for i in range(0, len(ERR_List)):
            FileName = FeatVecPwdPrefix+'Feat_'+str(feat)+'_kmer='+kmer+'_mcorder='+mc+'_'+Nature_Data_Valid[i][0]+'.csv'
            FileReader = csv.reader(open(FileName),delimiter=',')
            for row in FileReader:
                Feat_Mat_All.append([float(x) for x in row])
        
        for i in range(0,1212):
            Feat_Mat_MolSysBio.append(Feat_Mat_All[i])

            Feat_Mat_Nature_MolSysBio.append(Feat_Mat_All[i])
            Feat_Mat_PlosOne_MolSysBio.append(Feat_Mat_All[i])

        for i in range(1212,1368):
            Feat_Mat_Nature.append(Feat_Mat_All[i])

            Feat_Mat_Nature_MolSysBio.append(Feat_Mat_All[i])
            Feat_Mat_Nature_PlosOne.append(Feat_Mat_All[i])

        for i in range(1368,1813):
            Feat_Mat_PlosOne.append(Feat_Mat_All[i])

            Feat_Mat_PlosOne_MolSysBio.append(Feat_Mat_All[i])
            Feat_Mat_Nature_PlosOne.append(Feat_Mat_All[i])
            
        Feat_Mat_All = np.asarray(Feat_Mat_All)
        Feat_Mat_Nature = np.asarray(Feat_Mat_Nature)
        Feat_Mat_PlosOne = np.asarray(Feat_Mat_PlosOne)
        Feat_Mat_MolSysBio = np.asarray(Feat_Mat_MolSysBio)
        Feat_Mat_Nature_PlosOne = np.asarray(Feat_Mat_Nature_PlosOne)
        Feat_Mat_Nature_MolSysBio = np.asarray(Feat_Mat_Nature_MolSysBio)
        Feat_Mat_PlosOne_MolSysBio = np.asarray(Feat_Mat_PlosOne_MolSysBio)

        pca = PCA(n_components=min(4**(int(kmer)),100))

        pca.fit(Feat_Mat_All)
        Feat_Mat_All_trans = pca.fit_transform(Feat_Mat_All, y=None)

        Output_Feat_Mat_All_trans = open('PCA_All_'+str(feat)+'_k'+kmer+'_m'+mc+'.csv', 'w')
        write_Feat_Mat_All_trans = csv.writer(Output_Feat_Mat_All_trans, delimiter = ',')
        write_Feat_Mat_All_trans.writerows(Feat_Mat_All_trans)

        pca.fit(Feat_Mat_Nature)
        Feat_Mat_Nature_trans = pca.fit_transform(Feat_Mat_Nature, y=None)

        Output_Feat_Mat_Nature_trans = open('PCA_Nature_'+str(feat)+'_k'+kmer+'_m'+mc+'.csv', 'w')
        write_Feat_Mat_Nature_trans = csv.writer(Output_Feat_Mat_Nature_trans, delimiter = ',')
        write_Feat_Mat_Nature_trans.writerows(Feat_Mat_Nature_trans)

        pca.fit(Feat_Mat_PlosOne)
        Feat_Mat_PlosOne_trans = pca.fit_transform(Feat_Mat_PlosOne, y=None)

        Output_Feat_Mat_PlosOne_trans = open('PCA_PlosOne_'+str(feat)+'_k'+kmer+'_m'+mc+'.csv', 'w')
        write_Feat_Mat_PlosOne_trans = csv.writer(Output_Feat_Mat_PlosOne_trans, delimiter = ',')
        write_Feat_Mat_PlosOne_trans.writerows(Feat_Mat_PlosOne_trans)

        pca.fit(Feat_Mat_MolSysBio)
        Feat_Mat_MolSysBio_trans = pca.fit_transform(Feat_Mat_MolSysBio, y=None)

        Output_Feat_Mat_MolSysBio_trans = open('PCA_MolSysBio_'+str(feat)+'_k'+kmer+'_m'+mc+'.csv', 'w')
        write_Feat_Mat_MolSysBio_trans = csv.writer(Output_Feat_Mat_MolSysBio_trans, delimiter = ',')
        write_Feat_Mat_MolSysBio_trans.writerows(Feat_Mat_MolSysBio_trans)

        pca.fit(Feat_Mat_Nature_PlosOne)
        Feat_Mat_Nature_PlosOne_trans = pca.fit_transform(Feat_Mat_Nature_PlosOne, y=None)

        Output_Feat_Mat_Nature_PlosOne_trans = open('PCA_Nature_PlosOne_'+str(feat)+'_k'+kmer+'_m'+mc+'.csv', 'w')
        write_Feat_Mat_Nature_PlosOne_trans = csv.writer(Output_Feat_Mat_Nature_PlosOne_trans, delimiter = ',')
        write_Feat_Mat_Nature_PlosOne_trans.writerows(Feat_Mat_Nature_PlosOne_trans)

        pca.fit(Feat_Mat_Nature_MolSysBio)
        Feat_Mat_Nature_MolSysBio_trans = pca.fit_transform(Feat_Mat_Nature_MolSysBio, y=None)

        Output_Feat_Mat_Nature_MolSysBio_trans = open('PCA_Nature_MolSysBio_'+str(feat)+'_k'+kmer+'_m'+mc+'.csv', 'w')
        write_Feat_Mat_Nature_MolSysBio_trans = csv.writer(Output_Feat_Mat_Nature_MolSysBio_trans, delimiter = ',')
        write_Feat_Mat_Nature_MolSysBio_trans.writerows(Feat_Mat_Nature_MolSysBio_trans)


        pca.fit(Feat_Mat_PlosOne_MolSysBio)
        Feat_Mat_PlosOne_MolSysBio_trans = pca.fit_transform(Feat_Mat_PlosOne_MolSysBio, y=None)

        Output_Feat_Mat_PlosOne_MolSysBio_trans = open('PCA_PlosOne_MolSysBio_'+str(feat)+'_k'+kmer+'_m'+mc+'.csv', 'w')
        write_Feat_Mat_PlosOne_MolSysBio_trans = csv.writer(Output_Feat_Mat_PlosOne_MolSysBio_trans, delimiter = ',')
        write_Feat_Mat_PlosOne_MolSysBio_trans.writerows(Feat_Mat_PlosOne_MolSysBio_trans)



        

    
    





if __name__ == "__main__":
    main()