import sys
import csv
import os
import numpy as np
from os.path import basename
from statistics import mean, stdev
import random


from scipy.stats import pearsonr
from scipy.stats import spearmanr

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

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

def choose_best_alpha(alphas,scores):
    max_score = scores[0]
    position=0
    for i in range(0,len(scores)):
        if scores[i]>max_score:
            max_score = scores[i]
            position=i
    return alphas[position]


def main():

    numCmdArgs = len(sys.argv)
    if (numCmdArgs != 4):
        print('Correct Usage:')
        print('python3 CRC_LASSO.py mc kmer Epi')
        sys.exit(0)

    mc = sys.argv[1]
    kmer = sys.argv[2]
    Epi = sys.argv[3]
    feat_vec = [1,2,3,4]
    epi = 6
    # Epi is chosen from 2-Age,3-Gender,4-BMI,5-Country,6-Diagnosis
    if Epi == 'Age':
        epi=2
    elif Epi == 'Gender':
        epi=3
    elif Epi == 'BMI':
        epi=4
    elif Epi == 'Country':
        epi=5 
    else:
        epi=6  

    FeatVecPwdPrefix = '/home/rcf-40/menggezh/panasas/Colorectal_Cancer_New/JellyFish_Shell_Commands/Feat_k'+kmer+'_m'+mc+'/'
    Complete_Data_Info_Fname = '/home/rcf-40/menggezh/panasas/Colorectal_Cancer_New/JellyFish_Shell_Commands/Data_Info/Formed_Data_Info.csv'

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

    Nature_Data_Valid = select_ERR(Nature_Data, epi)
    PlosOne_Data_Valid = select_ERR(PlosOne_Data, epi)
    MolSysBio_Data_Valid = select_ERR(MolSysBio_Data, epi)
    
    random.shuffle(Nature_Data_Valid)
    random.shuffle(PlosOne_Data_Valid)
    random.shuffle(MolSysBio_Data_Valid)
    # There are 4 features
    X_Nature = [[] for i in range(4)]
    y_Nature = [[] for i in range(4)]

    X_PlosOne = [[] for i in range(4)]
    y_PlosOne = [[] for i in range(4)]

    X_MolSysBio = [[] for i in range(4)]
    y_MolSysBio = [[] for i in range(4)]

    Total_X=[]
    Total_Y=[]
    Total_Datasets_Names = ['Nature','PlosOne','MolSysBio']
    Total_Valid_Datasets = []

    for feat in feat_vec:
        for i in range(len(Nature_Data_Valid)):
            FileName = FeatVecPwdPrefix+'Feat_'+str(feat)+'_kmer='+kmer+'_mcorder='+mc+'_'+Nature_Data_Valid[i][0]+'.csv'
            FileReader = csv.reader(open(FileName),delimiter=',')
            for row in FileReader:
                X_Nature[feat-1].append([float(x) for x in row])
                y_Nature[feat-1].append(map_y_value(Nature_Data_Valid[i][epi],epi))
        
        for i in range(len(PlosOne_Data_Valid)):
            FileName = FeatVecPwdPrefix+'Feat_'+str(feat)+'_kmer='+kmer+'_mcorder='+mc+'_'+PlosOne_Data_Valid[i][0]+'.csv'
            FileReader = csv.reader(open(FileName),delimiter=',')
            for row in FileReader:
                X_PlosOne[feat-1].append([float(x) for x in row])
                y_PlosOne[feat-1].append(map_y_value(PlosOne_Data_Valid[i][epi],epi))

        for i in range(len(MolSysBio_Data_Valid)):
            FileName = FeatVecPwdPrefix+'Feat_'+str(feat)+'_kmer='+kmer+'_mcorder='+mc+'_'+MolSysBio_Data_Valid[i][0]+'.csv'
            FileReader = csv.reader(open(FileName),delimiter=',')
            for row in FileReader:
                X_MolSysBio[feat-1].append([float(x) for x in row])
                y_MolSysBio[feat-1].append(map_y_value(MolSysBio_Data_Valid[i][epi],epi))

    Total_X.append(X_Nature)
    Total_Y.append(y_Nature)
    Total_Valid_Datasets.append(Nature_Data_Valid)
    Total_X.append(X_PlosOne)
    Total_Y.append(y_PlosOne)
    Total_Valid_Datasets.append(PlosOne_Data_Valid)
    Total_X.append(X_MolSysBio)
    Total_Y.append(y_MolSysBio)
    Total_Valid_Datasets.append(MolSysBio_Data_Valid)

    # print(Total_X[0][0][:5])
    # print(len(Total_X[0][0][:5]))

    # print(Total_Y[0][0][:5])
    # print(len(Total_Y[0][0][:5]))

    for ft in range(0,4):
        Output_List_File = open('LASSO_F'+str(ft+1)+'_k'+kmer+'_m'+mc+'.csv', 'w')
        OutputMatrix = []
        OutputMatrix.append(['Training_Data','Testing_Data','AUC_Score'])
        # Train and test on each data itself: 4/5 as training, 1/5 as testing
        for i in range(0,3):
            num_training = (4*len(Total_X[i][ft]))//5
            reg = LassoCV(cv=5, random_state=None,max_iter =1000,normalize=True).fit(Total_X[i][ft][:num_training], Total_Y[i][ft][:num_training])
            predicted_Y_Lasso = reg.predict(Total_X[i][ft][num_training:])
            print(predicted_Y_Lasso)
            fpr_lasso, tpr_lasso, _ = roc_curve(Total_Y[i][ft][num_training:], predicted_Y_Lasso)
            auc_lasso = auc(fpr_lasso, tpr_lasso)
            OutputMatrix.append([Total_Datasets_Names[i],Total_Datasets_Names[i],str(auc_lasso)])
        
        # Use one dataset for training, test on the other two datasets
        Combination_1 = [[0,1],[0,2],[1,0],[1,2],[2,0],[2,1]]
        for vec in Combination_1:
            reg = LassoCV(cv=5, random_state=None,max_iter =1000,normalize=True).fit(Total_X[vec[0]][ft],Total_Y[vec[0]][ft])
            predicted_Y_Lasso = reg.predict(Total_X[vec[1]][ft])
            fpr_lasso, tpr_lasso, _ = roc_curve(Total_Y[vec[1]][ft], predicted_Y_Lasso)
            auc_lasso = auc(fpr_lasso, tpr_lasso)
            OutputMatrix.append([Total_Datasets_Names[vec[0]],Total_Datasets_Names[vec[1]],str(auc_lasso)])

        # Use two dataset for training, test on the other dataset
        Combination_2 = [[0,1,2],[0,2,1],[1,2,0]]
        for vec in Combination_2:
            reg = LassoCV(cv=5, random_state=None,max_iter =1000,normalize=True).fit(Total_X[vec[0]][ft]+Total_X[vec[1]][ft],Total_Y[vec[0]][ft]+Total_Y[vec[1]][ft])
            predicted_Y_Lasso = reg.predict(Total_X[vec[2]][ft])
            fpr_lasso, tpr_lasso, _ = roc_curve(Total_Y[vec[2]][ft], predicted_Y_Lasso)
            auc_lasso = auc(fpr_lasso, tpr_lasso)
            OutputMatrix.append([Total_Datasets_Names[vec[0]]+'_'+Total_Datasets_Names[vec[1]],Total_Datasets_Names[vec[2]],str(auc_lasso)])

        write_list = csv.writer(Output_List_File, delimiter = ',')
        write_list.writerows(OutputMatrix) 

if __name__ == "__main__":
    main()