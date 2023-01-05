import pandas as pd
import numpy as np
import math
from operator import itemgetter

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import seaborn as sns
sns.set()

from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

from sklearn import tree
from sklearn.tree import _tree

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('ignore')


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

"""
ASSIGNMENT 1: Data Preparation
"""

"""
Getting Setup
"""

INFILE = "/volumes/MyLibrary/MSDS 422/Unit 04/Assignment 4/HMEQ_Loss.csv"

TARGET_F = "TARGET_BAD_FLAG"
TARGET_A = "TARGET_LOSS_AMT"

df = pd.read_csv(INFILE)

#print(df.head().T)
dt = df.dtypes
#print(dt)
#print(df.describe().T)

"""
List Variables by Type
"""

objList = []
numList = []

for i in dt.index:
    if i in ([TARGET_F, TARGET_A]): continue
    if dt[i] in (["object"]): objList.append(i)
    if dt[i] in (["int64","float64"]): numList.append(i)

"""
Explore Variables and Relationships
"""

#Probabilities and possible values for objects
##for i in objList:
##    print("----------------")
##    print("*** Variable:", i, "***")
##    print("Possible values=", df[i].unique())
##    g = df.groupby(i)
##    print(g[i].count())
##    print("Most Common=",df[i].mode()[0])
##    print("Missing=", df[i].isna().sum())
##    x = g[TARGET_F].mean()
##    print("*Probability of Default:", x)
##    x = g[TARGET_A].mean()
##    print("*Loss Amount:", x)
##    print("----------------")
##
##
##with PdfPages(r'/volumes/MyLibrary/MSDS 422/Unit 03/Assignment 3/charts3.pdf') as export_pdf:
##    #Pie charts for object variables
##    for i in objList:
##        x = df[i].value_counts(dropna=False)
##        theLabels = x.axes[0].tolist()
##        theSlices = list(x)
##        plt.pie(theSlices,
##                labels=theLabels,
##                startangle=180,
##                autopct="%1.0f%%",
##                pctdistance = 0.9,
##                labeldistance = 1.1)
##        plt.title("Pie Chart: " + i)
##        export_pdf.savefig()
##        #plt.show()
##        plt.close()
##
##    #Histograms for numerical variables
##    for i in numList:
##        plt.hist(df[i])
##        plt.xlabel(i)
##        export_pdf.savefig()
##        #plt.show()
##        plt.close()
##
##    #Histogram for Target Amount Variable
##    plt.hist(df[TARGET_A])
##    plt.xlabel("Loss Amount")
##    export_pdf.savefig()
##    #plt.show()
##    plt.close()


"""
Impute Missing Values
"""

#OBJECT VARIABLES

#Fill in Missing with Category MISSING
for i in objList:
    if df[i].isna().sum() == 0: continue
    NAME = "IMP_" + i
    df[NAME] = df[i]
    df[NAME] = df[NAME].fillna("MISSING")
    df = df.drop(i, axis=1)

#Update object list
dt = df.dtypes
objList = []
for i in dt.index:
    if i in ([TARGET_F, TARGET_A]): continue
    if dt[i] in (["object"]): objList.append(i)

#Confirm new class MISSING
##for i in objList:
##    g = df.groupby(i)
##    print(g[i].count())

#NUMERIC VARIABLES
    
#Save binary variables for later outlier handling
binaryVars = []

for i in numList:
    if df[i].isna().sum() == 0: continue
    FLAG = "M_"+i #numeric representation of missing or not
    NAME = "IMP_"+i #new variable for imputed values
    binaryVars.append(FLAG)
    df[FLAG] = df[i].isna()+0 #Bool return into int; flag to mark missing or not
    df[NAME] = df[i]
    df.loc[df[NAME].isna(), NAME] = df[i].median() #impute using median
    df = df.drop(i, axis=1)

#Confirm Imputed Variables
#print(df.head().T)

"""
Covert Categorical Variables into Numeric Variables (One Hot Encoding)
"""

for i in objList:
    thePrefix = "z_" + i
    y = pd.get_dummies(df[i], prefix = thePrefix)
    df = pd.concat([df,y], axis=1)

#Confirm dummy variables
#print(df.head().T)

"""
Handle Non-Target Outliers
"""

#collect numeric variables rejecting binary variables
dt = df.dtypes
numList = []
for i in dt.index:
    if i in ([TARGET_F, TARGET_A]): continue
    if i in (binaryVars): continue
    if dt[i] in (["float64", "int64"]): numList.append(i)

#remove outliers outside 3 STD DEV
for i in numList:
    theMean = df[i].mean()
    theSD = df[i].std()
    theMax = df[i].max()
    cutoff = round(theMean + 3*theSD)
    if theMax > cutoff:
        FLAG = "O_" + i #numeric representation of outliers or not
        NAME = "TRUNC_" + i #new variable for truncated values
        binaryVars.append(FLAG)
        df[FLAG] = (df[i] > cutoff) + 0 #Bool return into int; flag to mark outliers or not
        df[NAME] = df[i]
        df.loc[df[NAME] > cutoff, NAME] = cutoff
        df = df.drop(i, axis=1)

#print quick descriptions after outlier removal
##dt = df.dtypes
##numList = []
##for i in dt.index:
##    if i in ([TARGET_F, TARGET_A]): continue
##    if i in (binaryVars): continue
##    if dt[i] in (["float64", "int64"]): numList.append(i)
##
##print("Descriptive Statistics:")
##for i in numList:
##    print(i)
##    print(df[i].describe())
##    plt.hist(df[i])
##    plt.xlabel(i)
##    plt.show()


"""
ASSIGNMENT 2: Tree Models
"""

"""
Split the Data
"""

#flag variable
X = df.copy()
X = X.drop([TARGET_F, TARGET_A, 'IMP_REASON', 'IMP_JOB'], axis = 1)

Y = df[[TARGET_F, TARGET_A]]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8, test_size = 0.2, random_state=2)

#amt variable
F = ~ Y_train[TARGET_A].isna() #gather all target amounts that arent null
W_train = X_train[F].copy()
Z_train = Y_train[F].copy()

F = ~ Y_test[TARGET_A].isna()
W_test = X_test[F].copy()
Z_test = Y_test[F].copy()

"""
Handle Target Outliers
"""

##print(Z_train.describe())
##print(Z_test.describe())

F = Z_train[TARGET_A] > 50000
Z_train.loc[F, TARGET_A] = 50000

F = Z_test[TARGET_A] > 50000
Z_test.loc[F, TARGET_A] = 50000


"""
Model Accuracy Metrics
"""

def getProbAccuracyScores(NAME, MODEL, X, Y):
    #get predictions
    pred = MODEL.predict(X)
    #get probabilities
    probs = MODEL.predict_proba(X)[:,1]
    #compare predictions vs actual
    acc_score = metrics.accuracy_score(Y, pred)
    #ROC curve metrics
    fpr, tpr, threshold = metrics.roc_curve(Y, probs)
    auc = metrics.auc(fpr, tpr)
    return [NAME, acc_score, fpr, tpr, auc, acc_score]

def saveRocCurve(TITLE, LIST, OUTFILE):
    with PdfPages(r'/volumes/MyLibrary/MSDS 422/Unit 04/Assignment 4/' + OUTFILE) as export_pdf:
        fig = plt.figure(figsize=(8,6))
        plt.title(TITLE)
        for results in LIST:
            NAME = results[0]
            fpr = results[2]
            tpr = results[3]
            auc = results[4]
            label = "AUC " + NAME + " %0.2f" % auc
            plt.plot(fpr,tpr,label=label)
        plt.legend(loc = 'lower right')
        plt.plot([0,1], [0,1], 'r--')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.xlabel("True Positive Rate")
        plt.ylabel("False Positive Rate")
        export_pdf.savefig()
        #plt.show()
        plt.close()

def printAccuracy(TITLE, LIST):
    print(TITLE)
    for results in LIST:
        NAME = results[0]
        ACC = results[1]
        print(NAME, "=", ACC)
    print("------------------")

def getAmtAccuracyScores(NAME, MODEL, X, Y):
    pred = MODEL.predict(X)
    MEAN = Y.mean()
    RMSE = math.sqrt(metrics.mean_squared_error(Y,pred))
    return [NAME, RMSE, MEAN]


"""
Decision Tree
"""

def getTreeVars(TREE, varNames):
    tree_ = TREE.tree_
    varName = [varNames[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]
    nameSet = set()
    for i in tree_.feature:
        if i != _tree.TREE_UNDEFINED:
            nameSet.add(i)
    nameList = list(nameSet)
    parameter_list = list()
    for i in nameList:
        parameter_list.append(varNames[i])
    return parameter_list

theModel = "TREE"

# CLASSIFICATION MODEL

CLM = tree.DecisionTreeClassifier(max_depth=6) #depth of 7 began to lower test accuracy
CLM = CLM.fit(X_train, Y_train[TARGET_F])

#get scores for train and test sets
CLM_train = getProbAccuracyScores(theModel + "_Train", CLM, X_train, Y_train[TARGET_F])
CLM_test = getProbAccuracyScores(theModel, CLM, X_test, Y_test[TARGET_F])

#save or print ROC curve
saveRocCurve(theModel, [CLM_train, CLM_test], 'charts3a.pdf')

#print classification accuracy
#printAccuracy("Classification Accuracy", [CLM_train, CLM_test])

#export decision tree via graph viz
feature_cols = list(X.columns.values)
tree.export_graphviz(CLM, out_file = 'tree_f.txt', filled=True, rounded=True, feature_names = feature_cols, impurity=False, class_names = ['Good', 'Default'])

#get predictive variables
vars_tree_flag = getTreeVars(CLM, feature_cols)

##print("Tree CLM Vars:")
##for i in vars_tree_flag:
##    print(i)


# AMOUNT MODEL

AMT = tree.DecisionTreeRegressor(max_depth = 3)
AMT = AMT.fit(W_train, Z_train[TARGET_A])

#get scores for train and test sets
AMT_train = getAmtAccuracyScores(theModel + "_Train", AMT, W_train, Z_train[TARGET_A])
AMT_test = getAmtAccuracyScores(theModel, AMT, W_test, Z_test[TARGET_A])

#print RMSE accuracy
#printAccuracy("RMSE Accuracy", [AMT_train, AMT_test])

#export decision tree via graph viz
feature_cols = list(X.columns.values)
tree.export_graphviz(AMT, out_file = 'tree_a.txt', filled=True, rounded=True, feature_names = feature_cols, impurity=False, class_names = ['Good', 'Default'])

#get predictive variables
vars_tree_amt = getTreeVars(AMT, feature_cols)

##print("Tree AMT Vars:")
##for i in vars_tree_amt:
##    print(i)

#Store results for comparison
TREE_CLM = CLM_test.copy()
TREE_AMT = AMT_test.copy()



"""
Random Forest
"""

def getEnsembleTreeVars(ENSTREE, varNames):
    importance = ENSTREE.feature_importances_
    index = np.argsort(importance)
    theList = []
    for i in index:
        imp_val = importance[i]
        if imp_val > np.average(ENSTREE.feature_importances_):
            v = int(imp_val / np.max(ENSTREE.feature_importances_)*100)
            theList.append((varNames[i], v))
    theList = sorted(theList, key=itemgetter(1), reverse=True)
    return theList

theModel = "RANDOM_FOREST"

# CLASSIFICATION MODEL

CLM = RandomForestClassifier(n_estimators = 100, random_state=1)
CLM = CLM.fit(X_train, Y_train[TARGET_F])

#get scores for train and test sets
CLM_train = getProbAccuracyScores(theModel + "_Train", CLM, X_train, Y_train[TARGET_F])
CLM_test = getProbAccuracyScores(theModel, CLM, X_test, Y_test[TARGET_F])

#save or print ROC curve
saveRocCurve(theModel, [CLM_train, CLM_test], 'charts3b.pdf')

#print classification accuracy
#printAccuracy("Classification Accuracy", [CLM_train, CLM_test])

#get predictive variables
feature_cols = list(X.columns.values)
vars_RF_flag = getEnsembleTreeVars(CLM, feature_cols)

##print("RF CLM Vars:")
##for i in vars_RF_flag:
##    print(i)


# AMOUNT MODEL

AMT = RandomForestRegressor(n_estimators=20, random_state=1)
AMT = AMT.fit(W_train, Z_train[TARGET_A])

#get scores for train and test sets
AMT_train = getAmtAccuracyScores(theModel + "_Train", AMT, W_train, Z_train[TARGET_A])
AMT_test = getAmtAccuracyScores(theModel, AMT, W_test, Z_test[TARGET_A])

#print RMSE accuracy
#printAccuracy("RMSE Accuracy", [AMT_train, AMT_test])

#get predictive variables
feature_cols = list(X.columns.values)
vars_RF_amt = getEnsembleTreeVars(AMT, feature_cols)

##print("RF AMT Vars:")
##for i in vars_RF_amt:
##    print(i)

#Store results for comparison
RF_CLM = CLM_test.copy()
RF_AMT = AMT_test.copy()



"""
Gradient Boosting
"""

theModel = "GRADIENT_BOOSTING"

# CLASSIFICATION MODEL

CLM = GradientBoostingClassifier(random_state=1)
CLM = CLM.fit(X_train, Y_train[TARGET_F])

#get scores for train and test sets
CLM_train = getProbAccuracyScores(theModel + "_Train", CLM, X_train, Y_train[TARGET_F])
CLM_test = getProbAccuracyScores(theModel, CLM, X_test, Y_test[TARGET_F])

#save or print ROC curve
saveRocCurve(theModel, [CLM_train, CLM_test], 'charts3c.pdf')

#print classification accuracy
#printAccuracy("Classification Accuracy", [CLM_train, CLM_test])

#get predictive variables
feature_cols = list(X.columns.values)
vars_GB_flag = getEnsembleTreeVars(CLM, feature_cols)

##print("GB CLM Vars:")
##for i in vars_GB_flag:
##    print(i)


# AMOUNT MODEL

AMT = GradientBoostingRegressor(random_state=1)
AMT = AMT.fit(W_train, Z_train[TARGET_A])

#get scores for train and test sets
AMT_train = getAmtAccuracyScores(theModel + "_Train", AMT, W_train, Z_train[TARGET_A])
AMT_test = getAmtAccuracyScores(theModel, AMT, W_test, Z_test[TARGET_A])

#print RMSE accuracy
#printAccuracy("RMSE Accuracy", [AMT_train, AMT_test])

#get predictive variables
feature_cols = list(X.columns.values)
vars_GB_amt = getEnsembleTreeVars(AMT, feature_cols)

##print("GB AMT Vars:")
##for i in vars_GB_amt:
##    print(i)

#Store results for comparison
GB_CLM = CLM_test.copy()
GB_AMT = AMT_test.copy()


"""
ASSIGNMENT 3: Logistic and Linear Regression
"""

def getCoefLogit(MODEL, TRAIN_DATA):
    varNames = list(TRAIN_DATA.columns.values)
    coef_dict = {}
    for coef, feat in zip(MODEL.coef_[0], varNames):
        coef_dict[feat] = coef
    print("DEFAULT")
    print("------------")
    print("Total Variables:", len(coef_dict))
    for i in coef_dict:
        print(i, "=", coef_dict[i])

        
def getCoefLinear(MODEL, TRAIN_DATA):
    varNames = list(TRAIN_DATA.columns.values)
    coef_dict = {}
    coef_dict["INTERCEPT"] = MODEL.intercept_
    for coef, feat in zip(MODEL.coef_, varNames):
        coef_dict[feat] = coef
    print("AMOUNT")
    print("------------")
    print("Total Variables:", len(coef_dict))
    for i in coef_dict:
        print(i, "=", coef_dict[i])


"""
Regression- All variables
"""

theModel = "REG_ALL"

# CLASSIFICATION MODEL

CLM = LogisticRegression(solver='newton-cg', max_iter=3000)
CLM = CLM.fit(X_train, Y_train[TARGET_F])

CLM_train = getProbAccuracyScores(theModel + "_Train", CLM, X_train, Y_train[TARGET_F])
CLM_test = getProbAccuracyScores(theModel, CLM, X_test, Y_test[TARGET_F])

#save or print ROC curve
saveRocCurve(theModel, [CLM_train, CLM_test], 'charts3d.pdf')

#print classification accuracy
#printAccuracy("Classification Accuracy", [CLM_train, CLM_test])


# AMOUNT MODEL

AMT = LinearRegression()
AMT= AMT.fit(W_train, Z_train[TARGET_A])

#get scores for train and test sets
AMT_train = getAmtAccuracyScores(theModel + "_Train", AMT, W_train, Z_train[TARGET_A])
AMT_test = getAmtAccuracyScores(theModel, AMT, W_test, Z_test[TARGET_A])

#print RMSE accuracy
#printAccuracy("RMSE Accuracy", [AMT_train, AMT_test])

#get coeficients
##REG_ALL_CLM_COEF = getCoefLogit(CLM, X_train)
##REG_ALL_AMT_COEF = getCoefLinear(AMT, X_train)


#Store results for comparison
REG_ALL_CLM = CLM_test.copy()
REG_ALL_AMT = AMT_test.copy()



"""
Regression- Tree variables
"""

theModel = "REG_TREE"

# CLASSIFICATION MODEL

CLM = LogisticRegression(solver='newton-cg', max_iter=3000)
CLM = CLM.fit(X_train[vars_tree_flag], Y_train[TARGET_F])

CLM_train = getProbAccuracyScores(theModel + "_Train", CLM, X_train[vars_tree_flag], Y_train[TARGET_F])
CLM_test = getProbAccuracyScores(theModel, CLM, X_test[vars_tree_flag], Y_test[TARGET_F])

#save or print ROC curve
saveRocCurve(theModel, [CLM_train, CLM_test], 'charts3e.pdf')

#print classification accuracy
#printAccuracy("Classification Accuracy", [CLM_train, CLM_test])


# AMOUNT MODEL

AMT = LinearRegression()
AMT= AMT.fit(W_train[vars_tree_amt], Z_train[TARGET_A])

#get scores for train and test sets
AMT_train = getAmtAccuracyScores(theModel + "_Train", AMT, W_train[vars_tree_amt], Z_train[TARGET_A])
AMT_test = getAmtAccuracyScores(theModel, AMT, W_test[vars_tree_amt], Z_test[TARGET_A])

#print RMSE accuracy
#printAccuracy("RMSE Accuracy", [AMT_train, AMT_test])

#get coeficients
##REG_TREE_CLM_COEF = getCoefLogit(CLM, X_train[vars_tree_flag])
##REG_TREE_AMT_COEF = getCoefLinear(AMT, X_train[vars_tree_amt])


#Store results for comparison
REG_TREE_CLM = CLM_test.copy()
REG_TREE_AMT = AMT_test.copy()



"""
Regression- Random Forest variables
"""

theModel = "REG_RF"

RF_flag = []
for i in vars_RF_flag:
    RF_flag.append(i[0])

RF_amt = []
for i in vars_RF_amt:
    RF_amt.append(i[0])

# CLASSIFICATION MODEL
CLM = LogisticRegression(solver='newton-cg', max_iter=3000)
CLM = CLM.fit(X_train[RF_flag], Y_train[TARGET_F])

CLM_train = getProbAccuracyScores(theModel + "_Train", CLM, X_train[RF_flag], Y_train[TARGET_F])
CLM_test = getProbAccuracyScores(theModel, CLM, X_test[RF_flag], Y_test[TARGET_F])

#save or print ROC curve
saveRocCurve(theModel, [CLM_train, CLM_test], 'charts3f.pdf')

#print classification accuracy
#printAccuracy("Classification Accuracy", [CLM_train, CLM_test])


# AMOUNT MODEL

AMT = LinearRegression()
AMT= AMT.fit(W_train[RF_amt], Z_train[TARGET_A])

#get scores for train and test sets
AMT_train = getAmtAccuracyScores(theModel + "_Train", AMT, W_train[RF_amt], Z_train[TARGET_A])
AMT_test = getAmtAccuracyScores(theModel, AMT, W_test[RF_amt], Z_test[TARGET_A])

#print RMSE accuracy
#printAccuracy("RMSE Accuracy", [AMT_train, AMT_test])

#get coeficients
##REG_RF_CLM_COEF = getCoefLogit(CLM, X_train[RF_flag])
##REG_RF_AMT_COEF = getCoefLinear(AMT, X_train[RF_amt])


#Store results for comparison
REG_RF_CLM = CLM_test.copy()
REG_RF_AMT = AMT_test.copy()




"""
Regression- Gradient Boosting variables
"""

theModel = "REG_GB"

GB_flag = []
for i in vars_GB_flag:
    GB_flag.append(i[0])

GB_amt = []
for i in vars_GB_amt:
    GB_amt.append(i[0])

# CLASSIFICATION MODEL
CLM = LogisticRegression(solver='newton-cg', max_iter=3000)
CLM = CLM.fit(X_train[GB_flag], Y_train[TARGET_F])

CLM_train = getProbAccuracyScores(theModel + "_Train", CLM, X_train[GB_flag], Y_train[TARGET_F])
CLM_test = getProbAccuracyScores(theModel, CLM, X_test[GB_flag], Y_test[TARGET_F])

#save or print ROC curve
saveRocCurve(theModel, [CLM_train, CLM_test], 'charts3g.pdf')

#print classification accuracy
#printAccuracy("Classification Accuracy", [CLM_train, CLM_test])


# AMOUNT MODEL

AMT = LinearRegression()
AMT= AMT.fit(W_train[GB_amt], Z_train[TARGET_A])

#get scores for train and test sets
AMT_train = getAmtAccuracyScores(theModel + "_Train", AMT, W_train[GB_amt], Z_train[TARGET_A])
AMT_test = getAmtAccuracyScores(theModel, AMT, W_test[GB_amt], Z_test[TARGET_A])

#print RMSE accuracy
#printAccuracy("RMSE Accuracy", [AMT_train, AMT_test])

#get coeficients
##REG_GB_CLM_COEF = getCoefLogit(CLM, X_train[GB_flag])
##REG_GB_AMT_COEF = getCoefLinear(AMT, X_train[GB_amt])

#Store results for comparison
REG_GB_CLM = CLM_test.copy()
REG_GB_AMT = AMT_test.copy()



"""
Regression- Stepwise Selection variables
Utilizing statsmodels api as described from https://towardsdatascience.com/stepwise-regression-tutorial-in-python-ebf7c782c922

NOT FEASIBLE FOR MANY VARIABLES

"""

import statsmodels.api as sm

def getStats(X_DATA, Y_DATA):
    results = sm.OLS(Y_DATA, X_DATA).fit()
    print(results.summary())

theModel = "REG_STEP"

# CLASSIFICATION MODEL

# Variable Selection

# beginning with tree model variables (most variables of all models)
vars_step_flag = vars_tree_flag.copy()
#getStats(X_train[vars_step_flag], Y_train[TARGET_F])

# sytematically remove variable with largest p-value above .05
vars_step_flag.remove("TRUNC_IMP_VALUE")
#getStats(X_train[vars_step_flag], Y_train[TARGET_F])
vars_step_flag.remove("TRUNC_IMP_MORTDUE")
#getStats(X_train[vars_step_flag], Y_train[TARGET_F])

CLM = LogisticRegression(solver='newton-cg', max_iter=3000)
CLM = CLM.fit(X_train[vars_step_flag], Y_train[TARGET_F])

CLM_train = getProbAccuracyScores(theModel + "_Train", CLM, X_train[vars_step_flag], Y_train[TARGET_F])
CLM_test = getProbAccuracyScores(theModel, CLM, X_test[vars_step_flag], Y_test[TARGET_F])

# save or print ROC curve
saveRocCurve(theModel, [CLM_train, CLM_test], 'charts3h.pdf')

#print classification accuracy
#printAccuracy("Classification Accuracy", [CLM_train, CLM_test])


# AMOUNT MODEL


# Variable Selection

# beginning with GB model variables (most variables of all models)
vars_step_amt = GB_amt.copy()
#getStats(W_train[vars_step_amt], Z_train[TARGET_A])

#sytematically remove variable with largest p-value above .05
#no variables above .05; should give same results as GB

AMT = LinearRegression()
AMT= AMT.fit(W_train[vars_step_amt], Z_train[TARGET_A])

#get scores for train and test sets
AMT_train = getAmtAccuracyScores(theModel + "_Train", AMT, W_train[vars_step_amt], Z_train[TARGET_A])
AMT_test = getAmtAccuracyScores(theModel, AMT, W_test[vars_step_amt], Z_test[TARGET_A])

#print RMSE accuracy
#printAccuracy("RMSE Accuracy", [AMT_train, AMT_test])

#get coeficients
##REG_GB_CLM_COEF = getCoefLogit(CLM, X_train[GB_flag])
##REG_GB_AMT_COEF = getCoefLinear(AMT, X_train[GB_amt])

#Store results for comparison
REG_STEP_CLM = CLM_test.copy()
REG_STEP_AMT = AMT_test.copy()






"""
ASSIGNMENT 4: Neural Networks
"""

def getTensorFlowProbAccuracyScores(NAME, MODEL, X, Y):
    probs = MODEL.predict(X)
    pred_list = []
    for i in probs:
        pred_list.append(np.argmax(i))
    pred = np.array(pred_list)
    acc_score = metrics.accuracy_score(Y, pred)
    fpr, tpr, threshold = metrics.roc_curve(Y, probs[:,1])
    auc = metrics.auc(fpr, tpr)
    return [NAME, acc_score, fpr, tpr, auc]


theModel = "TensorFlow"

# CLASSIFICATION MODEL

# normalize our data
theScaler = MinMaxScaler()
theScaler.fit(X_train)


U_train = theScaler.transform(X_train)
U_test = theScaler.transform(X_test)

# put normalized data into dataframe
U_train = pd.DataFrame(U_train)
U_test = pd.DataFrame(U_test)
U_train.columns = list(X_train.columns.values)
U_test.columns = list(X_test.columns.values)

# variable selection
##U_train = U_train[RF_flag]
##U_test = U_test[RF_flag]

# define the model
F_theShapeSize = U_train.shape[1]               # size of the dataset
F_theActivation = tf.keras.activations.relu     # relu is most common
F_theLossMetric = tf.keras.losses.SparseCategoricalCrossentropy()
F_theOptimizer = tf.keras.optimizers.Adam()     # Adam is most common
F_theEpochs = 100

F_theUnits = int(2*F_theShapeSize / 3)              # starting point (no more than 2 * theShapeSize)

F_LAYER_01 = tf.keras.layers.Dense(units=F_theUnits, activation=F_theActivation, input_dim=F_theShapeSize)
#F_LAYER_02 = tf.keras.layers.Dense(units=F_theUnits, activation=F_theActivation)
F_LAYER_DROP = tf.keras.layers.Dropout(0.2)     # drop 20%
F_LAYER_OUTPUT = tf.keras.layers.Dense(units = 2, activation=tf.keras.activations.softmax)

CLM = tf.keras.Sequential()
CLM.add(F_LAYER_01)
CLM.add(F_LAYER_DROP)          
#CLM.add(F_LAYER_02)           
CLM.add(F_LAYER_OUTPUT)
CLM.compile(loss=F_theLossMetric, optimizer=F_theOptimizer)
CLM.fit(U_train, Y_train[TARGET_F], epochs=F_theEpochs, verbose=False)


# evaluate the model
CLM_train = getTensorFlowProbAccuracyScores(theModel + "_TRAIN", CLM, U_train, Y_train[TARGET_F])
CLM_test = getTensorFlowProbAccuracyScores(theModel, CLM, U_test, Y_test[TARGET_F])

saveRocCurve(theModel, [CLM_train, CLM_test], 'charts3i.pdf')
printAccuracy("Classification Accuracy", [CLM_train, CLM_test])



# AMOUNT MODEL

# normalzie data
V_train = theScaler.transform(W_train)
V_test = theScaler.transform(W_test)

# put normalized data into dataframe
V_train = pd.DataFrame(V_train)
V_test = pd.DataFrame(V_test)
V_train.columns = list(W_train.columns.values)
V_test.columns = list(W_test.columns.values)

# variable selection
##V_train = V_train[GB_flag]
##V_test = V_test[GB_flag]

# define the model
A_theShapeSize = V_train.shape[1]               # size of the dataset
A_theActivation = tf.keras.activations.relu     # relu is most common
A_theLossMetric = tf.keras.losses.MeanSquaredError()
A_theOptimizer = tf.keras.optimizers.Adam()     # Adam is most common
A_theEpochs = 100

A_theUnits = int(2*A_theShapeSize * .95)              # starting point (no more than 2 * theShapeSize)

A_LAYER_01 = tf.keras.layers.Dense(units=A_theUnits, activation=A_theActivation, input_dim=A_theShapeSize)
A_LAYER_02 = tf.keras.layers.Dense(units=A_theUnits, activation=A_theActivation)
#A_LAYER_DROP = tf.keras.layers.Dropout(0.2)     # drop 20%
A_LAYER_OUTPUT = tf.keras.layers.Dense(units = 1, activation=tf.keras.activations.linear)

AMT = tf.keras.Sequential()
AMT.add(A_LAYER_01)
#AMT.add(A_LAYER_DROP)          
AMT.add(A_LAYER_02)            
AMT.add(A_LAYER_OUTPUT)
AMT.compile(loss=A_theLossMetric, optimizer=A_theOptimizer)
AMT.fit(V_train, Z_train[TARGET_A], epochs=A_theEpochs, verbose=False)


# evaluate the model
AMT_train = getAmtAccuracyScores(theModel + "_TRAIN", AMT, V_train, Z_train[TARGET_A])
AMT_test = getAmtAccuracyScores(theModel, AMT, V_test, Z_test[TARGET_A])


printAccuracy("RMSE Accuracy", [AMT_train, AMT_test])


# save for comparison
TF_CLM = CLM_test.copy()
TF_AMT = AMT_test.copy()



"""
Compare Models
"""

#Compare Roc Curve Data
saveRocCurve("ALL MODELS", [TREE_CLM, RF_CLM, GB_CLM, REG_ALL_CLM, REG_TREE_CLM, REG_RF_CLM, REG_GB_CLM, REG_STEP_CLM, TF_CLM], 'charts3z.pdf')

#Compare RMSE
AMTS = [TREE_AMT, RF_AMT, GB_AMT, REG_ALL_AMT, REG_TREE_AMT, REG_RF_AMT, REG_GB_AMT, REG_STEP_AMT, TF_AMT]
print("Root Mean Squared Error by Model")

for i in AMTS:
    print(i[0], ":", i[1])
