import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random

# Model imports
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import f_classif
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier



def preprocessing(rawData):
    """
    All preprocessing of raw data
    """
    # Fill NaN with mode
    rawData['clinical_lvi'] = rawData['clinical_lvi'].fillna(rawData['clinical_lvi'].mode()[0])
    rawData['path_lvi'] = rawData['path_lvi'].fillna(rawData['path_lvi'].mode()[0])
    rawData['path_grade'] = rawData['path_grade'].fillna(rawData['path_grade'].mode()[0])
    rawData['recurrence'] = rawData['recurrence'].fillna(rawData['recurrence'].mode()[0])
    rawData['tls'] = rawData['tls'].fillna(rawData['tls'].mode()[0])

    for i in rawData.columns[21:]:
        rawData[i] = rawData[i].fillna(0)

    # Rows to be normalized
    normRows = ["age_diagnosis", "clinical_stage", "path_stage"]

    for i in normRows:
        max_value = rawData[i].max()
        min_value = rawData[i].min()
        rawData[i] = (rawData[i] - min_value) / (max_value - min_value)
        rawData

    rawData.drop(rawData.loc[rawData['Slice']!="Full"].index, inplace=True)

    dropRawData = rawData.drop(["ROI", 
                                "Slice", 
                                "Slide", "patient_id", "dead",
                                "Unknown"], 1)
    dropRawData = dropRawData[dropRawData.columns[16:]]
    dropRawData["act"] = rawData["act"]

    # # No outliers in this data, it's already been dealt with in pre-processing
    # plt.figure(figsize=(15,10))
    # dropRawData.boxplot()

    return dropRawData


def feat_ranked_selection(X, y, k):
    """
    Returns the first k columns based on ANOVA ranking
    """
    from sklearn.feature_selection import SelectKBest, chi2

    selector = SelectKBest(f_classif, k=k)
    selector.fit(X, y)
    # Get columns to keep and create new dataframe with those only
    cols = selector.get_support(indices=True)
    X_selected = X.iloc[:,cols]
    
    # Append patient_id and recurrence
    X_selected["patient_id"] = rawData["patient_id"]
    X_selected["act"] = rawData["act"]

    return X_selected


def feat_selection(X, feat):
    """
    Returns X but with only the features that were given as input
    """
    X_selected = X[feat]
    # Append patient_id and recurrence
    X_selected["patient_id"] = rawData["patient_id"]
    X_selected["act"] = rawData["act"]
    
    return X_selected


def get_sample_order(recurrPatient, nonRecurrPatient):
    """
    Returns the sample order for the folds, so that 2 recurrent 1 non-recurrent patient
    is in each fold.
    """
    smokerSample = random.sample(smokers, len(smokers))
    nonSmokerSample = random.sample(nonSmokers, len(nonSmokers))

    count = 0
    sampleOrder = []

    while count < 12:
        if count in [0,4,8]:
            patientId = smokerSample[0]
            sampleOrder.append(patientId)
            smokerSample.remove(patientId)
            count+=1
        else:
            patientId = nonSmokerSample[0]
            sampleOrder.append(patientId)
            nonSmokerSample.remove(patientId)
            count+=1

    sampleOrder = sampleOrder+smokerSample+nonSmokerSample
    
    return sampleOrder


def fold_labelling(X, sampleOrder):
    """
    Returns X with "fold_label" column that contains which fold each sample should be in for cross validation
    """
    X["fold_label"] = ""
    sampleOrderCount = 0
    
    for t in sampleOrder:
        if sampleOrderCount <= 3:
            X.loc[X['patient_id'].isin([t]), ['fold_label']] = 0
        elif sampleOrderCount <= 7:
            X.loc[X['patient_id'].isin([t]), ['fold_label']] = 1
        elif sampleOrderCount <= 11:
            X.loc[X['patient_id'].isin([t]), ['fold_label']] = 2
        else:
            X.loc[X['patient_id'].isin([t]), ['fold_label']] = 3
        sampleOrderCount += 1
        
    return X


def balance_folds(X):
    """
    Returns balanced dataset, such that each fold is balanced separately
    """
    catFeats = dropRawData.columns[1:16]
    
    # Balance each fold separately
    fold0 = X[X["fold_label"]==0]
    fold1 = X[X["fold_label"]==1]
    fold2 = X[X["fold_label"]==2]
    fold3 = X[X["fold_label"]==3]
    folds = [fold0, fold1, fold2, fold3]

    catIndexList = []
    catIndex=0

    for cat_i in fold0.columns:
        if cat_i in catFeats:
            catIndexList.append(catIndex)
            catIndex += 1

    X_balanced = pd.DataFrame()

    for fold in folds:
        y_fold = fold["act"]

        sm = SMOTE(k_neighbors=1)

        X_Trainfold, y_Trainfold = sm.fit_resample(fold, fold["act"])

        X_balanced = X_balanced.append(X_Trainfold, ignore_index=True)
        
    return X_balanced


def evaluate_features(varX, varY, balX, model):
    """General helper function for evaluating effectiveness of passed features in ML model
        Prints out Log loss, accuracy, and confusion matrix with 3-fold stratified cross-validation
        Args:
        X (array-like): Features array. Shape (n_samples, n_features)
        y (array-like): Labels array. Shape (n_samples,)        
        clf: Classifier to use. If None, default Log reg is use.
        Log loss should be closer to zero
    """
    from sklearn.model_selection import LeaveOneGroupOut

    groups = varX["fold_label"]
    varX_drop = varX.drop(["fold_label"],1)
    logo = LeaveOneGroupOut().split(varX_drop, varY, groups)
    probas = cross_val_predict(model, varX_drop, varY, cv=logo, n_jobs=-1, verbose=2, method='predict_proba')
    pred_indices = np.argmax(probas, axis=1)
    classes = np.unique(varY)
    preds = classes[pred_indices]
    tn, fp, fn, tp = confusion_matrix(varY, preds).ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp+fn)
    
    return accuracy_score(varY, preds), sensitivity, specificity


def get_incorr_patients(X, name, run):
    """
    Updates X dataset to show predictions as well as which were correct/incorrect. Saves to file
    Returns patient id's of the incorrect patients
    """
    X['result'] = np.where(X['cis'] == X['prediction'], 'correct', 'incorrect')
    X.to_csv("C:\\Users\\sindhura\\OneDrive - Queen's University\\Queens\\Research MSc\\Bladder Data\\Predictions\\Runs\\corrected_knnAccuracy_" + name + str(run) + ".csv")
    incorrResults = X[X["result"] == "incorrect"]
    incorrPatients = np.unique(incorrResults["patient_id"])
    
    return incorrPatients


def create_acc_dict(accuracy, sensitivity, specificity, sampleOrder, X_before, incorrPatients):
    """
    Returns dictionary of the accuracy for each run
    """
    accDict = {}
    accDict["Patient Folds"] = [sampleOrder[x:x+3] for x in range(0, len(sampleOrder), 3)]
    accDict["Accuracy"] = accuracy
    accDict["Sensitivity"] = sensitivity
    accDict["Specificity"] = specificity
    
    for pat in incorrPatients:
        patient = X_before[X_before["patient_id"] == pat]
        num = len(patient[patient["result"]=="incorrect"])
        accDict["Patient " + str(pat)] = num
        
    return accDict


def classifier_eval_all(Xlabelled, classifier):
    """
    Run train/test split and model evaluation, records results in finalDfRes
    """
    finalDfRes = pd.DataFrame()
    act = [32,56,33,59,61]
    nonAct = [23,60,54,8,6,45,64,22,19,55]

    for kNum in range(1,len(Xlabelled.columns)-1):
        knnAccDf = pd.DataFrame()
        # Select top ranked columns
        y = Xlabelled["act"]
        X = Xlabelled.drop(["act"],1)
        X_featSelect = feat_ranked_selection(X, y, kNum)
        for run in range(4):
            # Get sample order
            sample = get_sample_order(act, nonAct)
            # Append fold labels
            flabelData = fold_labelling(X_featSelect, sample)
            # Balance folds
            balancedData = balance_folds(flabelData)
            # Define X_Train and y_Train
            X_Train = balancedData.drop(["patient_id", "act"], 1)
            y_Train = balancedData["act"]
            # Do cross validation
            accuracy, sensitivity, specificity = evaluate_features(X_Train, y_Train, balancedData, classifier)
            # Create dictionary of accuracies for the run
            accuracyDict = create_acc_dict(accuracy, sensitivity, specificity)
            # Append dictionary to accuracy dataframe
            knnAccDf = knnAccDf.append(accuracyDict, ignore_index=True)
        finalRes = final_results(knnAccDf, kNum)
        finalDfRes = finalDfRes.append(finalRes, ignore_index=True)
    
    return finalDfRes


def final_results(accDf, name):
    """
    Calculate overall performance over all runs
    """
    finalDict = {}
    meanAccuracy = accDf["Accuracy"].mean()
    stdev = accDf["Accuracy"].std()
    finalDict["Features"] = name
    finalDict["Accuracy"] = meanAccuracy
    finalDict["St. Deviation"] = stdev
    finalDict["Sensitivity"] = accDf["Sensitivity"].mean()
    finalDict["Sensitivity StD"] = accDf["Sensitivity"].std()
    finalDict["Specificity"] = accDf["Specificity"].mean()
    finalDict["Specificity StD"] = accDf["Specificity"].std()
    
    return finalDict

    

def main():
    rawData = pd.read_csv("C:\\Users\\sindhura\\OneDrive - Queen's University\\Queens\\Research MSc\\Bladder Data\\slicedTrainData_50slices30angle.csv")
    dropRawData = preprocessing(rawData)

    # Evaluate classifier performance
    knn = KNeighborsClassifier()
    rf = RandomForestClassifier(n_estimators=1000, verbose=1)
    lr = LogisticRegression()
    dt = DecisionTreeClassifier()
    estimators=[('dt', dt), ('lr', lr), ('rf', rf), ('knn', knn)]
    #create our voting classifier, inputting our models
    ensemble = VotingClassifier(estimators, voting="soft")

    knnAcc = classifier_eval_all(dropRawData,knn)
    rfAcc = classifier_eval_all(dropRawData, rf)
    lrAcc = classifier_eval_all(dropRawData, lr)
    dtAcc = classifier_eval_all(dropRawData, dt)
    ensAcc = classifier_eval_all(dropRawData, ensemble)
    
    graphRes = pd.DataFrame()
    graphRes["KNN"] = knnAcc["Accuracy"]
    graphRes["LR"] = lrAcc["Accuracy"]
    graphRes["DT"] = dtAcc["Accuracy"]
    graphRes["RF"] = rfAcc["Accuracy"]
    graphRes["Ensemble"] = ensAcc["Accuracy"]

    graphRes["n_features"] = list(range(1,22))
    graphRes=graphRes.set_index('n_features')

    graphRes.to_csv("C:\\Users\\sindhura\\OneDrive - Queen's University\\Queens\\Research MSc\\Bladder Data\\Feature Scoring\\actClassifierScores.csv")

    fig, ax = plt.subplots(figsize = (15,10))

    sns.lineplot(data=graphRes)
    plt.show()

    knnResults = final_results(knnAcc, "K-NN")
    rfResults = final_results(rfAcc, "RF")
    lrResults = final_results(lrAcc, "LR")
    dtResults = final_results(dtAcc, "DT")
    ensResults = final_results(ensAcc, "Ensemble")

    resultsDf = pd.DataFrame()

    resultsDf = resultsDf.append(knnResults, ignore_index=True)
    resultsDf = resultsDf.append(rfResults, ignore_index=True)
    resultsDf = resultsDf.append(lrResults, ignore_index=True)
    resultsDf = resultsDf.append(dtResults, ignore_index=True)
    resultsDf = resultsDf.append(ensResults, ignore_index=True)

    return resultsDf

main()
