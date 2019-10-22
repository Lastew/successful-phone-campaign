import numpy as np
import pandas as pd 

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, auc #, classifiction_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
from yellowbrick.classifier import ConfusionMatrix

def conf_matrix(lr=None, X_train=None, y_train=None):
    fpr_list = []
    tpr_list = []
    for x in np.linspace(0, 1, 100):
        # predict based on predict_proba threshold 
        predicts = []
        for item in lr.predict_proba(X_train):
            if item[0] <= 0.5:
                predicts.append(1)
            else: 
                predicts.append(0)
                
        conf_matrix = pd.DataFrame(confusion_matrix(y_train, predicts),
                                   index= ['actual 0', 'actual 1'],
                                  columns= ['predicted 0', 'predicted 1'])
        # assign TP, TN, FP, FN
        true_positives = conf_matrix['predicted 1'][1]
        true_negatives = conf_matrix['predicted 0'][0]
        false_positives = conf_matrix['predicted 1'][0]
        false_negatives = conf_matrix['predicted 0'][1]
        
        # Calculate Sensitivity and Specificity
        sensitivity = true_positives / (true_positives + false_negatives)
        specificity = true_negatives / (true_negatives + false_positives)
    
        # Append to lists to graph
        fpr_list.append(1 - specificity)
        tpr_list.append(sensitivity)
    return conf_matrix, fpr_list, tpr_list


def roc_curve(fpr_lst, tpr_lst):
     
    # Plot logesitc regression -- Seaborns Beautiful Styling
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    plt.figure(figsize=(10,8))
    plt.plot(fpr_lst, tpr_lst, lw=2, label='ROC curve( Area = %0.2f)'%np.round(auc(fpr_lst, tpr_lst),3),
             color ='darkorange') #darkorange
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1])
    plt.ylim([-0.01, 1])
    # plt.yticks([i/20.0 for i in range(21)])
    # plt.xticks([i/20.0 for i in range(21)])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=15)
    plt.ylabel('True Positive Rate (Sensitibity)', fontsize=15)
    plt.title('Receiver operating characteristic (ROC) Curve', fontsize=20)
    plt.legend(loc="lower right")
    print('AUC: {}'.format(np.round(auc(fpr_lst, tpr_lst),3)))
    plt.show()
    
    
    
def nice_confusion(model, X_test= None, y_test=None): #X_train=None, y_train=None,
    """Creates a nice looking confusion matrix"""
    plt.figure(figsize=(10, 10))
    plt.xlabel('Predicted Class', fontsize=18)
    plt.ylabel('True Class', fontsize=18)
#     plt.xticks(labels=[''])
    viz = ConfusionMatrix(
        model,
        cmap='PuBu', fontsize=18)
#     viz.fit(X_train, y_train)
    viz.score(X_test, y_test)
    viz.poof()
    
def plot_threshold(min_cost_threshold, fpr=None, tpr=None):
    ax = plt.figure(figsize = (10, 8))
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize = 20)
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize = 15)
    plt.ylabel('True Positive Rate (Sensitibity)', fontsize = 15)
    plt.xlim(-.01, 1.01)
    plt.ylim(-.01, 1.01)
    plt.plot(fpr, tpr);
    plt.plot([0, 1], [0, 1]);
    plt.scatter(min_cost_threshold[0], min_cost_threshold[1], marker ='o', color = 'red', s=250)
    ax.text(min_cost_threshold[0] + 0.06, min_cost_threshold[1] - 0.03, 'Threshold:'+ str(round(min_cost_threshold[2], 2)))
