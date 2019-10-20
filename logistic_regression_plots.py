def conf_matrix(lr=lr, X_train=None, y_train=None):
    fpr_list = []
    tpr_list = []
    for fpr in np.linspace(0, 1, 100):
        # predict based on predict_proba threshold 
        predicts = []
        for item in lr.predict_proba(X_train):
            if item[0] <= fpr:
                predicts.append(1)
            else: 
                predicts.append(0)
                
        conf_matrix = pd.DataFrame(confusion_matrix(y_train, predicts),
                                   index= ['actual 0', 'actual 1'],
                                  columns= ['predicted 0', 'predicted 1'])
        # assign tp, tn, fp, fn
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
    plt.plot(fpr_lst, tpr_lst, color='darkorange', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1])
    plt.ylim([-0.01, 1])
    # plt.yticks([i/20.0 for i in range(21)])
    # plt.xticks([i/20.0 for i in range(21)])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=15)
    plt.ylabel('True Positive Rate (Sensitibity)', fontsize=15)
    plt.title('Receiver operating characteristic (ROC) Curve', fontsize=20)
    plt.legend(loc="lower right")
    print('AUC: {}'.format(auc(fpr_lst, tpr_lst)))
    plt.show()
    
    # plt.figure(figsize = (10, 8))
    # plt.title('Receiver operating characteristic (ROC) Curve', fontsize = 20)
    # plt.xlabel('False Positive Rate (1 - Specificity)', fontsize = 15)
    # plt.ylabel('True Positive Rate (Sensitibity)', fontsize = 15)
    # plt.xlim(-0.01, 1)
    # plt.ylim(-0.01, 1)
    # plt.plot(fpr_list, tpr_list, color='darkorange', lw=2, label='ROC curve');
    # plt.plot([0, 1], [0, 1], lw=2, linestyle='--');
    # plt.show()
    # return conf_matrix, fpr_list, tpr_list