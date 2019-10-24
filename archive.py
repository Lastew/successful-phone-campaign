def precision_recall(lr=None, X_train=None, y_train=None):
    """Return confusion_matrix, and list of recall and precision."""
    recall_list = []
    precision_list = []
    for x in np.linspace(0, 1, 100):
        # predict based on predict_proba threshold
        predicts = []
        for item in lr.predict_proba(X_train):
            if item[0] <= x:
                predicts.append(1)
            else:
                predicts.append(0)

        conf_matrix = pd.DataFrame(confusion_matrix(y_train, predicts),
                                   index=['actual 0', 'actual 1'],
                                   columns=['predicted 0', 'predicted 1'])
        # assign TP, TN, FP, FN
        true_positives = conf_matrix['predicted 1'][1]
        true_negatives = conf_matrix['predicted 0'][0]
        false_positives = conf_matrix['predicted 1'][0]
        false_negatives = conf_matrix['predicted 0'][1]

        # Calculate Sensitivity and Specificity
        sensitivity = true_positives / (true_positives + false_negatives)
        # recall = true_positives / (true_positives + false_negatives)
        specificity = true_negatives / (true_negatives + false_positives)
        # precision = true_positives / (true_positives + false_positives)

        # Append to lists to graph
        recall_list.append((recall, x))
        precision_list.append((precision, x))

    return conf_matrix, recall_list, precision_list

# def plot_threshold(min_cost_threshold, fpr=None, tpr=None):
#     ax = plt.figure(figsize = (10, 8))
#     plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize = 20)
#     plt.xlabel('False Positive Rate (1 - Specificity)', fontsize = 15)
#     plt.ylabel('True Positive Rate (Sensitibity)', fontsize = 15)
#     plt.xlim(-.01, 1.01)
#     plt.ylim(-.01, 1.01)
#     plt.plot(fpr, tpr);
#     plt.plot([0, 1], [0, 1]);
#     plt.scatter(min_cost_threshold[0], min_cost_threshold[1], marker ='o', color = 'red', s=250)
#     ax.text(min_cost_threshold[0] + 0.06, min_cost_threshold[1] - 0.03, 'Threshold:'+ str(round(min_cost_threshold[2], 2)))


# def nice_confusion(model, y_true= None, y_pred=None): #X_train=None, y_train=None,
#     """Creates a nice looking confusion matrix"""
#     plt.figure(figsize=(10, 10))
#     plt.xlabel('Predicted Class', fontsize=18)
#     plt.ylabel('True Class', fontsize=18)
# #     plt.xticks(labels=[''])
#     viz = ConfusionMatrix(
#         model,
#         cmap='PuBu', fontsize=18)
# #     viz.fit(X_train, y_train)
#     viz.score(y_true, y_pred)
#     viz.poof()
