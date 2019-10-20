""" Minimize the cost."""

def minimize_cost(num_thres=100, p_fp=3, p_tn=0.5, p_tp=1, p_fn=2,
                  lr=None, X_train=None, y_train=None):
    """Returns fpr, tpr, cost, and _thres. This function is to minimize the cost function."""
    _thres = []
    tpr = []
    fpr = []
    cost = []

    prediction = lr.predict_proba(X_train)
    # Different code for same objective to calculate metrics at thresholds
    for thres in np.linspace(0.01, 1, num_thres):

        _thres.append(thres)
        predicts = np.zeros((prediction.shape[0], 1))
        predicts[np.where(prediction[:, 1] >= thres)] = 1

        conf_matrix = confusion_matrix(y_train, predicts)

        tp = conf_matrix[1, 1]
        tn = conf_matrix[0, 0]
        fp = conf_matrix[0, 1]
        fn = conf_matrix[1, 0]

        sensitivity = tp / (tp + fn)
        # tnr =
        specificity = tn / (tn + fp)
        # fnr = 1 - sensitivity

        tpr.append(sensitivity)
        fpr.append(1 - specificity)

        # add a cost function (this involves domain knowledge)
        current_cost = (p_fp * fp) + (p_tn * tn) + (p_tp * tp) + (p_fn * fn)
        cost.append(current_cost)

    return fpr, tpr, cost, _thres



"""Cross validation."""

def cross_validation(n, shuffle=True, lr=None):
    """Cross validate logistic regression model n times."""
    "Retutns the train (index[0]) and test (index[1]) scores for chosen lr model"

    cv = StratifiedKFold(n_splits=n, random_state=1019, shuffle=True)

    # vanilla cross validation
    if lr == 'vanilla':
        lr_vanilla = LogisticRegression(C=1e9,
                                        solver='newton-cg',
                                        max_iter=1000)

        cv_vanilla = cross_validate(estimator=lr_vanilla,
                                    X=X_train, y=y_train,
                                    cv=cv,
                                    n_jobs=-1,
                                    return_estimator=True,
                                    return_train_score=True)

        vanilla_result = np.concatenate(
            (cv_vanilla['train_score'].reshape(-1, 1),
             cv_vanilla['test_score'].reshape(-1, 1)), axis=1)

        return vanilla_result

    # l2 or ridge cross validation
    if lr == 'l2':
        l2_reg = LogisticRegression(C=1,
                                    solver='newton-cg',
                                    max_iter=1000)

        cv_l2 = cross_validate(estimator=l2_reg, X=X_train, y=y_train,
                               cv=cv,
                               n_jobs=-1,
                               return_estimator=True,
                               return_train_score=True)

        l2_result = np.concatenate(
            (cv_l2['train_score'].reshape(-1, 1),
             cv_l2['test_score'].reshape(-1, 1)), axis=1)

        return l2_result

    # l1 or lasso cross validation
    if lr == 'l1':
        l1_reg = LogisticRegression(C=1,
                                    solver='saga',
                                    penalty='l1',
                                    max_iter=1000)

        cv_l1 = cross_validate(estimator=l1_reg, X=X_train, y=y_train,
                               cv=cv,
                               n_jobs=-1,
                               return_estimator=True,
                               return_train_score=True)

        l1_result = np.concatenate(
            (cv_l1['train_score'].reshape(-1, 1),
             cv_l1['test_score'].reshape(-1, 1)), axis=1)

        return l1_result
