def cv_model(clf, train_x, train_y, test_x, clf_name):
    folds = 5
    seed = 2021
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    test = np.zeros((test_x.shape[0],4))

    cv_scores = []
    onehot_encoder = OneHotEncoder(sparse=False)
    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print('************************************ {} ************************************'.format(str(i+1)))
        trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], train_y[valid_index]
        
        if clf_name == "lgb":
            train_matrix = clf.Dataset(trn_x, label=trn_y)
            valid_matrix = clf.Dataset(val_x, label=val_y)

            params = {
                'boosting_type': 'gbdt',
                'objective': 'multiclass',
                'num_class': 4,
                'num_leaves': 2 ** 5,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 4,
                'learning_rate': 0.1,
                'seed': seed,
                'nthread': 28,
                'n_jobs':24,
                'verbose': -1,
            }

            model = clf.train(params, 
                      train_set=train_matrix, 
                      valid_sets=valid_matrix, 
                      num_boost_round=2000, 
                      verbose_eval=100, 
                      early_stopping_rounds=200)
            val_pred = model.predict(val_x, num_iteration=model.best_iteration)
            test_pred = model.predict(test_x, num_iteration=model.best_iteration) 
            
        val_y=np.array(val_y).reshape(-1, 1)
        val_y = onehot_encoder.fit_transform(val_y)
        print('预测的概率矩阵为：')
        print(test_pred)
        test += test_pred
        score=abs_sum(val_y, val_pred)
        cv_scores.append(score)
        print(cv_scores)
    print("%s_scotrainre_list:" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    print("%s_score_std:" % clf_name, np.std(cv_scores))
    test=test/kf.n_splits

    return test


def lgb_model(x_train, y_train, x_test):
    lgb_test = cv_model(lgb, x_train, y_train, x_test, "lgb")
    return lgb_test


#这个过程很慢，如果使用DSW，请切换到GPU模式
lgb_test = lgb_model(x_train, y_train, x_test)