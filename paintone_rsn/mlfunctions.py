import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, LeaveOneOut, cross_val_predict,LeaveOneGroupOut, GroupKFold, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score, explained_variance_score
from scipy import stats
import statsmodels.api as sm
from matplotlib.colors import ListedColormap
import seaborn as sns
#from  Fear_conditiong_and_Rsns.data_preproc import removesubj_basedonotherbehavvalue
from mlxtend.evaluate import permutation_test
from ray.tune.sklearn import TuneGridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler

def io_definer(data,
               xvar_colnm,
               yvar_colnm,
               outl_upperlimit=(),
               outl_lowerlimit=()):
    '''
    Visualize the target vairables and if user can specify upper and lower limits to exclude subjects bassed on that variable.
    :param data: a pandass data frame include all the predictor and optional target variables
    :param xvar_colnm: the column names of predictor variables
    :param yvar_colnm: the column names of target variable
    :param outl_upperlimit: the upper limit for the behavior variable
    :param outl_lowerlimit: the lower limit for the behavior varlable
    :return: two numpy array which contains the outcome and the predictors, and one which contains the index of the outlier
    '''

    print('We work with the following target variable:',yvar_colnm)
    print('Which has the following values'+str(data[yvar_colnm].values))
    plt.hist((data[yvar_colnm].values))
    plt.title('The values of' + str(yvar_colnm))
    plt.show()
    # exclude outliers based on certain criteria
    if outl_lowerlimit and not outl_upperlimit:
        datawithexclusion = data[(data[yvar_colnm] > outl_lowerlimit)]
        outl_idx = data[data[yvar_colnm] < outl_lowerlimit].index.values
    elif outl_upperlimit and not outl_lowerlimit:
        datawithexclusion = data[(data[yvar_colnm] < outl_upperlimit)]
        outl_idx = data[data[yvar_colnm] > outl_upperlimit].index.values
    elif outl_upperlimit and outl_lowerlimit:
        datawithexclusion = data[(data[yvar_colnm] > outl_lowerlimit) & (data[yvar_colnm] < outl_upperlimit) ]
        outl_idx = data[(data[yvar_colnm] < outl_lowerlimit) | (data[yvar_colnm] > outl_upperlimit) ].index.values
    else:
        datawithexclusion = data
        outl_idx = []

    plt.hist(datawithexclusion[yvar_colnm].values)
    plt.title('The values of'+ str(yvar_colnm)+'after exclusion')
    plt.show()
    X_predictor = datawithexclusion.loc[:,xvar_colnm].to_numpy()
    print('The shape of our predictor matrix:',X_predictor.shape)
    y_predictor = datawithexclusion.loc[:,yvar_colnm].to_numpy()
    print('The shape of our outcome variable:', y_predictor.shape)

    return y_predictor, X_predictor, outl_idx

def pipe_scale_fsel_ridge_noscaler( fsel=SelectKBest(f_regression),
                                    model=Ridge(max_iter=100000),
                                    p_grid = {'fsel__k': np.linspace(10, 200, 39,dtype=int), 'model__alpha': [.001, .005, .01, .05, .1, .5, 1, 10]} # exhaustive, takes a lot of time
                        #p_grid = {'fsel__k': [20, 25, 30, 35, 40, 45, 50, 60, 70, 80], 'model__alpha': [.001, .005, .01, .05, .1, .5]} # for fast re-calculation
                                    ):
    mymodel = Pipeline(
        [ ('fsel', fsel),
         ('model', model)])
    return mymodel, p_grid

def pipe_scale_fsel_ridge( scaler=RobustScaler(),
                          fsel=SelectKBest(f_regression),
                          model=Ridge(max_iter=100000),
                        p_grid = {'fsel__k': [10, 20, 30, 40, 50, 60, 70, 80], 'model__alpha': [.001, .005, .01, .05, .1, .5]} # for fast re-calculation
                        #p_grid = {'fsel__k': np.linspace(10, 200, 39,dtype=int), 'model__alpha': [.001, .005, .01, .05, .1, .5, 1, 10]} # exhaustive, takes a lot of time
                          ):
    mymodel = Pipeline(
        [('scaler', scaler),
         ('fsel', fsel),
         ('model', model)])
    return mymodel, p_grid

def mytrain(X, y, model, p_grid, nested=False, model_averaging=True,
            inner_cv=LeaveOneOut(),
            outer_cv = LeaveOneOut()):

    clf = GridSearchCV(estimator=model, param_grid=p_grid, cv=inner_cv,
                       scoring="neg_mean_squared_error", verbose=False, return_train_score=False, n_jobs=-1)
    clf.fit(X, y)

    print("**** Non-nested analysis ****")
    print("** Best hyperparameters: " + str(clf.best_params_))

    print("** Score on full data as training set:\t" + str(
        -mean_squared_error(y_pred=clf.best_estimator_.predict(X), y_true=y)))
    print("** Score on mean as model: " + str(-mean_squared_error(np.repeat(y.mean(), len(y)), y)))
    print("** Best Non-nested cross-validated score on test:\t" + str(clf.best_score_))

    model = clf.best_estimator_

    print("XXXXX Explained Variance: " + str(1 - clf.best_score_ / -mean_squared_error(np.repeat(y.mean(), len(y)), y)))
    explvar_nonnested=1 - clf.best_score_ / -mean_squared_error(np.repeat(y.mean(), len(y)), y)
    avg_model = None
    all_models = []
    if nested:
        print("**** Nested analysis ****")

        # nested_scores = cross_val_score(clf, X, y, cv=outer_cv, scoring="explained_variance")
        # print "** Nested Score on test:\t" + str(nested_scores.mean())
        # this above has the same output as this below:

        best_params = []
        predicted = np.zeros(len(y))
        actual = np.zeros(len(y))
        nested_scores_train = np.zeros(outer_cv.get_n_splits(X))
        nested_scores_test = np.zeros(outer_cv.get_n_splits(X))
        #nested_scores_test2 = np.zeros(outer_cv.get_n_splits(X))
        i = 0
        avg = []
        # doing the crossval itewrations manually
        print("model\tinner_cv mean score\touter vc score")
        for train, test in outer_cv.split(X, y):
            clf.fit(X[train], y[train])

            # model avaraging
            # RES, mat, labels = get_full_coef(X[train], clf.best_estimator_, plot=False)
            # avg.append(RES)
            all_models.append(clf.best_estimator_)
            # plot histograms to check distributions
            # bins = np.linspace(-1.5, 1.5, 6)
            # pyplot.hist(y[train], bins, alpha=0.5, label='train')
            # pyplot.hist(y[test], bins, alpha=0.5, label='test')
            # pyplot.legend(loc='upper right')
            # pyplot.show()

            print(str(clf.best_params_) + " " + str(clf.best_score_) + " " + str(clf.score(X[test], y[test])))
            predicted[i] = clf.predict(X[test])
            actual[i] = y[test]

            best_params.append(clf.best_params_)
            nested_scores_train[i] = clf.best_score_
            nested_scores_test[i] = clf.score(X[test], y[test])
            # clf.score is the same as calculating the score to the prediced values of the test dataset:
            # nested_scores_test2[i] = explained_variance_score(y_pred=clf.predict(X[test]), y_true=y[test])
            i = i + 1

        print("*** Score on mean as model:\t" + str(-mean_squared_error(np.repeat(y.mean(), len(y)), y)))
        print("** Mean score in the inner crossvaludation (inner_cv):\t" + str(nested_scores_train.mean()))
        print("** Mean Nested Crossvalidation Score (outer_cv):\t" + str(nested_scores_test.mean()))
        explvar = 1 - nested_scores_test.mean() / -mean_squared_error(np.repeat(y.mean(), len(y)), y)
        print("Explained Variance: " + str(
            1 - nested_scores_test.mean() / -mean_squared_error(np.repeat(y.mean(), len(y)), y)))
        print("Correlation: " + str(np.corrcoef(actual, predicted)[0, 1]))

        avg_model = np.mean(np.array(avg), axis=0)

        # plot the prediction of the outer cv
        fig, ax = plt.subplots()
        ax.scatter(actual, predicted, edgecolors=(0, 0, 0))
        # ax.plot([y.min(), y.max()],
        #           [y.min(), y.max()],
        #          'k--',
        #         lw=2)
        ax.set_xlabel('Measured behavior variable')
        ax.set_ylabel('Predicted (Nested LOO)')
        plt.title(
            "Expl. Var.:" + str(1 - nested_scores_test.mean() / -mean_squared_error(np.repeat(y.mean(), len(y)), y)) +
            "\nCorrelation: " + str(np.corrcoef(actual, predicted)[0, 1]))
        plt.show()
    else:
        all_models = [model]
        fig = []
        explvar = []
        predicted = []


    model.fit(X, y)  # fit to whole data

    return model, predicted,explvar, explvar_nonnested, all_models, clf, fig

def mytrain_tune(X, y, model, p_grid, nested=False, model_averaging=True,
            inner_cv=LeaveOneOut(),
            outer_cv = LeaveOneOut()):

    clf = TuneGridSearchCV(estimator=model, param_grid=p_grid, cv=inner_cv,
                       scoring="neg_mean_squared_error", verbose=False, return_train_score=False, n_jobs=-1)
    clf.fit(X, y)

    print("**** Non-nested analysis ****")
    print("** Best hyperparameters: " + str(clf.best_params_))

    print("** Score on full data as training set:\t" + str(
        -mean_squared_error(y_pred=clf.best_estimator_.predict(X), y_true=y)))
    print("** Score on mean as model: " + str(-mean_squared_error(np.repeat(y.mean(), len(y)), y)))
    print("** Best Non-nested cross-validated score on test:\t" + str(clf.best_score_))

    model = clf.best_estimator_

    print("XXXXX Explained Variance: " + str(1 - clf.best_score_ / -mean_squared_error(np.repeat(y.mean(), len(y)), y)))

    avg_model = None
    all_models = []
    if nested:
        print("**** Nested analysis ****")

        # nested_scores = cross_val_score(clf, X, y, cv=outer_cv, scoring="explained_variance")
        # print "** Nested Score on test:\t" + str(nested_scores.mean())
        # this above has the same output as this below:

        best_params = []
        predicted = np.zeros(len(y))
        actual = np.zeros(len(y))
        nested_scores_train = np.zeros(outer_cv.get_n_splits(X))
        nested_scores_test = np.zeros(outer_cv.get_n_splits(X))
        #nested_scores_test2 = np.zeros(outer_cv.get_n_splits(X))
        i = 0
        avg = []
        # doing the crossval itewrations manually
        print("model\tinner_cv mean score\touter vc score")
        for train, test in outer_cv.split(X, y):
            clf.fit(X[train], y[train])

            # model avaraging
            # RES, mat, labels = get_full_coef(X[train], clf.best_estimator_, plot=False)
            # avg.append(RES)
            all_models.append(clf.best_estimator_)
            # plot histograms to check distributions
            # bins = np.linspace(-1.5, 1.5, 6)
            # pyplot.hist(y[train], bins, alpha=0.5, label='train')
            # pyplot.hist(y[test], bins, alpha=0.5, label='test')
            # pyplot.legend(loc='upper right')
            # pyplot.show()

            print(str(clf.best_params_) + " " + str(clf.best_score_) + " " + str(clf.score(X[test], y[test])))
            predicted[i] = clf.predict(X[test])
            actual[i] = y[test]

            best_params.append(clf.best_params_)
            nested_scores_train[i] = clf.best_score_
            nested_scores_test[i] = clf.score(X[test], y[test])
            # clf.score is the same as calculating the score to the prediced values of the test dataset:
            # nested_scores_test2[i] = explained_variance_score(y_pred=clf.predict(X[test]), y_true=y[test])
            i = i + 1

        print("*** Score on mean as model:\t" + str(-mean_squared_error(np.repeat(y.mean(), len(y)), y)))
        print("** Mean score in the inner crossvaludation (inner_cv):\t" + str(nested_scores_train.mean()))
        print("** Mean Nested Crossvalidation Score (outer_cv):\t" + str(nested_scores_test.mean()))
        explvar = 1 - nested_scores_test.mean() / -mean_squared_error(np.repeat(y.mean(), len(y)), y)
        print("Explained Variance: " + str(
            1 - nested_scores_test.mean() / -mean_squared_error(np.repeat(y.mean(), len(y)), y)))
        print("Correlation: " + str(np.corrcoef(actual, predicted)[0, 1]))

        avg_model = np.mean(np.array(avg), axis=0)

        # plot the prediction of the outer cv
        fig, ax = plt.subplots()
        ax.scatter(actual, predicted, edgecolors=(0, 0, 0))
        # ax.plot([y.min(), y.max()],
        #           [y.min(), y.max()],
        #          'k--',
        #         lw=2)
        ax.set_xlabel('Measured behavior variable')
        ax.set_ylabel('Predicted (Nested LOO)')
        plt.title(
            "Expl. Var.:" + str(1 - nested_scores_test.mean() / -mean_squared_error(np.repeat(y.mean(), len(y)), y)) +
            "\nCorrelation: " + str(np.corrcoef(actual, predicted)[0, 1]))
        plt.show()
    else:
        all_models = [model]
        fig = []
        explvar = []


    model.fit(X, y)  # fit to whole data

    return model, avg_model, all_models, clf, fig, explvar

def mytrain_error(X, y, model, p_grid, nested=False, model_averaging=True):
    inner_cv = LeaveOneOut()
    outer_cv = LeaveOneOut()

    clf = GridSearchCV(estimator=model, param_grid=p_grid, cv=inner_cv,
                       scoring="neg_mean_squared_error", verbose=False, return_train_score=False, n_jobs=10)

    clf.fit(X, y)
    # create a gridsearch for the errorterm
    clf_error = GridSearchCV(estimator=model, param_grid=p_grid, cv=inner_cv,
                       scoring="neg_mean_squared_error", verbose=False, return_train_score=False, n_jobs=10)

    print("**** Non-nested analysis ****")
    print("** Best hyperparameters: " + str(clf.best_params_))

    print("** Score on full data as training set:\t" + str(
        -mean_squared_error(y_pred=clf.best_estimator_.predict(X), y_true=y)))
    print("** Score on mean as model: " + str(-mean_squared_error(np.repeat(y.mean(), len(y)), y)))
    print("** Best Non-nested cross-validated score on test:\t" + str(clf.best_score_))

    model = clf.best_estimator_

    print("XXXXX Explained Variance: " + str(1 - clf.best_score_ / -mean_squared_error(np.repeat(y.mean(), len(y)), y)))

    avg_model = None
    all_models = []
    if nested:
        print("**** Nested analysis ****")

        # nested_scores = cross_val_score(clf, X, y, cv=outer_cv, scoring="explained_variance")
        # print "** Nested Score on test:\t" + str(nested_scores.mean())
        # this above has the same output as this below:

        best_params = []
        predicted = np.zeros(len(y))
        actual = np.zeros(len(y))
        predicted_error = np.zeros(len(y))
        error = np.zeros(len(y))

        nested_scores_train = np.zeros(outer_cv.get_n_splits(X))
        nested_scores_test = np.zeros(outer_cv.get_n_splits(X))
        #nested_scores_test2 = np.zeros(outer_cv.get_n_splits(X))
        i = 0
        avg = []
        # doing the crossval itewrations manually
        print("model\tinner_cv mean score\touter vc score")
        for train, test in outer_cv.split(X, y):
            clf.fit(X[train], y[train])

            # model avaraging
            # RES, mat, labels = get_full_coef(X[train], clf.best_estimator_, plot=False)
            # avg.append(RES)
            all_models.append(clf.best_estimator_)
            # plot histograms to check distributions
            # bins = np.linspace(-1.5, 1.5, 6)
            # pyplot.hist(y[train], bins, alpha=0.5, label='train')
            # pyplot.hist(y[test], bins, alpha=0.5, label='test')
            # pyplot.legend(loc='upper right')
            # pyplot.show()




            print(str(clf.best_params_) + " " + str(clf.best_score_) + " " + str(clf.score(X[test], y[test])))
            predicted[i] = clf.predict(X[test])
            actual[i] = y[test]

            # calculate error:
            errorterm = np.abs(clf.best_estimator_.predict(X[train]) - y[train])
            clf_error.fit(X[train], errorterm)

            predicted_error[i] = clf_error.predict(X[test])
            error[i] = np.abs(predicted[i]-y[test])

            print(str(clf_error.best_params_) + " " + str(clf_error.best_score_) + " " + str(clf_error.score(X[test], error[[i]])))
            print("Predicted error: " + str(predicted_error[i]))
            print("Observed error: " + str(error[i]))
            best_params.append(clf.best_params_)
            nested_scores_train[i] = clf.best_score_
            nested_scores_test[i] = clf.score(X[test], y[test])
            # clf.score is the same as calculating the score to the prediced values of the test dataset:
            # nested_scores_test2[i] = explained_variance_score(y_pred=clf.predict(X[test]), y_true=y[test])
            i = i + 1

        print("*** Score on mean as model:\t" + str(-mean_squared_error(np.repeat(y.mean(), len(y)), y)))
        print("** Mean score in the inner crossvaludation (inner_cv):\t" + str(nested_scores_train.mean()))
        print("** Mean Nested Crossvalidation Score (outer_cv):\t" + str(nested_scores_test.mean()))

        print("Explained Variance: " + str(
            1 - nested_scores_test.mean() / -mean_squared_error(np.repeat(y.mean(), len(y)), y)))
        print("Correlation: " + str(np.corrcoef(actual, predicted)[0, 1]))

        avg_model = np.mean(np.array(avg), axis=0)

        # plot the prediction of the outer cv
        fig, ax = plt.subplots()
        #ax.scatter(actual, predicted, edgecolors=(0, 0, 0))
        ax.scatter(error, predicted_error, edgecolors=(0, 0, 0))
        # ax.plot([y.min(), y.max()],
        #           [y.min(), y.max()],
        #          'k--',
        #         lw=2)
        ax.set_xlabel('Measured behavior variable')
        ax.set_ylabel('Predicted (Nested LOO)')
        plt.title(
            "Expl. Var.:" + str(1 - nested_scores_test.mean() / -mean_squared_error(np.repeat(y.mean(), len(y)), y)) +
            "\nCorrelation: " + str(np.corrcoef(actual, predicted)[0, 1]))
        plt.show()
    else:
        all_models = [model]
        fig = []


    model.fit(X, y)  # fit to whole data

    return model, avg_model, all_models, clf, fig, error

def mytrain_ws_goupkfold(X, y,group, model, p_grid, nested=False, model_averaging=True):
    inner_cv = GroupKFold(10)
    outer_cv = GroupKFold(10)

    clf = GridSearchCV(estimator=model, param_grid=p_grid, cv=inner_cv,
                       scoring="neg_mean_squared_error", verbose=False,
                       return_train_score=False, n_jobs=-1)
    clf.fit(X, y,groups=group)

    print("**** Non-nested analysis ****")
    print("** Best hyperparameters: " + str(clf.best_params_))
    print("** Score on full data as training set:\t" + str(
        -mean_squared_error(y_pred=clf.best_estimator_.predict(X), y_true=y)))
    print("** Score on mean as model: " + str(-mean_squared_error(np.repeat(y.mean(), len(y)), y)))
    print("** Best Non-nested cross-validated score on test:\t" + str(clf.best_score_))

    model = clf.best_estimator_

    print("XXXXX Explained Variance: " + str(1 - clf.best_score_ / -mean_squared_error(np.repeat(y.mean(), len(y)), y)))

    avg_model = None
    all_models = []
    if nested:
        print("**** Nested analysis ****")

        best_params = []
        predicted = list()
        actual = list()
        nested_scores_train = np.zeros(outer_cv.get_n_splits(X,groups=group))
        nested_scores_test = np.zeros(outer_cv.get_n_splits(X,groups=group))
        #nested_scores_test2 = np.zeros(outer_cv.get_n_splits(X))
        i = 0
        avg = []
        # doing the crossval itewrations manually
        print("model\tinner_cv mean score\touter vc score")
        for train, test in outer_cv.split(X, y,groups=group):
            group_train = group[train]

            clf.fit(X[train], y[train],groups=group_train)

            # model avaraging
            # RES, mat, labels = get_full_coef(X[train], clf.best_estimator_, plot=False)
            # avg.append(RES)
            all_models.append(clf.best_estimator_)

            print(str(clf.best_params_) + " " + str(clf.best_score_) + " " + str(clf.score(X[test], y[test])))
            predicted.append(clf.predict(X[test]))
            actual.append(y[test])

            best_params.append(clf.best_params_)
            nested_scores_train[i] = clf.best_score_
            nested_scores_test[i] = clf.score(X[test], y[test])

            i = i + 1

        print("*** Score on mean as model:\t" + str(-mean_squared_error(np.repeat(y.mean(), len(y)), y)))
        print("** Mean score in the inner crossvaludation (inner_cv):\t" + str(nested_scores_train.mean()))
        print("** Mean Nested Crossvalidation Score (outer_cv):\t" + str(nested_scores_test.mean()))

        print("Explained Variance: " + str(
            1 - nested_scores_test.mean() / -mean_squared_error(np.repeat(y.mean(), len(y)), y)))
        print("Correlation: " + str(np.corrcoef(np.concatenate(actual,axis=0), np.concatenate(predicted,axis=0))[0, 1]))

        avg_model = np.mean(np.array(avg), axis=0)

        # plot the prediction of the outer cv
        fig, ax = plt.subplots()
        ax.scatter(np.concatenate(actual,axis=0), np.concatenate(predicted,axis=0), edgecolors=(0, 0, 0))
        # ax.plot([y.min(), y.max()],
        #           [y.min(), y.max()],
        #          'k--',
        #         lw=2)
        ax.set_xlabel('Measured behavior variable')
        ax.set_ylabel('Predicted (Nested LOO)')
        plt.title(
            "Expl. Var.:" + str(1 - nested_scores_test.mean() / -mean_squared_error(np.repeat(y.mean(), len(y)), y)) +
            "\nCorrelation: " + str(np.corrcoef(np.concatenate(actual,axis=0), np.concatenate(predicted,axis=0))[0, 1]))
        plt.show()
    else:
        all_models = [model]
        fig = []

    model.fit(X, y)  # fit to whole data

    return model, avg_model, all_models, clf, fig

def mytrain_ws_loog(X, y,group, model, p_grid, nested=False, model_averaging=True):
    inner_cv = LeaveOneGroupOut()
    outer_cv = LeaveOneGroupOut()

    clf = GridSearchCV(estimator=model, param_grid=p_grid, cv=inner_cv,
                       scoring="neg_mean_squared_error", verbose=False,
                       return_train_score=False, n_jobs=-1)
    clf.fit(X, y,groups=group)

    print("**** Non-nested analysis ****")
    print("** Best hyperparameters: " + str(clf.best_params_))
    print("** Score on full data as training set:\t" + str(
        -mean_squared_error(y_pred=clf.best_estimator_.predict(X), y_true=y)))
    print("** Score on mean as model: " + str(-mean_squared_error(np.repeat(y.mean(), len(y)), y)))
    print("** Best Non-nested cross-validated score on test:\t" + str(clf.best_score_))

    model = clf.best_estimator_

    print("XXXXX Explained Variance: " + str(1 - clf.best_score_ / -mean_squared_error(np.repeat(y.mean(), len(y)), y)))

    avg_model = None
    all_models = []
    if nested:
        print("**** Nested analysis ****")

        best_params = []
        predicted = list()
        actual = list()
        nested_scores_train = np.zeros(outer_cv.get_n_splits(X,groups=group))
        nested_scores_test = np.zeros(outer_cv.get_n_splits(X,groups=group))
        #nested_scores_test2 = np.zeros(outer_cv.get_n_splits(X))
        i = 0
        avg = []
        # doing the crossval itewrations manually
        print("model\tinner_cv mean score\touter vc score")
        for train, test in outer_cv.split(X, y,groups=group):
            group_train = group[train]

            clf.fit(X[train], y[train],groups=group_train)

            # model avaraging
            # RES, mat, labels = get_full_coef(X[train], clf.best_estimator_, plot=False)
            # avg.append(RES)
            all_models.append(clf.best_estimator_)

            print(str(clf.best_params_) + " " + str(clf.best_score_) + " " + str(clf.score(X[test], y[test])))
            predicted.append(clf.predict(X[test]))
            actual.append(y[test])

            best_params.append(clf.best_params_)
            nested_scores_train[i] = clf.best_score_
            nested_scores_test[i] = clf.score(X[test], y[test])

            i = i + 1

        print("*** Score on mean as model:\t" + str(-mean_squared_error(np.repeat(y.mean(), len(y)), y)))
        print("** Mean score in the inner crossvaludation (inner_cv):\t" + str(nested_scores_train.mean()))
        print("** Mean Nested Crossvalidation Score (outer_cv):\t" + str(nested_scores_test.mean()))
        explvar = 1 - nested_scores_test.mean() / -mean_squared_error(np.repeat(y.mean(), len(y)), y)
        print("Explained Variance: " + str(
            1 - nested_scores_test.mean() / -mean_squared_error(np.repeat(y.mean(), len(y)), y)))
        print("Correlation: " + str(np.corrcoef(np.concatenate(actual,axis=0), np.concatenate(predicted,axis=0))[0, 1]))

        avg_model = np.mean(np.array(avg), axis=0)

        # plot the prediction of the outer cv
        fig, ax = plt.subplots()
        ax.scatter(np.concatenate(actual,axis=0), np.concatenate(predicted,axis=0), edgecolors=(0, 0, 0))
        # ax.plot([y.min(), y.max()],
        #           [y.min(), y.max()],
        #          'k--',
        #         lw=2)
        ax.set_xlabel('Measured behavior variable')
        ax.set_ylabel('Predicted (Nested LOO)')
        plt.title(
            "Expl. Var.:" + str(1 - nested_scores_test.mean() / -mean_squared_error(np.repeat(y.mean(), len(y)), y)) +
            "\nCorrelation: " + str(np.corrcoef(np.concatenate(actual,axis=0), np.concatenate(predicted,axis=0))[0, 1]))
        plt.show()
    else:
        all_models = [model]
        fig = []
        predicted=[]
        actual=[]
        explvar = []

    model.fit(X, y)  # fit to whole data

    return model, avg_model, all_models, clf, fig, predicted, actual, explvar

def pred_stat(observed, predicted, robust=False,regside='two-sided'):

    # convert to np.array
    observed = np.array(observed)
    predicted = np.array(predicted)

    #EXCLUDE NA-s:
    predicted = predicted[~np.isnan(observed)]
    observed = observed[~np.isnan(observed)]

    if robust:
        # based on RLM documentation: first variable is the dependent variable and the second is the independent.
        # https://www.statsmodels.org/dev/generated/statsmodels.robust.robust_linear_model.RLM.html#statsmodels.robust.robust_linear_model.RLM
        # based on this article (https://www.sciencedirect.com/science/article/abs/pii/S0304380008002305)
        # the dependent variable (y-axis) should be the observed and the indepeendet(x-axis) should be the predicted.
        res = sm.RLM(observed, sm.add_constant(predicted)).fit()
        #res = sm.RLM(predicted, sm.add_constant(observed)).fit()
        p_value = res.pvalues[1]
        regline = res.fittedvalues
        residual = res.sresid

        # this is a pseudo r_squared, see: https://stackoverflow.com/questions/31655196/how-to-get-r-squared-for-robust-regression-rlm-in-statsmodels
        r_2 = sm.WLS(observed, sm.add_constant(predicted), weights=res.weights).fit().rsquared
        #r_2 = sm.WLS(predicted, sm.add_constant(observed), weights=res.weights).fit().rsquared

    else:
        slope, intercept, r_value, p_value, std_err = stats.linregress( observed,predicted,alternative=regside)
        regline = slope*observed+intercept
        r_2 = r_value**2
        residual = observed - regline

    return p_value, r_2, residual, regline

def plot_prediction(observed, predicted, outfile="", covar=[], robust=False, sd=True, text=""):
    color = "black"
    if len(covar):
        g = sns.jointplot(observed, predicted, scatter=False, color=color, kind="reg", robust=robust, x_ci="sd", )
        plt.scatter(observed, predicted,
                    c=covar, cmap=ListedColormap(sns.color_palette(["#5B5BFF","#D73E68"])))
    else:
        g = sns.jointplot(observed, predicted, kind="reg", color=color, robust=robust, x_ci="sd")
    #sns.regplot(observed, predicted, color="b", x_bins=10, x_ci=None)

    if sd:
        xlims=np.array(g.ax_joint.get_xlim())
        if robust:
            res = sm.RLM(predicted, sm.add_constant(observed)).fit()
            coefs = res.params
            residual = res.resid
        else:
            slope, intercept, r_value, p_value, std_err = stats.linregress(observed, predicted)
            coefs=[intercept, slope]
            regline = slope * observed + intercept
            residual = observed - regline

        S = np.sqrt(np.mean(residual**2))
        upper = coefs[1] * xlims + coefs[0] + S/2
        lower = coefs[1] * xlims + coefs[0] - S/2

        plt.plot(xlims, upper, ':', color=color, linewidth=1, alpha=0.3)
        plt.plot(xlims, lower, ':', color=color, linewidth=1, alpha=0.3)

    if text:
        # plt.text(np.min(observed) - (np.max(predicted)-np.min(predicted))/3,
        #          np.max(predicted) + (np.max(predicted)-np.min(predicted))/3,
        #          text, fontsize=10)
        #print('most akkor betoltott')
        plt.gcf().text(0.7,0.7,text, fontsize=10)

    if outfile:
        figure = plt.gcf()
        figure.savefig(outfile, bbox_inches='tight')
        plt.close(figure)
    else:
        plt.show()
    return

def evaluate_crossval_prediction(model, X, y, outfile="", cv=LeaveOneOut(), group=np.array([]), robust=False):
    if group.size==0:
        predicted = cross_val_predict(model, X, y, cv=cv)
    else:
        predicted = cross_val_predict(model, X, y, cv=cv,groups=group)

    p_value, r_2, residual, regline = pred_stat(y, predicted, robust=robust)

    expl_var = ( 1- (-mean_squared_error(y_pred=predicted, y_true=y)
                   /
                   -mean_squared_error(np.repeat(y.mean(), len(y)), y) ))*100

    print("R2=" + "{:.3f}".format(r_2) + "  R=" + "{:.3f}".format(np.sqrt(r_2)) +
           "   p=" + "{:.6f}".format(p_value) + "  Expl. Var.: " + "{:.1f}".format(expl_var) + "%" +
           "  Expl. Var.2: " + "{:.1f}".format(explained_variance_score(y_pred=predicted, y_true=y)*100) + "%" +
           "  MSE=" + "{:.3f}".format(mean_squared_error(y_pred=predicted, y_true=y)) +
           " RMSE=" + "{:.3f}".format(np.sqrt(mean_squared_error(y_pred=predicted, y_true=y))) +
           "  MAE=" + "{:.3f}".format(mean_absolute_error(y_pred=predicted, y_true=y)) +
           " MedAE=" + "{:.3f}".format(median_absolute_error(y_pred=predicted, y_true=y)) +
           "  R^2=" + "{:.3f}".format(r2_score(y_pred=predicted, y_true=y)))


    plot_prediction(y, predicted, outfile, robust=robust, sd=True,
                         text="$R2$=" + "{:.3f}".format(r_2) +
                              "  p=" + "{:.3f}".format(p_value) +
                              "  Expl. Var.: " + "{:.1f}".format(expl_var) + "%"
                         )
    return predicted

def excl_basedonbehavandtestpredofregmodel(behavtbl,
                                           features,
                                           exclvar,
                                           limitforexcl=4.1,
                                           directionofexcl="highpass",
                                           lowerval=0,
                                           predval='prediction'):
    redtbl, redfeature = removesubj_basedonotherbehavvalue(behav_tbl=behavtbl,
                                          features=features,
                                          exclusionvar=exclvar,
                                          limitforexcl=limitforexcl,
                                          directionofexcl=directionofexcl,
                                          lowervalue=lowerval)

    plot_prediction(redtbl['y_valpainlearn_acq'], redtbl[predval])
    p_value, r_2, residual, regline= pred_stat(redtbl['y_valpainlearn_acq'], redtbl[predval], robust=True)
    print('Explained variance: ' + str(r_2))
    print('p-value: ' + str(p_value))


def permutationbasedcorr(x,y):
    print('Observed pearson R: %.2f' % np.corrcoef(x,y)[1][0])
    p_value = permutation_test(x, y,
                               method='approximate',
                               func=lambda x, y: np.corrcoef(x,y)[1][0],
                               num_rounds=10000,
                               seed=0,
                               paired=True)
    print('P value: %.7f' % p_value)