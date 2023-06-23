from nilearn.connectome import ConnectivityMeasure
from sklearn.covariance import LedoitWolf, EmpiricalCovariance
import numpy as np
import pandas as pd
from neuroCombat import neuroCombat
import matplotlib.pyplot as plt
import seaborn as sns


def connectivity_matrix(timeseries,
                        roilabelid=[],
                        kind='partial correlation',
                        vect=True,
                        allroi=True,
                        covest=LedoitWolf()):
    '''

    Calculates the connectivity between regions based on their timeseries data.

    :param timeseries: Timeseries of ROIs provided as output of load_timeserie function (list of numpy arrays in which columns represent the ROI, and rows are the time dimension)
    :param roilabelid: if allroi set to False, this has to be specified. These are the indices of the ROIs (the column number in the dataframe) which one would like to keep.
    :param kind: the tpye of correlation between timeseries: parital correlation, full correlation, tangent
    :param vect: if TRUE the result will be returned as the lower triangle of the connectivity matrix, see also for connection names the labelvectorizin function
    :param allroi: if TRUE all ROIs' timeseries are used in the calculation of the connectivity matrix, no need for roilabelid
    :param covest: the estimator of covariance calculation. The deault is the LedoitWolf method which slightly different thatn the EmpiricalCOvariance. To compare the connectivity calculated in different softwares.
    :return:
    '''

    correlation_measure = ConnectivityMeasure(kind=kind, vectorize=vect, discard_diagonal=True)
    correlation_measure.cov_estimator = covest
    if allroi:
        correlation_matrix = correlation_measure.fit_transform(timeseries)
    else:

        for i in range(0,len(timeseries)):
            timeseries[i] = timeseries[i][:,roilabelid]

        correlation_matrix = correlation_measure.fit_transform(timeseries)
    return correlation_matrix, correlation_measure


def removeNAs_frombehav(yvar,
                        behavtable,
                        featurespace):

    X_predictor = featurespace.to_numpy()
    if isinstance(behavtable, pd.DataFrame):
        y_predictor = behavtable[yvar].to_numpy()
    elif isinstance(behavtable, pd.Series):
        print('nananana')
        y_predictor = behavtable.to_numpy()


    # if there are na values in the behavior param
    inidcesofnonna = ~np.isnan(y_predictor)
    y_predictor = y_predictor[inidcesofnonna]
    X_predictor= X_predictor[inidcesofnonna,:]
    #y_predictor = behavforcorr['medianFD'].values
    print('The shape of our predictor matrix:',X_predictor.shape)
    print('The shape of our outcome variable:', y_predictor.shape)
    print('The indices of participants who has missing values: ' +
          str(behavtable.loc[~inidcesofnonna,].index))

    return y_predictor, X_predictor, inidcesofnonna

def removesubj_basedonotherbehavvalue(behav_tbl,
                                      features,
                                      exclusionvar,
                                      limitforexcl,
                                      directionofexcl='highpass',
                                      lowervalue = -30):
    '''
    Exclude participants based on some behavior parameter limit. This script do the exclusion and returns the behavior and the feature space as well.
    The aim here is to validate that the model can predict learner's performance.
    :param behav_tbl:
    :param features:
    :param exclusionvar:
    :param limitforexcl:
    :param directionofexcl:
    :param lowervalue:
    :return:
    '''


    if directionofexcl=='highpass':
        reduced_tbl = behav_tbl.loc[behav_tbl[exclusionvar] > limitforexcl, ]
        reduced_features = features.loc[behav_tbl[exclusionvar] > limitforexcl, ]
    elif directionofexcl=='lowpass':
        reduced_tbl = behav_tbl.loc[behav_tbl[exclusionvar] < limitforexcl,]
        reduced_features = features.loc[behav_tbl[exclusionvar] < limitforexcl,]
    elif directionofexcl == 'bandpass':
        reduced_tbl = behav_tbl.loc[(behav_tbl[exclusionvar] < limitforexcl) & (behav_tbl[exclusionvar] > lowervalue),]
        reduced_features = features.loc[(behav_tbl[exclusionvar] < limitforexcl) & (behav_tbl[exclusionvar] > lowervalue),]

    #print(str(behav_tbl.shape[0] - reduced_tbl.shape[0]) + ' particpant were excluded.')
    # ax.scatter(othervars, preds_curr_above20, edgecolors=(0, 0, 0))
    # ax.plot([y.min(), y.max()],
    #           [y.min(), y.max()],
    #          'k--',
    #         lw=2)
    # ax.set_xlabel('Observed value')
    # ax.set_ylabel('Predicted')
    # ax.title('The relation of model prediction and observed variables in the current dataset')
    # myml.plot_prediction(reduced_tbl[measuredvar], reduced_tbl[predictedvar])
    #
    # print(reduced_tbl[[measuredvar,predictedvar]].corr())
    # myml.pred_stat(reduced_tbl[measuredvar], reduced_tbl[predictedvar])

    return reduced_tbl, reduced_features

def combat_harmonizetwofeaturespc(featurespace1,
                                  featurespace2,
                                  behav1,
                                  behav2,
                                  modelfeauters=[],
                                  othercovars=[]):
    '''

    rows are subjects, columns are features. It should be a pandas df.
    :param featurespace2: rows are subjects, columns are features. It should be a pandas df.
    :param behav1:
    :param behav2:
    :param modelfeauters: if it is specified, the function will plot the original and the combat transformed data on those features with a boxplot.
    :param othercovars:
    :return:
    '''
    featurespace1_t = featurespace1.to_numpy().transpose()
    featurespace2_t = featurespace2.to_numpy().transpose()
    all_np = np.hstack((featurespace1_t, featurespace2_t))
    if othercovars:
        covars = {'batch': [1] * featurespace1_t.shape[1] + [2] * featurespace2_t.shape[1],
                  'gender': list(behav1[othercovars[0][0]]) + list(behav2[othercovars[1][0]]),
                  'age':list(behav1[othercovars[0][1]]) + list(behav2[othercovars[1][1]])
                  }  # ,'gender':[1,2,1,2,1,2,1,2,1,2]
    else:
        covars = {'batch': [1] * featurespace1_t.shape[1] + [2] * featurespace2_t.shape[1]}#,
              #'gender': list(orig_behav['sex']) + list(curr_behav['sex'])}  # ,'gender':[1,2,1,2,1,2,1,2,1,2]
    covars = pd.DataFrame(covars)
    categorical_cols = []  # 'gender'
    batch_col = 'batch'
    data_combat = neuroCombat(dat=all_np,
                              covars=covars,
                              batch_col=batch_col,
                              categorical_cols=categorical_cols)["data"]

    # %%
    featurespace1_aftcombat = data_combat[:, :featurespace1_t.shape[1]].transpose()
    featurespace2_aftcombat = data_combat[:, -featurespace2_t.shape[1]:].transpose()

    if modelfeauters:

        fig, ax = plt.subplots(2, 2)
        #original data for feature 1
        sns.boxplot(data=pd.concat([pd.DataFrame(featurespace1_aftcombat[:, modelfeauters[0]], columns=["combat"]),
                                    pd.DataFrame(featurespace1.iloc[:, modelfeauters[0]].values, columns=['original'])],
                                   axis=1),ax=ax[0, 0]).set_title("Feature1 in sample1")
        # dataset2 for feature 1
        sns.boxplot(data=pd.concat([pd.DataFrame(featurespace2_aftcombat[:, modelfeauters[0]], columns=["combat"]),
                                    pd.DataFrame(featurespace2.iloc[:, modelfeauters[0]].values, columns=['original'])],
                                   axis=1),ax=ax[1, 0]).set_title("Feature1 in sample2")
        # original data for feature 2
        sns.boxplot(data=pd.concat([pd.DataFrame(featurespace1_aftcombat[:, modelfeauters[1]], columns=["combat"]),
                                    pd.DataFrame(featurespace1.iloc[:, modelfeauters[1]].values, columns=['original'])],
                                   axis=1),ax=ax[0, 1]).set_title("Feature2 in sample1")
        # dataset2 for feature 2
        sns.boxplot(data=pd.concat([pd.DataFrame(featurespace2_aftcombat[:, modelfeauters[1]], columns=["combat"]),
                                    pd.DataFrame(featurespace2.iloc[:, modelfeauters[1]].values, columns=['original'])],
                                   axis=1),ax=ax[1, 1]).set_title("Feature2 in sample2")

        plt.tight_layout()
        plt.show()

    return featurespace1_aftcombat, featurespace2_aftcombat


def connectivitylabelname(model,
                          listoflabelpairs,
                          labeldir='../data_in/MIST_122.csv'):
    labelinf = pd.read_csv(labeldir,sep=';')
    parentinfo = pd.read_csv('../data_in/MIST_PARCEL_ORDER.csv', sep=',')
    parentnames = pd.read_csv('../data_in/MIST_7.csv', sep=';')
    zip_dict_labelnames=zip(list(labelinf['label']),list(labelinf['name']))
    dict_labelnames=dict(zip_dict_labelnames)
    # # add global signal by hand
    dict_labelnames['GlobSig'] = 'Global Signal'
    # #define the connections in our model, this include the short name of the connection pair and the strenght in the model
    connections = pd.DataFrame(list((listoflabelpairs[i],model[1].coef_[idx]) for idx,i in enumerate(model[0].get_support(indices=True))),
                           columns=['conn','strenght'])
    print('the total number of predictive connections:'+str(len(connections)))


    # this part only
    dict_roivalues = dict(zip(list(labelinf['roi']),list(labelinf['label'])))
    #add global signal by hand
    dict_roivalues[123]='GlobSig'
    # we have to play around a little as the short names of certain ROIs are in the list two times because they are neighbouring areas in the MIST122 atlas.
    # (e.g. the superior parietal lobule is the 1st and 76th ROI as well, but that two region anatomical coordinates are differ a bit(see the MIST122 table for detail)
    for row, ind in enumerate(connections['conn']):


        if ind[0]=='GlobSig':
            connections.loc[row,'region1']='Globalsignal'
        elif ind[0][-2:]=='.1':
            print(ind)
            properconnname = ind[0][:-2]

            connections.loc[row,'region1'] = dict_labelnames[properconnname]
            connections.loc[row,'roival_region1'] = [list(dict_roivalues.keys())[i] for i, n in enumerate(list(dict_roivalues.values())) if n == ind[0][:-2]][1]
            # connections.loc[row,'parent_region1'] = labelinf.loc[labelinf['label']==properconnname,"parent"].values[1]
            # connections.loc[row,'neighbour_region1'] = labelinf.loc[labelinf['label']==properconnname,"neighbour"].values[1]
            connections.loc[row,'mainparent_region1'] = np.unique(parentinfo.loc[parentinfo['s122']==connections.loc[row,'roival_region1'],'s7'].values)
            connections.loc[row,'mainparentname_region1'] = parentnames.loc[parentnames['roi']==connections.loc[row,'mainparent_region1'],'name'].values
        else :
            connections.loc[row,'region1'] = dict_labelnames[ind[0]]
            connections.loc[row,'roival_region1'] = [list(dict_roivalues.keys())[i] for i, n in enumerate(list(dict_roivalues.values())) if n == ind[0]][0]
            # connections.loc[row,'parent_region1'] = labelinf.loc[labelinf['label']==ind[0],"parent"].values[0]
            # connections.loc[row,'neighbour_region1'] = labelinf.loc[labelinf['label']==ind[0],"neighbour"].values[0]
            connections.loc[row,'mainparent_region1'] = np.unique(parentinfo.loc[parentinfo['s122']==connections.loc[row,'roival_region1'],'s7'].values)
            connections.loc[row,'mainparentname_region1'] = parentnames.loc[parentnames['roi']==connections.loc[row,'mainparent_region1'],'name'].values
        if ind[1]=='GlobSig':
            connections.loc[row,'region2'] = 'Globalsignal'
        elif ind[1][-2:]=='.1':
            print(ind)
            properconnname= ind[1][:-2]
            connections.loc[row,'region2'] = dict_labelnames[properconnname]
            connections.loc[row,'roival_region2'] = [list(dict_roivalues.keys())[i] for i, n in enumerate(list(dict_roivalues.values())) if n == ind[1][:-2]][1]
            # connections.loc[row,'parent_region2'] = labelinf.loc[labelinf['label']==properconnname,"parent"].values[1]
            # connections.loc[row,'neighbour_region2'] = labelinf.loc[labelinf['label']==properconnname,"neighbour"].values[1]
            connections.loc[row,'mainparent_region2'] = np.unique(parentinfo.loc[parentinfo['s122']==connections.loc[row,'roival_region2'],'s7'].values)
            connections.loc[row,'mainparentname_region2'] = parentnames.loc[parentnames['roi']==connections.loc[row,'mainparent_region2'],'name'].values
        else:
            connections.loc[row,'region2'] = dict_labelnames[ind[1]]
            connections.loc[row,'roival_region2'] = [list(dict_roivalues.keys())[i] for i, n in enumerate(list(dict_roivalues.values())) if n == ind[1]][0]
            # connections.loc[row,'parent_region2'] = labelinf.loc[labelinf['label']==ind[1],"parent"].values
            # connections.loc[row,'neighbour_region2'] = labelinf.loc[labelinf['label']==ind[1],"neighbour"].values
            connections.loc[row,'mainparent_region2'] = np.unique(parentinfo.loc[parentinfo['s122']==connections.loc[row,'roival_region2'],'s7'].values)
            connections.loc[row,'mainparentname_region2'] = parentnames.loc[parentnames['roi']==connections.loc[row,'mainparent_region2'],'name'].values
    ordered_connections_nwmdl = connections.reindex(connections.strenght.abs().sort_values(ascending=False).index)
    return ordered_connections_nwmdl