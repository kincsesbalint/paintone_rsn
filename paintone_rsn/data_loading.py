import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_subjectiid(inputtxt):
    '''

    Load the sujectID file from the RPN output to easily pair the indices(RPNID) with the appropriate participants' id (subjid):
    create a dictionary in which the keys are the subject IDs defined in the study (study ID) and the values are the ids in the RPN(RPNID)

    Balint Kincses
    Balint.Kincses@uk-essen.de
    2022

    :param inputtxt:
    :return: dictionary (key-study id(eg:01 in sub-01) and values-RPN id, starts with _0 ), input to
    '''

    subjID_idx = pd.read_csv(inputtxt,header=None)

    dict_subjID_idx={}
    for idx, my_string in enumerate(subjID_idx[0].values):
        dict_subjID_idx[np.int((my_string.split("func/sub-",1)[1].split('_task',1)[0]))] = idx

    return dict_subjID_idx

def exclusion(motionfile,
              dict_subjID_idx,
              excl_subj=[],
              FD_limit=0.15,
              percscrublimit = 25
              ):
    '''

    Collect all particpants subjID who we would like to exclude and include based on the info from the measurement (eg: fell asleep, incidental brain abnormality finding) and the motion info.


    Balint Kincses
    Balint.Kincses@uk-essen.de
    2022

    :param motionfile: The rpn output motion_summary.csv, which contains info about the motion during the fMRI
    :param dict_subjID_idx: the dictionary which contains the IDs of participants (the output of the load_subjectid function)
    :param excl_subj: a list of numbers of participants which has to be exlcuded beforehand (eg: incidental finding on the MRI, fell asleep during experiment...)
    :param FD_limit: the limit of the mean functional displacement, default is 0.15
    :param percscrublimit: the limit of the precent of scrubbed volumes from the fMRI, default is 25
    :return: dictionaries of the included and excluded subjects' ID
    '''


    dict_subjID_idx_incl = dict()
    dict_subjID_idx_excl = dict()

    excl_indices = list(dict_subjID_idx[k] for k in excl_subj if k in dict_subjID_idx.keys())
    if os.path.isfile(motionfile):
        motioninf = pd.read_csv(motionfile)
        orderidx_incl = motioninf.loc[(motioninf['meanFD'] < FD_limit) &
                                  (motioninf['perc_scrubbed'] < percscrublimit) &
                                    #(motioninf['maxFD'] < 1) &
                                  (~motioninf['Unnamed: 0'].isin(excl_indices)) ,:].index.values
    else:
        print('The specified motion_summary file does NOT exist, we include all the participants!')
        orderidx_incl = dict_subjID_idx.values()

    for key, value in dict_subjID_idx.items():
        if value in orderidx_incl:
            dict_subjID_idx_incl[key]=value
        else:
            dict_subjID_idx_excl[key]=value

    print('These are the subjects subjectID who were excluded:\t' + str(dict_subjID_idx_excl.keys()))
    print('These are the subjects ordererID who were excluded:\t' + str(dict_subjID_idx_excl.values()))
    print('These are the subjects subjectID who were included:\t' + str(dict_subjID_idx_incl.keys()))
    print('These are the subjects ordererID who were included:\t' + str(dict_subjID_idx_incl.values()))
    print('In total we work with\t'+ str(len(orderidx_incl)) + '\t participants.\n Check if all the aprticipanst have behavior data!!!!')
    nexclbasedonmotion = len(dict_subjID_idx) - len(excl_indices) - len(orderidx_incl)
    print('Number of participant excluded based in the motion parameters(FD>' +str(FD_limit)+ ',scrubbed percent>' +str(percscrublimit)+'):' +str(nexclbasedonmotion))
    if os.path.isfile(motionfile):
        for key, value in dict_subjID_idx.items():
            if value in motioninf.drop(orderidx_incl)['Unnamed: 0']:
                print('this is their subjID \t'+str(key)+'\tand this is thier orderID:\t'+str(value))
    else:
        print('No motion info available!')

    print('These are the participants subjID whose excluded beforehand:\t' + str(excl_subj))

    return dict_subjID_idx_incl, dict_subjID_idx_excl, motioninf

def timeseriespath(ts_directory):
    '''
    Collect all the subjects' timeseries file path to a list.
    :param ts_directory: the path to the folder wchic contains all the subjects' timeseries data
    :return: a list of participants' timeseries data path
    '''

    ts_filelist = os.listdir(ts_directory)
    ts_filelist.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    ts_fullpathfs = []
    for ff in ts_filelist:
        ts_fullpathfs.append(os.path.join(ts_directory, ff))
    return ts_fullpathfs

def load_timeseries(ts_directory,
                    subjid_kept,
                    keepgs=True,
                    standardise=True,
                    fddir=[],
                    fdlimit=0.15,
                    numofexlcvolsatthebeg=0,
                    subjchr='subj_',
                    origdatasets=True
                    ):
    '''
    This functions aims to load the particiapnts' timeseries. It prints out the path of the loaded participants for double checking.

    :param ts_files: the path to the timeseries files
    :param subjid_kept: the index of participants (RPNID-based on the RPN output, that is the order of the participants and it starts with 0) which are included in the analysis, see returns of exclusion function
    :param keepgs: if TRUE keep the global signal in the loaded dataframe
    :param standardise: if TRUE a standard sclaer is used in every ROI's timeseries (default is TRUE)
    :param fddir: if it is specified, volumes with higher fd than the fdlimit are scrubbed
    :param fdlimit: the framewise displacement limit what we base our scrubbing on
    :param numofexlcvolsatthebeg: exclude the first few volumes to reach equilibrium state(however, I am not sure if the data what we have at hand already delete thos vols,so we do not use it as a degault)
    :param subjchr: the name of the regional timeseries file of the ceratin study
    :param origdatasets: I manually added a 0 to the regional timeseries files' name for aprticipant with one digit in the original dataset, this was not done to other datasets
    :return: the list of timeseries of the included subjects, the ROI labels, and the loadedsubjectpath(this is only for double check if the of the ROI as
    '''

    #ts_fullpathfs = timeseriespath(ts_directory)

    timeseries = []
    loadedsubjpath = []
    # if len(fddir):
    #     filelist = os.listdir(fddir[0])
    #     filelist.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    for i in subjid_kept:
        #print('-'*65)
        if origdatasets:
            if len(str(i)) == 1:
                subjfile = ts_directory + subjchr + '0' + str(i) + '_ts.tsv'
            elif len(str(i)) == 2:
                subjfile = ts_directory + subjchr + str(i) + '_ts.tsv'
        else:
            subjfile = ts_directory + subjchr + str(i) + '.tsv'
        #print(subjfile)
        if keepgs:
            ts = pd.read_csv(subjfile, sep="\t")  # keep the global signal
            labels = pd.read_csv(subjfile, sep="\t").columns
        else:
            ts = pd.read_csv(subjfile, sep="\t").drop('GlobSig', 1).values  # delete the global signal
            labels = pd.read_csv(subjfile, sep="\t").columns[1:]

        loadedsubjpath.append(subjfile)
        ts=ts[numofexlcvolsatthebeg:]
        if len(fddir):

            # sort them in order, but we will work with the file name(to get the rpn-id from the filename we hardcoded some stuff here),
            # We use the same pipeline, so it should work.
            if origdatasets:
                subjfddir = fddir[0] + '/_calculate_FD_Power'+ str(i) + "/fd_power_2012.txt"
            else:
                subjfddir =fddir[0] +  '/_calculate_FD_Power'+ str(i) + ".txt"

            #print('the fddir is specified by the user:\n' + subjfddir)
            fd = pd.read_csv(subjfddir).to_numpy()
            scrubbingidx = fd < fdlimit
            scrubbingidx = np.insert(np.reshape(scrubbingidx, -1), 0, True, axis=0)
            scrubbingidx = scrubbingidx[numofexlcvolsatthebeg:]
            # print('this is the scrubbingidx what we loaded', scrubbingidx)
            #print('this is the length of the scrubbing boolec:',len(scrubbingidx))
            #print('the ts:'+len(ts))
            ts = ts[scrubbingidx]
            # print('-'*65)
            # print((1 - len(ts)/192)*100)
            # print('-' * 65)
        else:
            print('No scrubbing has happened')


        # standardise timeseries
        if standardise:
            ts = StandardScaler().fit_transform(ts)
        else:
            ts = np.array(ts)

        timeseries.append(ts)
    #labels = pd.read_csv(subjfile, sep="\t").columns

    vectorzedlabelpairs,  listoflabelpairs = labelvectorizin(labels)

    return timeseries, labels, loadedsubjpath, vectorzedlabelpairs, listoflabelpairs

def labelvectorizin(labels):

    '''

    The issue is that the order in the vectorized form of the connectivity matrix is not known.
    With this custom built modul we solve this:
    one has to define the number of predictors to determine the vectorized format of the connectivity matricies

    :param labels: This is the lables of the ROIs what we work with.
    :return: Vectorized form of the labels in two forms:
    '''

    numberofpredictors=np.int(((len(labels)*len(labels))-len(labels))/2)
    pairofrois=np.zeros(shape=(numberofpredictors,2),dtype=int)
    i=0
    for ff in range(0,len(labels)):
        for gg in range(0,ff):
            pairofrois[i]= [gg,ff]
            i=i+1

    # we tested it and it works. That is, the place in the vectorized matrix and the place in the variable of pairofrois
    # is the same. while the former is the exact correlation value, the latter is the labels of ROIs
    # so call as roilabels[pairofrois[z]]
    vectorzedlabels = []
    listoflabelpairs = []
    for ii in pairofrois:
        vectorzedlabels.append(labels[ii][0]+'+'+labels[ii][1])
        listoflabelpairs.append([labels[ii][0],labels[ii][1]])
    # checked again and it works properly. So in the vectorized format and the matrix format the certain cells match.
    print('The number of ROIs:' + str(len(listoflabelpairs)))

    return vectorzedlabels, listoflabelpairs


def load_behaviordata(includedsubj,
                      pathtofile_val='/home/balint/Documents/Katistudy/C_statistics/alldata.csv',
                      collist_val=['i', 'hpt', 'CS', 'valence', 'timepoint_phase', 'timepoint_phase2', 'phase',
                                   'anxiety_pain', 'anxiety_tone', 'diff_anxiety'],
                      pathtofile_cont='/home/balint/Documents/Katistudy/C_statistics/data/df_allsubs.csv',
                      collist_cont=['i', 'sex', 'age', 'acqkon_CS_tone_transf', 'acqkon_CS_pain_transf',
                                    'acqkon_CS_minus_transf', 'extkon_CS_tone_transf',
                                    'extkon_CS_pain_transf', 'extkon_CS_minus_transf']):
    """
    Load the valence rating table and return pandas data frame with th included subjeccts.
    :param includedsubj: list of subject indices (serial number)
    :param pathtofile_val: tha path to the valence behavior file
    :param collist_val: the columns' name which contain the valence data
    :param pathtofile_cont: the path to the contingency behavior file
    :param collist_cont: the columns' name which contain the contingency data
    :return: two pandas data frame, one include all the interestingbehavior variables and the other includes the raw valence ratings
    """

    alldata = pd.read_csv(pathtofile_val, usecols=collist_val)
    # define the python based index (starts with 0)
    #alldata.loc[:, 'id_pos'] = alldata.loc[:, 'i'] - 1
    # get a table which include all the participants derived behavior data
    # alldata_behav = alldata.pivot_table(index=['sub'],
    #                                     values=[])
    # alldata_behav_excl = alldata_behav.loc[includedsubj, :]

    # Reorganize the data in a wide format (long to wide) to get valence rating in an organized way
    valencedata_wide = alldata.pivot_table(index=['sub'], columns=['phase', 'CS', 'trialnummer'],
                                           values='valence')
    # get the multilevel indexing names in the columns:
    valencedata_wide_excl = valencedata_wide.loc[includedsubj, :]

    # create new behavior variables, and save them in the alldata_behav df. examples:
    # pain learning based on my defintion in acquisition phase
    valencedata_wide_excl.loc[:, 'y_valpainlearn_acq'] = \
        (valencedata_wide_excl.loc[:, ('acq', 'pain', 5)] - valencedata_wide_excl.loc[:, ('hab', 'pain', 1)]) - \
        (valencedata_wide_excl.loc[:, ('acq', 'minus', 5)] - valencedata_wide_excl.loc[:, ('hab', 'minus', 1)])
    # tone learning based on my definition in acquisition phase
    valencedata_wide_excl.loc[:, 'y_valtonelearn_acq'] = \
        (valencedata_wide_excl.loc[:, ('acq', 'tone', 5)] - valencedata_wide_excl.loc[:, ('hab', 'tone', 1)]) - \
        (valencedata_wide_excl.loc[:, ('acq', 'minus', 5)] - valencedata_wide_excl.loc[:, ('hab', 'minus', 1)])
    # pain learning during extinction
    # valencedata_wide_excl.loc[:, 'y_valpainlearn_ext'] = \
    #     (valencedata_wide_excl.loc[:, ('ext', 'pain', 3)] - valencedata_wide_excl.loc[:, ('ext', 'pain', 1)]) - \
    #     (valencedata_wide_excl.loc[:, ('ext', 'minus', 3)] - valencedata_wide_excl.loc[:, ('ext', 'minus', 1)])
    # # tone learning during extinction
    # valencedata_wide_excl.loc[:, 'y_valtonelearn_ext'] = \
    #     (valencedata_wide_excl.loc[:, ('ext', 'tone', 3)] - valencedata_wide_excl.loc[:, ('ext', 'tone', 1)]) - \
    #     (valencedata_wide_excl.loc[:, ('ext', 'minus', 3)] - valencedata_wide_excl.loc[:, ('ext', 'minus', 1)])

    # fear learning, solely include the CS pain valence rating
    # valencedata_wide_excl.loc[:, 'y_valpainlearn_onlypain_acq'] = \
    #     (valencedata_wide_excl.loc[:, ('acq', 'pain', 4)] - valencedata_wide_excl.loc[:, ('hab', 'pain', 0)])
    # # safety learning, solely include the CS neutral valence ratings
    # valencedata_wide_excl.loc[:, 'y_safetylearning_acq'] = \
    #     (valencedata_wide_excl.loc[:, ('acq', 'minus', 4)] - valencedata_wide_excl.loc[:, ('hab', 'minus', 0)])
    # # contingency ratings for the different stimuli
    # # CSpain cont
    # alldata_behav_excl.loc[:, 'y_paincont_acq'] = \
    #     contingencyrat_excl.loc[:, 'acqkon_CS_pain_transf']
    # # CStone cont
    # alldata_behav_excl.loc[:, 'y_tonecont_acq'] = \
    #     contingencyrat_excl.loc[:, 'acqkon_CS_tone_transf']
    # # CS minus cont
    # alldata_behav_excl.loc[:, 'y_safetycont_acq'] = \
    #     contingencyrat_excl.loc[:, 'acqkon_CS_minus_transf']

    return valencedata_wide_excl

#def loadframewisedisplacement():
def samplecalibration(sample1,
                      sample2,
                      observedvar='y_valpainlearn_acq',
                      predictedvar='prediction_scr'):
    sample1.loc[:,'study']='s1'
    sample2.loc[:,'study']='s2'
    #dataset=pd.concat([sample2,sample1], axis=0)[[observedvar,predictedvar,'study']]
    s1_delta=sample1[observedvar].to_numpy().mean()-sample1[predictedvar].to_numpy().mean()
    s2_delta=sample2[observedvar].to_numpy().mean()-sample2[predictedvar].to_numpy().mean()
    sample1.loc[:,predictedvar+'_calib']=sample1[predictedvar]+s1_delta
    sample2.loc[:,predictedvar+'_calib']=sample2[predictedvar]+s2_delta
    concat_behav=pd.concat([sample1,sample2], axis=0)[[observedvar,predictedvar,predictedvar+'_calib','study']]

    return concat_behav
