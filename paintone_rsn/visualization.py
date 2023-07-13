import nibabel as nb
import numpy as np
import pandas as pd
from nilearn.plotting import plot_roi
from nilearn.image import threshold_img, math_img, new_img_like
from matplotlib import cm


def interestingrois(model,pairoflabels,mistatlaslabel='../data_in/MIST_122.csv',doubledrois=['L_IPlob']):
    labelinf = pd.read_csv(mistatlaslabel, sep=';')
    for roiswith1inthename in doubledrois:
        doubledidx = [ i for i, n in enumerate(labelinf['label']) if n == roiswith1inthename][1]
        print(labelinf.loc[doubledidx,'label'])
        labelinf.loc[doubledidx,'label']=roiswith1inthename+'.1'
    print(labelinf.loc[doubledidx,'label'])
    print('These are the hyperparameters of our model (fulldata fit):' + str(model[0].get_params()))
    roiidx = model[0].get_support()
    interestingroipairs = np.array(pairoflabels)[roiidx]

    allroiinfo = interestingroipairs.tolist()
    for idx, j in enumerate(allroiinfo):
        j.append(model[1].coef_[idx])
        j.append([labelinf[labelinf['label'].isin([j[0]])][['name']].values,
                  labelinf[labelinf['label'].isin([j[1]])][['name']].values])
        j.append([labelinf[labelinf['label'].isin([j[0]])][['roi']].values,
                  labelinf[labelinf['label'].isin([j[1]])][['roi']].values])

    print('These are the weights of the most important connections which are between the two regions listed after the value:')
    [print(hami[0:3]) for hami in allroiinfo]
    print('-' * 65)
    print('The indices of interesting ROI pairs:')
    [print(hami[4]) for hami in allroiinfo]
    print('-' * 65)
    print('The list of interesting ROIs connections with full name:')
    [print(hami[3]) for hami in allroiinfo]
    return allroiinfo

def plotrois(allinfo,outfilepath=[],atlas='../data_in/MIST_122.nii.gz'):
    mistatlas = nb.load(atlas)
    for idd, roiinfo in enumerate(allinfo):

        for k, roivalues in enumerate(roiinfo[4]):

            tmpimg1 = math_img('img == %s' % roivalues,img=mistatlas)


        #tmpimg2 = math_img('img == %s' % j[4][1], img=mistatlas)
        #result_img = math_img("img1 + img2",
        #                  img1=tmpimg1, img2=tmpimg2)
        #filename = outfilepath+ 'conn' +j[0]+j[1]+ '.png'
            if roiinfo[2] < 0:
                col = cm.get_cmap('Blues_r')
            elif roiinfo[2] >=0:
                col = cm.get_cmap('RdBu')
            if outfilepath:
                filename = outfilepath + str(idd) + roiinfo[k]+'.png'
                plot_roi(tmpimg1, output_file=filename,cmap=col,draw_cross=False)
            else:
                plot_roi(tmpimg1, cmap=col, draw_cross=False)

def modelmistatlasnaming(finalemodel,
                    listoflabelpairs):
    '''
    This function aims to return all the information of our model in a tabular format.
    :param finalemodel: the predictive model(which contains a feature selection and a ridge regression step)
    :param listoflabelpairs: the names of the feature space in which the model was trained. It is a list of the short names of label pairs
    :return: a table contains the predictive connections, their strength, the long names of the regions, the name of their "main" region in the 7network resolution.
    '''

    labelinf = pd.read_csv('../data_in/MIST_122.csv', sep=';')
    parentinfo = pd.read_csv('../data_in/MIST_PARCEL_ORDER.csv', sep=',')
    parentnames = pd.read_csv('../data_in/MIST_7.csv', sep=';')
    # print(labelinf)
    # get the most important connections from our model

    print('This is the main model. The connections are listed in strength order.')

    zip_dict_labelnames = zip(list(labelinf['label']), list(labelinf['name']))
    dict_labelnames = dict(zip_dict_labelnames)
    # add global signal by hand
    dict_labelnames['GlobSig'] = 'Global Signal'
    #define the connections in our model, this include the short name of the connection pair and the strenght in the model
    connections = pd.DataFrame(list((listoflabelpairs[i], finalemodel[1].coef_[idx]) for idx, i in
                                    enumerate(finalemodel[0].get_support(indices=True))),
                               columns=['conn', 'strenght'])

    print('the total number of predictive connections:' + str(len(connections)))


    # this part only
    dict_roivalues = dict(zip(list(labelinf['roi']), list(labelinf['label'])))
    # add global signal by hand
    dict_roivalues[123] = 'GlobSig'
    # we have to play around a little as the short names of certain ROIs are in the list two times because they are neighbouring areas in the MIST122 atlas.
    # (e.g. the superior parietal lobule is the 1st and 76th ROI as well, but that two region anatomical coordinates are differ a bit(see the MIST122 table for detail)
    for row, ind in enumerate(connections['conn']):

        if ind[0] == 'GlobSig':
            connections.loc[row, 'region1'] = 'Globalsignal'
        elif ind[0][-2:] == '.1':
            print(ind)
            properconnname = ind[0][:-2]

            connections.loc[row, 'region1'] = dict_labelnames[properconnname]
            connections.loc[row, 'roival_region1'] = \
            [list(dict_roivalues.keys())[i] for i, n in enumerate(list(dict_roivalues.values())) if n == ind[0][:-2]][1]
            connections.loc[row, 'parent_region1'] = labelinf.loc[labelinf['label'] == properconnname, "parent"].values[
                1]
            connections.loc[row, 'neighbour_region1'] = \
            labelinf.loc[labelinf['label'] == properconnname, "neighbour"].values[1]
            connections.loc[row, 'mainparent_region1'] = np.unique(
                parentinfo.loc[parentinfo['s122'] == connections.loc[row, 'roival_region1'], 's7'].values)
            connections.loc[row, 'mainparentname_region1'] = parentnames.loc[
                parentnames['roi'] == connections.loc[row, 'mainparent_region1'], 'name'].values
        else:
            connections.loc[row, 'region1'] = dict_labelnames[ind[0]]
            connections.loc[row, 'roival_region1'] = \
            [list(dict_roivalues.keys())[i] for i, n in enumerate(list(dict_roivalues.values())) if n == ind[0]][0]
            connections.loc[row, 'parent_region1'] = labelinf.loc[labelinf['label'] == ind[0], "parent"].values[0]
            connections.loc[row, 'neighbour_region1'] = labelinf.loc[labelinf['label'] == ind[0], "neighbour"].values[0]
            connections.loc[row, 'mainparent_region1'] = np.unique(
                parentinfo.loc[parentinfo['s122'] == connections.loc[row, 'roival_region1'], 's7'].values)
            connections.loc[row, 'mainparentname_region1'] = parentnames.loc[
                parentnames['roi'] == connections.loc[row, 'mainparent_region1'], 'name'].values
        if ind[1] == 'GlobSig':
            connections.loc[row, 'region2'] = 'Globalsignal'
        elif ind[1][-2:] == '.1':
            print(ind)
            properconnname = ind[1][:-2]
            connections.loc[row, 'region2'] = dict_labelnames[properconnname]
            connections.loc[row, 'roival_region2'] = \
            [list(dict_roivalues.keys())[i] for i, n in enumerate(list(dict_roivalues.values())) if n == ind[1][:-2]][1]
            connections.loc[row, 'parent_region2'] = labelinf.loc[labelinf['label'] == properconnname, "parent"].values[
                1]
            connections.loc[row, 'neighbour_region2'] = \
            labelinf.loc[labelinf['label'] == properconnname, "neighbour"].values[1]
            connections.loc[row, 'mainparent_region2'] = np.unique(
                parentinfo.loc[parentinfo['s122'] == connections.loc[row, 'roival_region2'], 's7'].values)
            connections.loc[row, 'mainparentname_region2'] = parentnames.loc[
                parentnames['roi'] == connections.loc[row, 'mainparent_region2'], 'name'].values
        else:

            connections.loc[row, 'region2'] = dict_labelnames[ind[1]]
            connections.loc[row, 'roival_region2'] = \
            [list(dict_roivalues.keys())[i] for i, n in enumerate(list(dict_roivalues.values())) if n == ind[1]][0]
            connections.loc[row, 'parent_region2'] = labelinf.loc[labelinf['label'] == ind[1], "parent"].values
            connections.loc[row, 'neighbour_region2'] = labelinf.loc[labelinf['label'] == ind[1], "neighbour"].values
            connections.loc[row, 'mainparent_region2'] = np.unique(
                parentinfo.loc[parentinfo['s122'] == connections.loc[row, 'roival_region2'], 's7'].values)
            connections.loc[row, 'mainparentname_region2'] = parentnames.loc[
                parentnames['roi'] == connections.loc[row, 'mainparent_region2'], 'name'].values
    ordered_connections_nwmdl = connections.reindex(connections.strenght.abs().sort_values(ascending=False).index)

    return ordered_connections_nwmdl
def parentconnectionname(labels):
    '''
    This function returns a table with the information about which region in the 122 resolution MIST atlas is in which 7 great functional network (MIST_7)
    :param labels: the names of the timeseries labels, derived from the RPN-output timeseries
    :return: a table which contains the roi short name, long name, value,  mainparent value, main parent name
    '''
    roinames = pd.DataFrame(labels,columns=['rois'])

    labelinf = pd.read_csv('../data_in/MIST_122.csv', sep=';')
    parentinfo = pd.read_csv('../data_in/MIST_PARCEL_ORDER.csv', sep=',')
    parentnames = pd.read_csv('../data_in/MIST_7.csv', sep=';')

    zip_dict_labelnames = zip(list(labelinf['label']), list(labelinf['name']))
    dict_labelnames = dict(zip_dict_labelnames)

    dict_roivalues = dict(zip(list(labelinf['roi']), list(labelinf['label'])))
    # add global signal by hand
    dict_roivalues[123] = 'GlobSig'

    for row, ind in enumerate(roinames['rois']):

        if ind == 'GlobSig':
            roinames.loc[row, 'region_longname'] = 'Globalsignal'
            roinames.loc[row, 'roival_region'] = 123
            roinames.loc[row, 'mainparent_region'] = 0
            roinames.loc[row, 'mainparentname_region'] = 'GS'
        elif ind[-2:] == '.1':
            #print(ind)
            propername = ind[:-2]
            roinames.loc[row, 'region_longname'] = dict_labelnames[propername]
            roinames.loc[row, 'roival_region'] = \
            [list(dict_roivalues.keys())[i] for i, n in enumerate(list(dict_roivalues.values())) if n == ind[:-2]][1]
            # roinames.loc[row, 'parent_region1'] = labelinf.loc[labelinf['label'] == propername, "parent"].values[
            #     1]
            # roinames.loc[row, 'neighbour_region1'] = \
            # labelinf.loc[labelinf['label'] == propername, "neighbour"].values[1]
            roinames.loc[row, 'mainparent_region'] = np.unique(
                parentinfo.loc[parentinfo['s122'] == roinames.loc[row, 'roival_region'], 's7'].values)
            roinames.loc[row, 'mainparentname_region'] = parentnames.loc[
                parentnames['roi'] == roinames.loc[row, 'mainparent_region'], 'name'].values
        else:
            roinames.loc[row, 'region_longname'] = dict_labelnames[ind]
            roinames.loc[row, 'roival_region'] = \
            [list(dict_roivalues.keys())[i] for i, n in enumerate(list(dict_roivalues.values())) if n == ind][0]
            # roinames.loc[row, 'parent_region1'] = labelinf.loc[labelinf['label'] == ind[0], "parent"].values[0]
            # roinames.loc[row, 'neighbour_region1'] = labelinf.loc[labelinf['label'] == ind[0], "neighbour"].values[0]
            roinames.loc[row, 'mainparent_region'] = np.unique(
                parentinfo.loc[parentinfo['s122'] == roinames.loc[row, 'roival_region'], 's7'].values)
            roinames.loc[row, 'mainparentname_region'] = parentnames.loc[
                parentnames['roi'] == roinames.loc[row, 'mainparent_region'], 'name'].values


    return roinames
