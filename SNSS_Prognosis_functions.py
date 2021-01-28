#!/usr/bin/env python3.6
"""This module describes functions for analysis of the SNSS Dataset"""
import os
import pandas as pd
from sas7bdat import SAS7BDAT
import numpy as np
import subprocess
from datetime import datetime, date
from csv import DictReader
from shutil import rmtree
from json import load as jsonLoad
import functools
import itertools

from colour import Color
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import seaborn as sb
import textwrap as txtwrp
import ast
import imageio as imgio
import tqdm
import pickle

import scipy.stats as stats
import statsmodels.stats.api as sms
import scipy.interpolate as interpolate
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant
from sklearn import preprocessing, decomposition, manifold
from sklearn.metrics import confusion_matrix, \
    accuracy_score, roc_auc_score, roc_curve, \
    classification_report, precision_score, recall_score, explained_variance_score, r2_score, f1_score
from scipy.stats import logistic
from scipy.optimize import curve_fit

# import graphviz
import pydot
from tensorflow.keras.metrics import top_k_categorical_accuracy
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, AlphaDropout, LeakyReLU
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import CSVLogger, TensorBoard, Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.backend import clear_session
import tensorflow.compat as tfCompat


__author__ = "Oliver Shipston-Sharman"
__copyright__ = "Copyright 2018, Oliver Shipston-Sharman"
__credits__ = ["Oliver Shipston-Sharman", "Christian Hansen", "Alan Carson", "Jon Stone"]

__license__ = "Apache-2.0"
__version__ = "0.1.0"
__maintainer__ = "Oliver Shipston-Sharman"
__email__ = "oliver.shipston-sharman@nhs.net"


""" Short title

Description

Args:
    arg1: arg1 Description

Returns:
    output1: output1 description.

Raises:
    excpeption1: excpetion circumstances.
"""

def loadJSON(fname):
    # Load configuration
    f = open(fname)  # Open config file...
    cfg = jsonLoad(f)  # Load data...
    f.close()  # Close config file...
    return cfg

def moduleInit():
    pd.options.display.max_columns = None
    pd.options.display.max_rows = 20
    tfCompat.v1.disable_eager_execution()

def rmws(strList):
    stripList = []
    for s in strList:
        stripList.append(s.replace(" ", ""))
    return stripList

def timeAppend(varList, T):
    timeVarList = []
    for v in varList:
        timeVarList.append(T + '_' + v)
    return timeVarList

def autoscale(x):
    return (x-np.min(x))/np.max(x)

def normalise(x):
    return (x-np.mean(x))/np.std(x)

def import_SNSS(usr, pwd, local_file=0):
    """ Mount UoE CMVM smb and import SNSS as dataframe.

    Note you must have access permissions to specific share.
    Keyword arguments:
    usr = Edinburgh University matriculation number
    pwd = Edinburgh University Passowrd

    Location of data is specified in a JSON config file not included.
    The SNSS dataset includes confidential patient information and must be
    handled according to Caldicott principles.
    """
    cfg = loadJSON("config.json")
    if local_file:
        print('Importing local data file...')
        # Open and read SNSS data file
        fp = '../../../../../Volumes/mount/SNSSFull.sas7bdat'
        f = SAS7BDAT(fp)
        rawDf = f.to_data_frame()
        print('Dataframe loaded!')
    else:
        cmd = "mount_smbfs"
        mountCmd = cmd+" //'"+cfg['dom']+";"+usr+":"+pwd+"'@"+cfg['shr']+" "+cfg['mnt']
        uMountCmd = 'umount raw_data/mount/'
        # Send smb mount command..
        print('Mounting datashare...')
        smbCall = subprocess.call(mountCmd, shell=True)
        # Open and read SNSS data file
        f = SAS7BDAT(cfg['fpath'])
        print('Converting sas7bdat file to pd.dataframe...')
        rawDf = f.to_data_frame()
        print('Conversion completed! Closing file...')
        f.close()
        print('Attempting Unmount..')
        try:
            smbCall = subprocess.call(uMountCmd, shell=True)
            print('dataShare Unmounted Successfully!')
        except(OSError, EOFError):
            print('Unmount failed...')
    return rawDf


def SNSSNullity(raw):
    """ Assess nullity of raw data import

    Takes the raw imported dataset, ensures index integrity, assigns new binary
    variables for follow up at each study timepoint and computes attrittion numbers
    and ratios for each.

    Args:
        raw: Pandas DataFrame object from SAS7BDAT file.

    Returns:
        raw: The validated raw dataframe.
        retentionTable: A pandas dataframe of counts for use in scripts if required.

    Raises:
        NONE
        """
    # Assign nPatid as index variable.
    raw = raw.set_index('nPatid', verify_integrity=True)
    # Convert diagnostic nullity into binary variable in dataframe.
    raw['All'] = raw.T0_PatID.notna()
    raw['T1_HCData'] = raw.T1_HealthChange.notna()
    raw['T2_HCData'] = raw.T2_HealthChange.notna()
    raw['T1and2_HCData'] = (raw.T2_HealthChange.notna()) & (raw.T1_HealthChange.notna())
    # Quantify diagnostic nullity and export
    T = []
    FULabels = ['T1_HCData', 'T2_HCData', 'T1and2_HCData']
    for FU in FULabels:
        T.append(raw.groupby(FU)['ExpGroups'].agg([('Total', 'count'),
                                                   ('Label', lambda x:
                                                       tuple(np.unique(x[~np.isnan(x)],
                                                             return_counts=True)[0])),
                                                   ('N(i)', lambda x:
                                                       tuple(np.unique(x[~np.isnan(x)],
                                                             return_counts=True)[1])),
                                                   ('%', lambda x:
                                                       tuple((np.unique(x[~np.isnan(x)],
                                                             return_counts=True)[1]/sum(~np.isnan(x))*100).round(2)))]))

    retentionTable = pd.concat(T, keys=FULabels, axis=0)
    retentionTable.index = retentionTable.index.rename(['', 'FUDataAvailable'])
    retentionTable.to_csv('output/0_SNSS_retention.tsv', sep='\t')
    return raw, retentionTable


def SNSSCompoundVariables(df):
    """Produce variable compund measures e.g. SF12, HADS etc.

    Adds the specified custom variables normally products or sums of other Variables
    or binarisation etc to the provided dataframe. This function also undertakes
    SIMD quintile mapping to patient postcodes.

    Args:
        df: Pandas dataframe.

    Returns:
        df: The dataframe with new variables added..

    Raises:
        KeyError, ValueError: If errors in postcode mapping.
    """
    # Deactivate assignment warning which slows down SIMD processing.
    pd.options.mode.chained_assignment = None

    # Declare variable groups
    varGroups = {'PHQ13': ['StomachPain', 'BackPain', 'Paininarmslegsjoints',
                           'Headaches', 'Chestpain', 'Dizziness',
                           'FaintingSpells', 'HeartPoundingorRacing', 'ShortnessofBreath',
                           'Constipation', 'NauseaorGas', 'Tired', 'Sleeping'],
                 'NeuroSymptoms': ['Lackofcoordination', 'MemorConcentration', 'LossofSensation',
                                   'LossofVision', 'LossofHearing', 'Paralysisorweakness',
                                   'DoubleorBlurredVision', 'DifficultySwallowing',
                                   'DifficultySpeaking', 'SeizureorFit',
                                   'AnxietyattackorPanicAttack', 'Littleinterestorpleasure',
                                   'Feelingdownorhopeless', 'Nervesorfeelinganxious',
                                   'Worryingalot'],
                 'IllnessWorry': ['Wworry', 'Wseriousworry', 'Wattention'],
                 'Satisfaction': ['Sat1', 'Sat2', 'Sat3', 'Sat4', 'Sat5', 'Sat6', 'Sat7', 'Sat8'],
                 'other': ['LossofHearing', 'Littleinterestorpleasure', 'Feelingdownorhopeless',
                           'Nervesorfeelinganxious', 'Worryingalot', 'AnxietyattackorPanicAttack']}
    #  Time specify certain groups into useful keysets.
    T0IllnessWorryKeys = timeAppend(varGroups['IllnessWorry'], 'T0')

    T0PHQ13Keys = timeAppend(varGroups['PHQ13'], 'T0')
    T1PHQ13Keys = timeAppend(varGroups['PHQ13'], 'T1')
    T2PHQ13Keys = timeAppend(varGroups['PHQ13'], 'T2')

    T0PHQNeuro28Keys = timeAppend(varGroups['PHQ13'] + varGroups['NeuroSymptoms'], 'T0')
    T1PHQNeuro28Keys = timeAppend(varGroups['PHQ13'] + varGroups['NeuroSymptoms'], 'T1')
    T2PHQNeuro28Keys = timeAppend(varGroups['PHQ13'] + varGroups['NeuroSymptoms'], 'T2')

    T0SatisfactionKeys = timeAppend(varGroups['Satisfaction'], 'T0')
    T1SatisfactionKeys = timeAppend(varGroups['Satisfaction'], 'T1')

    # Criteria Used for defining successful follow up as T1 any satisfaction data available..
    # df['T1_Satisfaction_Bool'] = df['T1_Satisfaction_Total'].notna()  # Strict
    df['T1_Satisfaction_Bool'] = df[T1SatisfactionKeys].notna().any(axis=1)  # Loose

    # Add binarised ExpGroups.
    df['ExpGroups_bin'] = (df['ExpGroups']-2)*-1

    # Add binarised gender.
    df['Gender_bin'] = df['Gender']-1

    # Adding summative compound measures
    df['T0_PHQNeuro28_Total'] = df[T0PHQNeuro28Keys].sum(axis=1, skipna=False)
    df['T1_PHQNeuro28_Total'] = df[T1PHQNeuro28Keys].sum(axis=1, skipna=False)
    df['T2_PHQNeuro28_Total'] = df[T2PHQNeuro28Keys].sum(axis=1, skipna=False)

    df['T0_PHQ13_Total'] = df[T0PHQ13Keys].sum(axis=1, skipna=False)
    df['T1_PHQ13_Total'] = df[T1PHQ13Keys].sum(axis=1, skipna=False)
    df['T2_PHQ13_Total'] = df[T2PHQ13Keys].sum(axis=1, skipna=False)

    df['T0_IllnessWorry'] = df[T0IllnessWorryKeys].sum(axis=1, skipna=False)

    df['T0_Satisfaction_Total'] = df[T0SatisfactionKeys].sum(axis=1, skipna=False)
    df['T1_Satisfaction_Total'] = df[T1SatisfactionKeys].sum(axis=1, skipna=False)
    df['T2_Satisfaction_Total'] = df[T1SatisfactionKeys].sum(axis=1, skipna=False)

    # Adding boolean compound measures
    df['T0_NegExpectation'] = (df['T0_IPQ1'] > 3).astype(int)  # Define "Negative Expectation"
    df['T0_NegExpectation'].loc[df['T0_IPQ1'].isna()] = np.nan  # Boolean operator treats NaN as 0 so replace with NaNs

    df['T0_PsychAttribution'] = ((df['T0_C7'] > 3) | (df['T0_C8'] > 3)).astype(int)
    df['T0_PsychAttribution'].loc[(df['T0_C7'].isna()) | (df['T0_C8'].isna())] = np.nan
    df['T0_LackofPsychAttribution'] = (df['T0_PsychAttribution']-1)*-1


    for S in ['T0_Sat1', 'T0_Sat2', 'T0_Sat3',
              'T0_Sat4', 'T0_Sat5', 'T0_Sat6', 'T0_Sat7', 'T0_Sat8']:
        satNAIdx = df[S].isna()
        df[S + '_Poor_Bin'] = df[S] <= 2  # Binarise Satsifaction into Poor/Fair or not
        df[S + '_Poor_Bin'].loc[satNAIdx] = np.nan

    # Add binned measures
    df['T0_PHQ13_Binned'] = pd.cut(df['T0_PHQ13_Total'], [0, 2.1, 5.1, 8.1, 13.1],
                                   labels=['0-2', '3-5', '6-8', '9-13'],
                                   right=True, include_lowest=True)
    df['T0_PHQ13_BinInt'] = pd.cut(df['T0_PHQ13_Total'], [0, 2.1, 5.1, 8.1, 13.1],
                                   labels=False,
                                   right=True, include_lowest=True)
    df['T0_PHQNeuro28_Binned'] = pd.cut(df['T0_PHQNeuro28_Total'], [0, 5.1, 8.1, 13.1, 27.1],
                                   labels=['0-5', '6-8', '9-13', '14-27'],
                                   right=True, include_lowest=True)
    df['T0_PHQNeuro28_BinInt'] = pd.cut(df['T0_PHQNeuro28_Total'], [0, 5.1, 8.1, 13.1, 27.1],
                                   labels=False,
                                   right=True, include_lowest=True)
    df['AgeBins'] = pd.cut(df['Age'], [0, 36, 46, 56, max(df['Age'])+0.1],
                           labels=['<=35', '36-45', '46-55', '>=56'],
                           right=True, include_lowest=True)
    df['AgeBinInt'] = pd.cut(df['Age'], [0, 36, 46, 56, max(df['Age'])+0.1],
                             labels=False,
                             right=True, include_lowest=True)
    df['T0_HADS_Binned'] = pd.cut(df['T0_HADS'], [0, 7.1, 14.1, 21.1, max(df['T0_HADS'])+0.1],
                                  labels=['0-7', '8-14', '15-21', '>=22'],
                                  right=True, include_lowest=True)
    df['T0_HADS_BinInt'] = pd.cut(df['T0_HADS'], [0, 7.1, 14.1, 21.1, max(df['T0_HADS'])+0.1],
                                  labels=False,
                                  right=True, include_lowest=True)
    df['T0_SF12_PF_Binned'] = pd.cut(df['T0_SF12_PF'], [-0.1, 24.9, 49.9, 74.9, 99.9, 100.1],
                                  labels=['0', '25', '50', '75', '100'],
                                  right=True, include_lowest=True)
    df['T0_SF12_PF_BinInt'] = pd.cut(df['T0_SF12_PF'], [-0.1, 24.9, 49.9, 74.9, 99.9, 100.1],
                                  labels=False,
                                  right=True, include_lowest=True)

    # Add binarised outcomes
    poorOutcomeDict = {0: 1, 1: 1, 2: 1, 3: 0, 4: 0}
    strictPoorOutcomeDict = {0: 1, 1: 1, 2: 0, 3: 0, 4: 0}
    ternaryPoorOutcomeDict = {0: 2, 1: 2, 2: 1, 3: 0, 4: 0}
    df['T1_poorCGI'] = df['T1_HealthChange'].replace(poorOutcomeDict)
    df['T1_poorIPS'] = df['T1_SymptomsChange'].replace(poorOutcomeDict)
    df['T2_poorCGI'] = df['T2_HealthChange'].replace(poorOutcomeDict)
    df['T2_poorIPS'] = df['T2_SymptomsChange'].replace(poorOutcomeDict)
    df['T2_strictPoorCGI'] = df['T2_HealthChange'].replace(strictPoorOutcomeDict)
    df['T2_strictPoorIPS'] = df['T2_SymptomsChange'].replace(strictPoorOutcomeDict)
    df['T2_ternaryCGI'] = df['T2_HealthChange'].replace(ternaryPoorOutcomeDict)
    df['T2_ternaryIPS'] = df['T2_SymptomsChange'].replace(ternaryPoorOutcomeDict)

    # Add relative secondary outcomes
    df['T0T1_SF12_NormedMCS'] = df['T1_SF12_NormedMCS'] - df['T0_SF12_NormedMCS']
    df['T1T2_SF12_NormedMCS'] = df['T2_SF12_NormedMCS'] - df['T1_SF12_NormedMCS']
    df['T0T2_SF12_NormedMCS'] = df['T2_SF12_NormedMCS'] - df['T0_SF12_NormedMCS']
    df['T0T2_SF12_binaryNormedMCS'] = (df['T0T2_SF12_NormedMCS'] < 0).astype(int)
    df['T0T2_SF12_binaryNormedMCS'].loc[df['T0T2_SF12_NormedMCS'].isna()] = np.nan

    df['T0T1_SF12_NormedPCS'] = df['T1_SF12_NormedPCS'] - df['T0_SF12_NormedPCS']
    df['T1T2_SF12_NormedPCS'] = df['T2_SF12_NormedPCS'] - df['T1_SF12_NormedPCS']
    df['T0T2_SF12_NormedPCS'] = df['T2_SF12_NormedPCS'] - df['T0_SF12_NormedPCS']
    df['T0T2_SF12_binaryNormedPCS'] = (df['T0T2_SF12_NormedPCS'] < 0).astype(int)
    df['T0T2_SF12_binaryNormedPCS'].loc[df['T0T2_SF12_NormedPCS'].isna()] = np.nan

    df['T0T1_HADS'] = df['T1_HADS'] - df['T0_HADS']
    df['T1T2_HADS'] = df['T2_HADS'] - df['T1_HADS']
    df['T0T2_HADS'] = df['T2_HADS'] - df['T0_HADS']
    df['T0T2_binaryHADS'] = (df['T0T2_HADS'] < 0).astype(int)
    df['T0T2_binaryHADS'].loc[df['T0T2_HADS'].isna()] = np.nan

    df['T0T1_PHQNeuro28_Total'] = df['T1_PHQNeuro28_Total'] - df['T0_PHQNeuro28_Total']
    df['T1T2_PHQNeuro28_Total'] = df['T2_PHQNeuro28_Total'] - df['T1_PHQNeuro28_Total']
    df['T0T2_PHQNeuro28_Total'] = df['T2_PHQNeuro28_Total'] - df['T0_PHQNeuro28_Total']
    df['T0T2_binaryPHQNeuro28_Total'] = (df['T0T2_PHQNeuro28_Total'] < 0).astype(int)
    df['T0T2_binaryPHQNeuro28_Total'].loc[df['T0T2_PHQNeuro28_Total'].isna()] = np.nan

    print('SIMD 2004 to 2006 Postcode conversion...')
    SIMD04 = pd.read_csv('raw_data/SIMDData/postcode_2006_2_simd2004.csv', index_col=0)
    nullIdx = SIMD04['simd2004rank'].str.contains(' ')
    domains = ['inc', 'emp', 'hlth', 'educ', 'access', 'house']
    for d in domains:
        SIMD04['simd2004_' + d + '_quintile'] = 5-pd.qcut(SIMD04['simd2004_' + d + '_rank']
                                                          [~nullIdx].astype(float), 5,
                                                          retbins=False, labels=False)

    SIMDDict = dict(zip([str.replace(' ', '') for str in SIMD04.sort_index().index.values.tolist()],
                        SIMD04[['simd2004_sc_quintile',
                                'simd2004score',
                                'simd2004_inc_score',
                                'simd2004_emp_score',
                                'simd2004_hlth_score',
                                'simd2004_educ_score',
                                'simd2004_access_score',
                                'simd2004_house_score',
                                'simd2004_inc_quintile',
                                'simd2004_emp_quintile',
                                'simd2004_hlth_quintile',
                                'simd2004_educ_quintile',
                                'simd2004_access_quintile',
                                'simd2004_house_quintile']].values))
    # Initialising variables as NaN arrays
    df['T0_SIMD04'] = np.nan
    df['T0_SIMD04_score'] = np.nan
    for d in domains:
        df['T0_SIMD04_' + d + '_score'] = np.nan
        df['T0_SIMD04_' + d + '_quintile'] = np.nan

    print('Constructed SIMD quintiles and Initialised Panda Variables')
    print('Iterating through postcodes')
    i = 0
    for p in df['Postcode']:
        if (p == '') | pd.isnull(p):
            df['Postcode'].iloc[i] = np.nan
            df['T0_SIMD04'].iloc[i] = np.nan
            i = i + 1
        # print('No Postcode Data')
        else:
            try:
                p = p.replace(' ', '')
                # print(p)
                df['T0_SIMD04'].iloc[i] = int(SIMDDict[p][0])
                df['T0_SIMD04_score'].iloc[i] = float(SIMDDict[p][1])
                dd = 2
                for d in domains:
                    df['T0_SIMD04_' + d + '_score'].iloc[i] = float(SIMDDict[p][dd])
                    df['T0_SIMD04_' + d + '_quintile'].iloc[i] = int(SIMDDict[p][dd+len(domains)])
                    dd += 1
            except (KeyError, ValueError) as err:
                # print('%s: Error!' % (p))
                df['T0_SIMD04'].iloc[i] = np.nan
                # print('No SIMD04 postcode map')
            i = i + 1

    # Add most deprived binarisation
    df['T0_SIMD04_bin'] = df['T0_SIMD04'] >= 4

    # Add interaction variables
    df['Diagnosis*T0_IncapacityBenefitorDLA'] = df['Diagnosis']*df['T0_IncapacityBenefitorDLA']
    df['ExpGroups*T0_IncapacityBenefitorDLA'] = df['ExpGroups']*df['T0_IncapacityBenefitorDLA']
    df['ExpGroups_bin*T0_IncapacityBenefitorDLA'] = df['ExpGroups_bin']*df['T0_IncapacityBenefitorDLA']
    df['ExpGroups_bin*T0_LackofPsychAttribution'] = df['ExpGroups_bin']*df['T0_LackofPsychAttribution']
    df['ExpGroups_bin*T0_SIMD04_bin'] = df['ExpGroups_bin']*df['T0_SIMD04_bin']
    df['ExpGroups_bin*T0_SF12_PF_BinInt'] = df['ExpGroups_bin']*df['T0_SF12_PF_BinInt']
    df['ExpGroups_bin*T0_NegExpectation'] = df['ExpGroups_bin']*df['T0_NegExpectation']
    df['ExpGroups_bin*Gender_bin'] = df['ExpGroups_bin']*df['Gender_bin']

    print('Complete!')
    return df


def cohen_d(x, y):
    stats = {}
    nx = len(x); meanx = np.mean(x); stdx = np.std(x, ddof=1); semx = stdx/np.sqrt(nx);
    ny = len(y); meany = np.mean(y); stdy = np.std(y, ddof=1); semy = stdy/np.sqrt(ny);
    meancix = [meanx+(1.96*i*semx) for i in [-1, 1]]
    meanciy = [meany+(1.96*i*semy) for i in [-1, 1]]

    dof = nx + ny - 2
    d = (meanx - meany) / np.sqrt(((nx-1)*stdx ** 2 +
                                             (ny-1)*stdy ** 2) / dof)
    vard = (((nx+ny)/(nx*ny))+((d**2)/(2*(nx+ny-2))))*((nx+ny)/(nx+ny-2))
    sed = np.sqrt(vard)
    cid = [d+(1.96*i*sed) for i in [-1, 1]]

    stats['d'] = d
    stats['cid'] = cid
    stats['mean'] = [meanx, meany]
    stats['std'] = [stdx, stdy]
    stats['sem'] = [semx, semy]
    return d, stats


def cramersV(nrows, ncols, chisquared, correct_bias=True):
    nobs = nrows*ncols
    if correct_bias is True:
        phi = 0
    else:
        phi = chisquared/nobs
    V = np.sqrt((phi**2)/(min(nrows-1, ncols-1)))
    return V, phi


def partitionData(df, partitionRatio=0.7):
    """ Partition data into training and evaluation sets

    Takes a dataframe and returns two arrays with the proportion to use for
    training declared as the partition ratio and the other as evaluation of
    (1-partitionRatio) size.

    Args:
        df: Pandas DataFrame to be partitioned.
        partitionRatio: Ratio of the data to be used for training.

    Returns:
        trainIdx: The indices of data asssigned to training set.
        evalIdx: The indices of data asssigned to eval set.

    Raises:
        NONE
    """
    randIdx = np.linspace(0, df.shape[0]-1, df.shape[0]).astype(int)
    np.random.shuffle(randIdx)
    trainIdx = randIdx[0:round(df.shape[0]*partitionRatio)]
    evalIdx = randIdx[round(df.shape[0]*(partitionRatio)):len(randIdx)]
    return trainIdx, evalIdx


def FollowUpandBaselineComparison(df):
    """ A group-wise and follow-up wise comparison of declared Vars

    Takes a pandas dataframe and as per the declared variables of interest below,
    compares between groups and between lost to follow up and retained.

    Args:
        df: Pandas DataFrame to be assessed.

    Returns:
        NONE: All relevant tables are exported to CSV in the function.

    Raises:
        NONE
    """
    def sigTest(G, varList, vType, df):
        sigDict = {}
        if vType == 'cat':
            for v in varList:
                T = pd.crosstab(index=df[G], columns=df[v],
                                margins=False, normalize=False)
                chi2Stat, chi2p, _, _ = stats.chi2_contingency(T, correction=True)
                cats = np.unique(df[v].dropna())
                if len(cats) == 2:
                    LOR = np.log((T.iloc[0,0]*T.iloc[1,1])/(T.iloc[1,0]*T.iloc[0,1]))
                    SE = np.sqrt((1/T.iloc[0,0])+(1/T.iloc[1,0])+(1/T.iloc[0,1])+(1/T.iloc[1,1]))
                    CI = [np.exp(LOR-1.96*SE), np.exp(LOR+1.96*SE)]
                    OR = np.exp(LOR)
                else:
                    OR = np.nan
                    CI = np.nan
                sigDict[v] = [chi2p, chi2Stat, OR, CI]
        elif vType == 'cont':
            for v in varList:
                if G == 'ExpGroups':
                    Gi = [1, 2]
                elif G == 'T2_HCData':
                    Gi = [0, 1]
                elif G == 'T2_poorCGI':
                    Gi = [0, 1]
                cm = CompareMeans.from_data(df[v][(df[G] == Gi[0]) & (df[v].notna())],
                                            df[v][(df[G] == Gi[1]) & (df[v].notna())])
                tStat, tp, _ = cm.ttest_ind()
                cohend, cohenstat = cohen_d(cm.d1.data, cm.d2.data)
                sigDict[v] = [tp, tStat, cohend, cohenstat['cid']]
        sigT = pd.DataFrame.from_dict(sigDict, orient='index', columns=['p', 'stat', 'effect', 'effectCI'])
        return sigT

    def varTables(G, varList, vType, df):
        if vType == 'cont':
            T = df.groupby(G)[varList].agg([('N', 'count'), ('Mean', 'mean'),
                                            ('SD', 'std')])
        elif vType == 'cat':
            T = df.groupby(G)[varList].\
                agg([('N', 'count'),
                     ('i', lambda x:
                     tuple(np.unique(x[~np.isnan(x)],
                           return_counts=True)[0])),
                     ('N(i)', lambda x:
                     tuple(np.unique(x[~np.isnan(x)],
                           return_counts=True)[1])),
                     ('%', lambda x:
                     tuple(np.unique(x[~np.isnan(x)],
                           return_counts=True)[1]/sum(~np.isnan(x))))])
        return T

    contVars = ['Age', 'T0_PHQ13_Total', 'T0_PHQNeuro28_Total', 'T0_HADS', 'T0_IllnessWorry', 'T0_SF12_PF']
    catVars = ['AgeBins', 'Gender', 'ExpGroups', 'T0_PHQ13_Binned', 'T0_SF12_PF', 'T0_HADS_Binned',
               'T0_NegExpectation', 'T0_PsychAttribution', 'T0_IllnessWorry', 'T0_IncapacityBenefitorDLA', 'T0_SIMD04_bin',
               'ExpGroups_bin*T0_IncapacityBenefitorDLA', 'ExpGroups_bin*T0_LackofPsychAttribution','T0_Inemployment']

    groupVar = 'T2_HCData'
    catT = varTables(G=groupVar, varList=catVars, vType='cat', df=df)
    catStats = sigTest(G=groupVar, varList=catVars, vType='cat', df=df)
    catT.transpose().to_csv('output/0_FollowUpCategoricalTable.tsv', sep='\t')
    catStats.transpose().to_csv('output/0_FollowUpCategoricalStats.tsv', sep='\t')

    contT = varTables(G=groupVar, varList=contVars, vType='cont', df=df)
    contStats = sigTest(G=groupVar, varList=contVars, vType='cont', df=df)
    contT.transpose().to_csv('output/0_FollowUpContinuousTable.tsv', sep='\t')
    contStats.transpose().to_csv('output/0_FollowUpContinuousStats.tsv', sep='\t')

    groupVar = 'ExpGroups'
    catT = varTables(G=groupVar, varList=catVars, vType='cat', df=df[df.T2_HCData == 1])
    catStats = sigTest(G=groupVar, varList=catVars, vType='cat', df=df[df.T2_HCData == 1])
    catT.transpose().to_csv('output/0_BaselineCategoricalTable.tsv', sep='\t')
    catStats.transpose().to_csv('output/0_BaselineCategoricalStats.tsv', sep='\t')

    contT = varTables(G=groupVar, varList=contVars, vType='cont', df=df[df.T2_HCData == 1])
    contStats = sigTest(G=groupVar, varList=contVars, vType='cont', df=df[df.T2_HCData == 1])
    contT.transpose().to_csv('output/0_BaselineContinuousTable.tsv', sep='\t')
    contStats.transpose().to_csv('output/0_BaselineContinuousStats.tsv', sep='\t')

    groupVar = 'T2_poorCGI'
    catT = varTables(G=groupVar, varList=catVars, vType='cat', df=df[df.T2_HCData == 1])
    catStats = sigTest(G=groupVar, varList=catVars, vType='cat', df=df[df.T2_HCData == 1])
    catT.transpose().to_csv('output/0_OutcomeCategoricalTable.tsv', sep='\t')
    catStats.transpose().to_csv('output/0_OutcomeCategoricalStats.tsv', sep='\t')

    contT = varTables(G=groupVar, varList=contVars, vType='cont', df=df[df.T2_HCData == 1])
    contStats = sigTest(G=groupVar, varList=contVars, vType='cont', df=df[df.T2_HCData == 1])
    contT.transpose().to_csv('output/0_OutcomeContinuousTable.tsv', sep='\t')
    contStats.transpose().to_csv('output/0_OutcomeContinuousStats.tsv', sep='\t')
    return


def SNSSPrimaryOutcomeMeasures(df):
    """ Compare IPS and CGI outcomes between functional groups.

    This function compares CGI and IPS both in raw and pooled form between
    functional groups. Outputs tables of counts and proportions of reported outcomes.

    Args:
        df: Pandas DataFrame to be assessed.

    Returns:
        NONE: All relevant tables are exported to CSV in the function.

    Raises:
        NONE
    """
    outcomes = [['T2_HealthChange', 'T2_SymptomsChange'], ['T2_poorCGI', 'T2_poorIPS']]
    outcomeTag = ['', 'Pool']
    i = 0
    for O in outcomes:
        PrimaryOutcomeGroupT = []
        PrimaryOutcomeGroupT.append(pd.crosstab(index=df.ExpGroups, columns=df[O[0]],
                                                margins=False, normalize=False,
                                                dropna=True))
        PrimaryOutcomeGroupT.append(pd.crosstab(index=df.ExpGroups, columns=df[O[0]],
                                                margins=False, normalize='index',
                                                dropna=True))
        PrimaryOutcomeGroupT.append(pd.crosstab(index=df.ExpGroups, columns=df[O[1]],
                                                margins=False, normalize=False,
                                                dropna=True))
        PrimaryOutcomeGroupT.append(pd.crosstab(index=df.ExpGroups, columns=df[O[1]],
                                                margins=False, normalize='index',
                                                dropna=True))
        PrimaryOutcomeGroupTExport = pd.concat(PrimaryOutcomeGroupT,
                                               keys=['CGI_N', 'CGI_%',
                                                     'IPS_N', 'IPS_%'],
                                               axis=0)
        if i:
            CGIchi2stat, CGIchi2p, _, _ = stats.chi2_contingency(PrimaryOutcomeGroupT[0],
                                                                 correction=True)
            CGIfisherOR, CGIfisherp = stats.fisher_exact(PrimaryOutcomeGroupT[0])
            IPSchi2stat, IPSchi2p, _, _ = stats.chi2_contingency(PrimaryOutcomeGroupT[2],
                                                                 correction=True)
            IPSfisherOR, IPSfisherp = stats.fisher_exact(PrimaryOutcomeGroupT[2])
            PrimaryOutcomeGroupTExport['chi2p'] = [CGIchi2p]*4 + [IPSchi2p]*4
            PrimaryOutcomeGroupTExport['fisher2p'] = [CGIfisherp]*4 + [IPSfisherp]*4
        PrimaryOutcomeGroupTExport.to_csv('output/1_PrimaryOutcome' + outcomeTag[i] + 'byGroup.tsv',
                                          sep='\t')
        i = i+1
    return


def multi_text(ax, x, y, s, txt_params={}):
    """ Matplotlib multi-line text plotting

    Takes a matplotlib axes, set of strings and positions and plots.

    Args:
        ax: Matplotlib axes.
        x: Array of x values
        y: constant y value
        s: Array of strings.
        txt_params: Dict of text params.

    Returns:
        NONE: Text is plotted onto provided axes.

    Raises:
        NONE
    """
    for i in range(len(s)):
        ax.text(x[i], y, s[i], **txt_params)


def stackedBarPlot(x_var, y_vars, df, featMetaData):
    """ Plots stacked bar charts as per declared variables.

    Takes a matplotlib axes, set of strings and positions and plots a stacked bar
    chart with the X variables being subdivided by the Y variables.

    Args:
        x_var: Names of variables on X_axis
        y_vars: Names of variables with which to subdivide X variables.
        df: Pandas dataframe to be used.
        featMetaData: Variable meta data provided in JSON file.

    Returns:
        NONE: Figure is saved in function.

    Raises:
        NONE
    """
    if not isinstance(y_vars, list):
        y_vars = [y_vars]
    fig_params={'num': 1,
                'figsize': (6*len(y_vars), 6),
                'dpi': 200,
                'frameon': False}
    txt_params={'fontsize': 6,
                'ha': 'center',
                'va': 'center'}
    label_params={'fontsize': 10,
                  'ha': 'center',
                  'va': 'top'}
    fig = plt.figure(**fig_params)
    sp = 1
    for y_var in y_vars:
        data = df.dropna(subset=[y_var])
        ax_params={'title': featMetaData[y_var]['label'],
                   'ylabel': 'Normalised Frequency',
                   'xlabel': y_var}
        ax = fig.add_subplot(1, len(y_vars), sp, **ax_params)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        x_cats = np.unique(data[x_var])
        x_cats = x_cats[~np.isnan(x_cats)]
        x_var_meta = featMetaData[x_var]
        y_cats = np.unique(data[y_var])
        y_cats = y_cats[~np.isnan(y_cats)]
        y_var_meta = featMetaData[y_var]

        xMinorTicks = []
        xMinorLabels = []
        x = 0
        bw = 0.8
        y_bottom=0

        for xc in x_cats:
            for yc in y_cats:
                y = np.nanmean(data[y_var][data[x_var] == xc] == yc)
                t = str(int(round(y*100, 0)))+'%'
                ax.bar(x=x, height=y, width=bw,
                       color=ast.literal_eval(y_var_meta['colors'][y_var_meta['values'].index(yc)]),
                       bottom=y_bottom)
                ax.text(x, y_bottom+(y/2), t, **txt_params)
                xMinorLabels.append(x_var_meta['truncvaluelabels'][x_var_meta['values'].index(xc)])
                xMinorTicks.append(x)
                y_bottom = y+y_bottom
            y_bottom=0
            x += 1

        ax.set_xticks(xMinorTicks)
        ax.set_xticklabels(xMinorLabels, **label_params)
        sp+=1
    fig.savefig('output/1_SNSSPrimaryOutcomeStackedBars.pdf', dpi=300,
                format='pdf', pad_inches=0.1, bbox_inches='tight')
    plt.close()


def subCatBarPlot(x_vars, x_sub_var, df, featMetaData):
    """ Plots stacked bar charts as per declared variables.

    Takes a matplotlib axes, set of strings and positions and plots a bar
    chart with the X variables being subdivided by the X_sub variables and the
    subdivisions being plotted side by side.

    Args:
        x_vars: Names of variables on X_axis
        x_sub_var: Names of variables with which to subdivide X variables.
        df: Pandas dataframe to be used.
        featMetaData: Variable meta data provided in JSON file.

    Returns:
        NONE: Figure is saved in function.

    Raises:
        NONE
    """
    if  not isinstance(x_vars, list):
        x_vars = [x_vars]
        print('is not list')
    fig_params={'num': 1,
                'figsize': (6*len(x_vars), 6),
                'dpi': 200,
                'frameon': False}
    txt_params={'fontsize': 6,
                'ha': 'center',
                'va': 'bottom'}
    label_params={'fontsize': 10,
                  'ha': 'center',
                  'va': 'top'}
    fig = plt.figure(**fig_params)
    sp = 1
    for x_var in x_vars:
        data = df.dropna(subset=[x_var])
        ax_params={'title': featMetaData[x_var]['label'],
                   'ylabel': 'Normalised Frequency',
                   'xlabel': ''}
        ax = fig.add_subplot(1, len(x_vars), sp, **ax_params)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        x_cats = np.unique(data[x_var])
        x_cats = x_cats[~np.isnan(x_cats)]
        x_var_meta = featMetaData[x_var]
        x_sub_cats = np.unique(data[x_sub_var])
        x_sub_cats = x_sub_cats[~np.isnan(x_sub_cats)]
        x_sub_var_meta = featMetaData[x_sub_var]

        xMinorTicks = []
        xMajorTicks = []
        xMinorLabels = []
        xMajorLabels = []
        x = 0
        bw = 1

        for xc in x_cats:
            for xsc in x_sub_cats:
                y = np.nanmean(data[x_var][data[x_sub_var] == xsc] == xc)
                t = str(int(round(y*100, 0)))+'%'
                ax.bar(x=x, height=y, width=bw,
                       color=x_sub_var_meta['colors'][x_sub_var_meta['values'].index(xsc)])
                ax.text(x, y, t, **txt_params)
                xMinorLabels.append(x_sub_var_meta['truncvaluelabels'][x_sub_var_meta['values'].index(xsc)])
                xMinorTicks.append(x)
                x += 1
            xMajorLabels.append(x_var_meta['truncvaluelabels'][x_var_meta['values'].index(xc)])
            xMajorTicks.append(x-1-((len(x_sub_cats)-1)/2))
            x += 1

        ax.set_xticks(xMinorTicks)
        ax.set_xticklabels(xMinorLabels, **label_params)
        multi_text(ax, xMajorTicks, ax.get_ylim()[1]*-0.1, xMajorLabels, label_params)
        sp+=1
    fig.savefig('output/1_SNSSPrimaryOutcomeBars.pdf', dpi=300,
                format='pdf', pad_inches=0.1, bbox_inches='tight')
    plt.close()


def primaryOutcomePlot(outcome, group_var, data, featMetaData, style='subCat'):
    """ Plots bar charts of declared outcome and grouping var.

    Takes declared variables and plots bar chart accordingly.

    Args:
        outcome: name of outcome variable
        group_var: Names grouping variable i.e. X variables to be used
        data: Pandas dataframe to be used.
        featMetaData: Variable meta data provided in JSON file.
        style: Defaults to side-by-side vs stacked plotting.

    Returns:
        NONE: Figure is saved in respective function.

    Raises:
        NONE
    """
    if style == 'subCat':
        subCatBarPlot(outcome, group_var, data, featMetaData)
    elif style == 'stacked':
        stackedBarPlot(group_var, outcome, data, featMetaData)


def SNSSSecondaryOutcomeMeasures(df):
    """ Plots line chart and produced table of secondary SNSS outcomes

    Takes pandas dataframe and assesses between group differences over time
    of secindary outcome measures including depressino scales and physical/mental
    functioning.

    Args:
        df: Pandas dataframe

    Returns:
        outcomeT: Table of outcome measures grouped by functional diagnosis.

    Raises:
        NONE
    """
    groupVar = 'ExpGroups'
    SNSSVars = loadJSON('raw_data/SNSS_vars.json')
    rowDict = dict(zip(SNSSVars[groupVar]['values'],
                       SNSSVars[groupVar]['valuelabels']))
    outcomes = ['T0_SF12_NormedMCS', 'T1_SF12_NormedMCS', 'T2_SF12_NormedMCS',
                'T0_SF12_NormedPCS', 'T1_SF12_NormedPCS', 'T2_SF12_NormedPCS',
                'T0_PHQNeuro28_Total', 'T1_PHQNeuro28_Total', 'T2_PHQNeuro28_Total',
                'T0_HADS', 'T1_HADS', 'T2_HADS',
                'T0T1_SF12_NormedMCS', 'T1T2_SF12_NormedMCS',
                'T0T1_SF12_NormedPCS', 'T1T2_SF12_NormedPCS',
                'T0T1_PHQNeuro28_Total', 'T1T2_PHQNeuro28_Total',
                'T0T1_HADS', 'T1T2_HADS']

    outcomeT = df.groupby(groupVar)[outcomes].agg([('N', 'count'),
                                                   ('Mean', 'mean'),
                                                   ('SD', 'std'),
                                                   ('CI', lambda x:
                                                   tuple(np.round(
                                                    DescrStatsW(x.dropna()).
                                                    tconfint_mean(), 2)))])
    # Significance testing
    for O in outcomes:
        NE = (df.ExpGroups == 1) & (df[O].notna())
        E = (df.ExpGroups == 2) & (df[O].notna())
        cm = CompareMeans.from_data(df[O].loc[NE], df[O].loc[E])
        outcomeT[O, 'tTestp'] = [cm.ttest_ind()[1]]*2
        outcomeT[O, 'cohend'], _ = cohen_d(cm.d1.data, cm.d2.data)
    outcomeT = outcomeT.sort_index(axis=1)
    outcomeT.rename(index=rowDict).transpose().\
        to_csv('output/2_SecondaryOutcomeMeasures.tsv', sep='\t')
    return outcomeT

def plot_ci(ax, x, y, color, style='t'):
    if style == 't':
        for i in range(len(y)):
            ax.plot([x[i], x[i]], [y[i][0], y[i][1]],
                    color=color, alpha=0.4,
                    marker='_', linewidth=2)


def lineTimeSeriesPlot(y_vars, groupVar, df, featMetaData):
    fig_params={'num': 1,
                'figsize': (6*4, 6),
                'dpi': 200,
                'frameon': False}
    txt_params={'fontsize': 6,
                'ha': 'center',
                'va': 'center'}
    label_params={'fontsize': 10,
                  'ha': 'center',
                  'va': 'top'}
    fig = plt.figure(**fig_params)

    grps = np.unique(df[groupVar])
    grps = grps[~np.isnan(grps)]
    groupVar_meta = featMetaData[groupVar]
    sp = 1
    time = [0, 3, 12]
    for y_var_group in y_vars:
        for y_var in y_var_group:
            ax_params={'title': y_var[0],
                       'ylabel': 'Secondary Measure',
                       'xlabel': 'Time',
                       'xticks': [0, 3, 12],
                       'xticklabels': ['Baseline', '3 Months', '12 Months']}
            ax = fig.add_subplot(1, 4, sp, **ax_params)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            grp_jitter = [0.1, 0.1, 0.1]
            for grp in grps:
                mean_array = []
                ci_array = []
                for T_var in y_var:
                    data = df.dropna(subset=[T_var])
                    mean_array.append(np.nanmean(data[T_var][data[groupVar] == grp]))
                    ci_array.append(DescrStatsW(data[T_var][data[groupVar] == grp]).tconfint_mean())
                ax.plot(time, mean_array, color=groupVar_meta['colors'][groupVar_meta['values'].index(grp)],
                        alpha=0.9, linewidth=4)
                plot_ci(ax, time, ci_array, groupVar_meta['colors'][groupVar_meta['values'].index(grp)],
                        't')
            # ax.set_ylim([0, ax.get_ylim()[1]])
            sp += 1
    fig.subplots_adjust(wspace=0.3)
    fig.savefig('output/2_SNSSSecondaryOutcomePlot.pdf', dpi=300,
                format='pdf', pad_inches=0.1, bbox_inches='tight')
    plt.close()
    # color=groupVar_meta['colors'][groupVar_meta['values'].index(grp)]


def secondaryOutcomePlot(outcome, groupVar, df, featMetaData, style='line'):
    if style == 'line':
        lineTimeSeriesPlot(outcome, groupVar, df, featMetaData)


def SNSSSocioeconomicAssessment(df):
    """ Multiple plots comparing SIMD quintile to functional diagnosis and outcome

    Takes pandas dataframe and plots SIMD quintiles as per each functional Diagnosis
    and primary and secondary outcomes.

    Args:
        df: Pandas dataframe

    Returns:
        NONE: All plots saved within function.

    Raises:
        NONE
    """
# Figure & Table 1: Are functional vs structural patients from different SIMD quintiles?
    SIMDGroupT = []
    SIMDGroupT.append(pd.crosstab(index=[df.ExpGroups], columns=df.T0_SIMD04,
                                  margins=False, normalize='index',
                                  dropna=True))
    SIMDGroupT.append(pd.crosstab(index=[df.ExpGroups], columns=df.T0_SIMD04,
                                  margins=False, normalize=False,
                                  dropna=True))
    SIMDGroupTExport = pd.concat(SIMDGroupT, keys=['N', '%'])
    SIMDGroupTExport.to_csv('output/3_DeprivationGroups.tsv', sep='\t')

    SIMDOutcomeT = []
    SIMDOutcomeT.append(pd.crosstab(index=[df.ExpGroups, df.T2_poorCGI], columns=df.T0_SIMD04,
                                    margins=False, normalize=False,
                                    dropna=True))
    SIMDOutcomeT.append(pd.crosstab(index=[df.ExpGroups, df.T2_poorCGI], columns=df.T0_SIMD04,
                                    margins=False, normalize='index',
                                    dropna=True))
    SIMDOutcomeT.append(pd.crosstab(index=[df.ExpGroups, df.T2_poorIPS], columns=df.T0_SIMD04,
                                    margins=False, normalize=False,
                                    dropna=True))
    SIMDOutcomeT.append(pd.crosstab(index=[df.ExpGroups, df.T2_poorIPS], columns=df.T0_SIMD04,
                                    margins=False, normalize='index',
                                    dropna=True))

    SIMDOutcomeTExport = pd.concat(SIMDOutcomeT, keys=['CGI_N', 'CGI_%', 'IPS_N', 'IPS_%'])
    SIMDOutcomeTExport.to_csv('output/3_DeprivationOutcomeAndGroup.tsv', sep='\t')

    fig1 = plt.figure(num=1, figsize=(5, 5), dpi=200, frameon=False)
    ax = fig1.add_subplot(111)
    sb.distplot(df.T0_SIMD04[(df.T0_SIMD04.notna()) & (df.ExpGroups == 1)],
                ax=ax, kde=False, norm_hist=True, bins=5,
                kde_kws={'bw': 0.55}, hist_kws={'rwidth': 0.8},
                color='xkcd:blood red')
    sb.distplot(df.T0_SIMD04[(df.T0_SIMD04.notna()) & (df.ExpGroups == 2)],
                ax=ax, kde=False, norm_hist=True, bins=5,
                kde_kws={'bw': 0.55}, hist_kws={'rwidth': 0.8},
                color='xkcd:ocean blue')

    1.4+0.8*4
    ax.set_xlabel('SIMD04 Quintile')
    ax.set_xticks(np.linspace(start=1.4, stop=4.6, num=5))
    ax.set_xticklabels(['1 (Least Deprived)',
                        '2', '3', '4',
                        '5 (Most Deprived)'],
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Proportion')
    ax.set_xlim([1, 5])
    ax.legend(labels=['Not Explained', 'Explained'],
              bbox_to_anchor=(1.25, 1), loc=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig1.savefig('output/3_SNSSSocioeconomicGroups.pdf', dpi=300,
                 format='pdf', pad_inches=0.1, bbox_inches='tight')
    plt.close()

# Figure 2: Does SIMD correlate with socioeconomic questions in SNSS and are outcomes different?
    contOutcomes = ['T0_PHQNeuro28_Total',
                    'T0_HADS',
                    'T0_SF12_NormedMCS',
                    'T0_SF12_NormedPCS',
                    'T0T2_PHQNeuro28_Total',
                    'T0T2_HADS',
                    'T0T2_SF12_NormedMCS',
                    'T0T2_SF12_NormedPCS']

    catOutcomes = ['T0_Inemployment',
                   'T0_IncapacityBenefitorDLA',
                   'T2_poorIPS',
                   'T2_poorCGI']

    ylabels = ['Symptom Count (Baseline)',
               'HADS Score (Baseline)',
               'SF12 MCS (Baseline)',
               'SF12 PCS (Baseline)',
               'Symptom Count (12 Month Change)',
               'HADS Score (12 Month Change)',
               'SF12 MCS (12 Month Change)',
               'SF12 PCS (12 Month Change)',
               '% in Employment (Baseline)',
               '% in Receipt of DLA (Baseline)',
               '% Reporting Poor IPS (12 Months)',
               '% Reporting Poor CGI (12 Months)']

    fig2 = plt.figure(num=1, figsize=(16, 12), dpi=200, frameon=False)
    i = 0
    ax = []
    for o in contOutcomes:
        ax.append(fig2.add_subplot(3, 4, i+1))
        sb.boxplot(x='ExpGroups', y=o, hue='T0_SIMD04',
                   data=df, ax=ax[i],
                   palette=sb.cubehelix_palette(5, start=0, reverse=False),
                   flierprops={'marker': '+'})
        ax[i].set_xticklabels(labels=['Unexplained', 'Explained'])
        ax[i].set_ylabel(ylabels[i])
        if i == 3:
            handles, _ = ax[i].get_legend_handles_labels()
            ax[i].legend(handles=handles, labels=['1 (Least Deprived)',
                                                  '2', '3', '4',
                                                  '5 (Most Deprived)'],
                         bbox_to_anchor=(1.55, 1), loc=1)
        else:
            ax[i].legend_.remove()
        i = i+1

    for o in catOutcomes:
        ax.append(fig2.add_subplot(3, 4, i+1))
        sb.barplot(x='ExpGroups', y=o, hue='T0_SIMD04', data=df,
                   palette=sb.cubehelix_palette(5, start=0, reverse=False),
                   ax=ax[i])
        ax[i].set_ylabel(ylabels[i])
        ax[i].set_xticklabels(labels=['Unexplained', 'Explained'])
        ax[i].set_ylim([0, 1])
        ax[i].legend_.remove()
        i = i+1
    fig2.subplots_adjust(wspace=0.3, hspace=0.3)
    fig2.savefig('output/3_SNSSSocioeconomicAssessment.pdf', dpi=300,
                 format='pdf', pad_inches=0.1, bbox_inches='tight')
    plt.close()
# Figure 3: Do individual domains differ in outcome (!!! Not population weighted)
    for Y in ['T0_Inemployment', 'T0_IncapacityBenefitorDLA', 'T2_poorCGI']:
        fig3 = plt.figure(num=1, figsize=(9, 6), dpi=200, frameon=False)
        i = 0
        ax = []
        domains = ['inc', 'emp', 'hlth', 'educ', 'access', 'house']

        for d in domains:
            ax.append(fig3.add_subplot(2, 3, i+1))
            sb.barplot(x='ExpGroups', y=Y, hue='T0_SIMD04_' + d + '_quintile', data=df,
                       palette=sb.cubehelix_palette(5, start=0, reverse=False),
                       ax=ax[i])
            # ax[i].set_ylabel(ylabels[i])
            ax[i].set_xticklabels(labels=['Unexplained', 'Explained'])
            ax[i].set_ylim([0, 1])
            ax[i].legend_.remove()
            ax[i].set_title(d)
            i = i+1
        fig3.subplots_adjust(wspace=0.3, hspace=0.3)
        plt.close()
        # sb.violinplot(x='ExpGroups', y='T0_SIMD04_access_score', hue='T2_SymptomsChange',
        #               palette=sb.cubehelix_palette(5, start=2, reverse=True), data=df)
        fig3.savefig('output/3_SNSSSocioeconomicDomainsAssessment_' + Y + '.pdf', dpi=300,
                     format='pdf', pad_inches=0.1, bbox_inches='tight')
    return


def performanceMetrics(trainDat, evalDat):
    """ General function for assessing training vs eval performance

    Takes two arrays of Nx2 size. Each array is made up of a TRUE label [0] and
    a PREDICTED Score [1], the arrays are training and eval. The function computes
    binary or multivariate performance metrics and outputs a dictionary.

    Args:
        trainDat: An Nx2 array of true labels and predicted scores for the training set.
        trainDat: An Nx2 array of true labels and predicted scores for the eval set.

    Returns:
        perfDict: A dictionary which includes the original scores and labels as well as
                  all computed metrics.
    Raises:
        NONE
    """
    perfDict = {}
    nClasses = len(np.unique(trainDat[0]))
    dLabels = ['train', 'eval']
    i = 0
    for d in [trainDat, evalDat]:
        true = d[0]
        score = d[1]
        if nClasses == 2:  # If binary classification problem...
            perfDict['problemType'] = 'binaryProblem'
            # Calculate 'optimal' ROC operating threshold to assign binary pred.
            fpr, tpr, t = roc_curve(true, score)
            optimalIdx = np.argmax(tpr - fpr)
            optimalThreshold = t[optimalIdx]
            pred = np.reshape((score >= optimalThreshold).astype(int), [len(score), ])
            # Compute Accuracy Scores
            Acc = accuracy_score(true, pred, normalize=True)
            Auroc = roc_auc_score(true, score)
            R2 = r2_score(true, pred)
            f1 = f1_score(true, pred)
            precision = precision_score(true, pred, average='binary')
            recall = recall_score(true, pred, average='binary')
            CM = confusion_matrix(true, pred)
            TN = CM[0][0]
            TP = CM[1][1]
            FN = CM[1][0]
            FP = CM[0][1]
            Sens = TP/(TP+FN)
            Spec = TN/(TN+FP)

            perfDict[dLabels[i] + 'True'] = true
            perfDict[dLabels[i] + 'Pred'] = pred
            perfDict[dLabels[i] + 'Score'] = score

            perfDict[dLabels[i] + 'Acc'] = Acc
            perfDict[dLabels[i] + 'Auroc'] = Auroc
            perfDict[dLabels[i] + 'R2'] = R2
            perfDict[dLabels[i] + 'F1'] = f1
            perfDict[dLabels[i] + 'Precision'] = precision
            perfDict[dLabels[i] + 'Recall'] = recall
            perfDict[dLabels[i] + 'CM'] = CM
            perfDict[dLabels[i] + 'Sens'] = Sens
            perfDict[dLabels[i] + 'Spec'] = Spec
            perfDict[dLabels[i] + 'OperatingThreshold'] = optimalThreshold
            i += 1
        else:  # If multiclass classification problem...
            perfDict['problemType'] = 'multiClassProblem'
            pred = np.argmax(score, axis=1)

            Acc = accuracy_score(true, pred, normalize=True)
            CM = confusion_matrix(true, pred)
            microPrecision = precision_score(true, pred, average='micro')
            microRecall = recall_score(true, pred, average='micro')
            macroPrecision = precision_score(true, pred, average='macro')
            macroRecall = recall_score(true, pred, average='macro')
            # microAuroc = roc_auc_score(true, score, average='micro')
            # macroAuroc = roc_auc_score(true, score, average='macro')

            perfDict[dLabels[i] + 'True'] = true
            perfDict[dLabels[i] + 'Pred'] = pred
            perfDict[dLabels[i] + 'Score'] = score

            perfDict[dLabels[i] + 'Acc'] = Acc
            perfDict[dLabels[i] + 'CM'] = CM
            perfDict[dLabels[i] + 'Precision'] = microPrecision
            perfDict[dLabels[i] + 'Recall'] = microRecall
            perfDict[dLabels[i] + 'MicroPrecision'] = microPrecision
            perfDict[dLabels[i] + 'MicroRecall'] = microRecall
            perfDict[dLabels[i] + 'MacroPrecision'] = macroPrecision
            perfDict[dLabels[i] + 'MacroRecall'] = macroRecall
            # perfDict[dLabels[i] + 'Auroc'] = microAuroc
            # perfDict[dLabels[i] + 'MicroAuroc'] = microAuroc
            # perfDict[dLabels[i] + 'MacroAuroc'] = macroAuroc
            i += 1
    return perfDict

def UVLogisticRegression_v2(df, featureSet, outcomeVar, featMetaData, featDataTypeDict,
                            dummyExceptionDict, trainIdx=[], evalIdx=[]):
    """ Derived from MV code"""
    mdlExportTArray = []
    mdlArray = []
    modelSummaryInfoDict = {}
    for P in featureSet:  # For each feature construct k-1 dummy array and construct model.
        # Exclude missing data from featureSet subset
        rDat = df.dropna(subset=[P] + [outcomeVar])

        # Initialise feat & outcome arrays
        outcome = np.asarray(rDat[outcomeVar]).astype(int)  # Initialise outcome array, MUST BE BINARY
        feats = np.ones([len(rDat), 1]).astype(int)  # Initialise dummy feat array with constant
        featNames = ['constant']  # Initialise dummy featNames array with constant
        featNameIndex = ['constant']
        sigTestIdx = {}
        modelSummaryInfo = {}
        if featDataTypeDict[P] in ['nominal', 'binary']:
            if rDat[P].dtype.name != 'category':  # If not a categorical then convert...
                rDat[P] = pd.Categorical(rDat[P])

            # Drop single category as constant.
            # Decision based on dummy exception dict, defaults to first category.
            try:
                X = pd.get_dummies(rDat[P], drop_first=False).drop(axis=1,
                                                                   columns=dummyExceptionDict[P])
            except (KeyError) as err:
                X = pd.get_dummies(rDat[P], drop_first=True)

            # Translate categorical series labels into SNSS var value labels..
            varDict = dict(zip(featMetaData[P]['values'],
                               featMetaData[P]['valuelabels']))
            for col in range(X.shape[1]):  # Construct featNames array for output
                try:
                    featNames.append(featMetaData[P]['label'] + ' - ' +
                                     varDict[X.columns.
                                             categories[X.columns.codes].values.tolist()[col]])
                except (KeyError) as err:
                    # Convert int column names to str
                    featNames.append(featMetaData[P]['label'] + ' - ' +
                                     str(X.columns.
                                         categories[X.columns.codes].values.tolist()[col]))
        elif featDataTypeDict[P] in ['continuous', 'ordinal']:
            X = np.array(rDat[P]).reshape(len(rDat[P]),1)
            # X = (X-min(X))/max(X) # Option to autoscale
            featNames.append(featMetaData[P]['label'])

        featNameIndex = featNameIndex + ([P]*len(range(X.shape[1])))  # Label for indexing in pandas export T

        # Save column indices of each P in dict for significance testing later...
        sigTestIdx[P] = range(feats.shape[1], feats.shape[1]+X.shape[1])
        # Append dummy encoded variable to exog array...
        feats = np.append(feats, X, axis=1)

        # If no evaluation partition is provided just use the whole dataset for training/eval
        trainIdx = np.linspace(0, len(outcome)-1, len(outcome)).astype(int)
        evalIdx = np.linspace(0, len(outcome)-1, len(outcome)).astype(int)

        # Construct Logistic model from all variable array...
        lr = Logit(endog=outcome[trainIdx], exog=feats[trainIdx])
        mdl = lr.fit(disp=0)

        # Export salient mdl features into table for writing...
        mdlExportT = pd.DataFrame(mdl.params, index=[[P]*len(featNames), featNames],
                                  columns=['coeff'])
        mdlExportT['coeffLCI'] = mdl.conf_int()[:, 0]
        mdlExportT['coeffUCI'] = mdl.conf_int()[:, 1]
        mdlExportT['OR'] = np.exp(mdl.params)
        mdlExportT['ORLCI'] = np.exp(mdl.conf_int())[:, 0]
        mdlExportT['ORUCI'] = np.exp(mdl.conf_int())[:, 1]
        mdlExportT['p'] = mdl.pvalues

        pValArray = [1]
        # Variable significance testing...
        testLr = Logit(endog=outcome, exog=np.delete(feats, sigTestIdx[P], axis=1))
        testMdl = testLr.fit(disp=0)
        Chi2p = 1 - stats.chi2.cdf(2*(mdl.llf - testMdl.llf), df=len(sigTestIdx[P]))
        pValArray = pValArray + [Chi2p]*len(sigTestIdx[P])
        mdlExportT['llrp'] = pValArray

        # Assess trained model predictive capacity
        trainTrue = outcome[trainIdx]
        trainScore = mdl.predict()
        evalTrue = outcome[evalIdx]
        evalScore = mdl.predict(feats[evalIdx])
        modelSummaryInfo.update(performanceMetrics([trainTrue, trainScore], [evalTrue, evalScore]))

        # Store common info and model objectin model summary output dict.
        modelSummaryInfo['nTotal'] = len(outcome)
        modelSummaryInfo['nTrain'] = len(trainIdx)
        modelSummaryInfo['nEval'] = len(evalIdx)
        modelSummaryInfo['partitionRatio'] = len(trainIdx)/(len(evalIdx)+len(trainIdx))
        modelSummaryInfo['outcomeVar'] = outcome
        modelSummaryInfo['outcomeLabels'] = featMetaData[outcomeVar]['truncvaluelabels']
        modelSummaryInfo['modelType'] = 'logisticRegression'
        modelSummaryInfo['featureSet'] = feats
        modelSummaryInfo['nFeatures'] = len(feats)

        # Add to array of univariate models for export.
        mdlExportTArray = mdlExportTArray + [mdlExportT]
        mdlArray = mdlArray + [mdl]
        modelSummaryInfoDict[P] = modelSummaryInfo

    UVMdlExportT = pd.concat(mdlExportTArray, axis=0)
    return UVMdlExportT, mdlArray, modelSummaryInfoDict

def MVLogisticRegression_v2(df, featureSet, outcomeVar, featMetaData, featDataTypeDict,
                            dummyExceptionDict, trainIdx=[], evalIdx=[]):
    """ DESCRIPTION NEEDED """
    # Exclude missing data from featureSet subset
    rDat = df.dropna(subset=featureSet + [outcomeVar])

    # Initialise feat & outcome arrays
    outcome = np.asarray(rDat[outcomeVar]).astype(int)  # Initialise outcome array, MUST BE BINARY
    feats = np.ones([len(rDat), 1]).astype(int)  # Initialise dummy feat array with constant
    featNames = ['constant']  # Initialise dummy featNames array with constant
    featNameIndex = ['constant']
    sigTestIdx = {}
    modelSummaryInfo = {}
    for P in featureSet:  # For each feature construct k-1 dummy array and add to feats array.
        if featDataTypeDict[P] in ['nominal', 'binary']:
            if rDat[P].dtype.name != 'category':  # If not a categorical then convert...
                rDat[P] = pd.Categorical(rDat[P])

            # Drop single category as constant.
            # Decision based on dummy exception dict, defaults to first category.
            try:
                X = pd.get_dummies(rDat[P], drop_first=False).drop(axis=1,
                                                                   columns=dummyExceptionDict[P])
            except (KeyError) as err:
                X = pd.get_dummies(rDat[P], drop_first=True)

            # Translate categorical series labels into SNSS var value labels..
            varDict = dict(zip(featMetaData[P]['values'],
                               featMetaData[P]['valuelabels']))
            for col in range(X.shape[1]):  # Construct featNames array for output
                try:
                    featNames.append(featMetaData[P]['label'] + ' - ' +
                                     varDict[X.columns.
                                             categories[X.columns.codes].values.tolist()[col]])
                except (KeyError) as err:
                    # Convert int column names to str
                    featNames.append(featMetaData[P]['label'] + ' - ' +
                                     str(X.columns.
                                         categories[X.columns.codes].values.tolist()[col]))
        elif featDataTypeDict[P] in ['continuous', 'ordinal']:
            X = np.array(rDat[P]).reshape(len(rDat[P]),1)
            # X = (X-min(X))/max(X) # Option to autoscale
            featNames.append(featMetaData[P]['label'])
        featNameIndex = featNameIndex + ([P]*len(range(X.shape[1])))  # Label for indexing in pandas export T

        # Save column indices of each P in dict for significance testing later...
        sigTestIdx[P] = range(feats.shape[1], feats.shape[1]+X.shape[1])
        # Append dummy encoded variable to exog array...
        feats = np.append(feats, X, axis=1)

    # If no evaluation partition is provided just use the whole dataset for training/eval
    if len(trainIdx) == 0:
        trainIdx = np.linspace(0, len(outcome)-1, len(outcome)).astype(int)
    if len(evalIdx) == 0:
        evalIdx = np.linspace(0, len(outcome)-1, len(outcome)).astype(int)
    # Construct Logistic model from all variable array...
    lr = Logit(endog=outcome[trainIdx], exog=feats[trainIdx, :])
    mdl = lr.fit(disp=0)

    # Variable significance testing...
    pValArray = [1]
    for P in featureSet:
        testLr = Logit(endog=outcome, exog=np.delete(feats, sigTestIdx[P], axis=1))
        testMdl = testLr.fit(disp=0)
        Chi2p = 1 - stats.chi2.cdf(2*(mdl.llf - testMdl.llf), df=len(sigTestIdx[P]))
        pValArray = pValArray + [Chi2p]*len(sigTestIdx[P])

    # Export salient mdl features into table for writing...
    mdlExportT = pd.DataFrame(mdl.params, index=[featNameIndex, featNames],
                              columns=['coeff'])
    mdlExportT['coeffLCI'] = mdl.conf_int()[:, 0]
    mdlExportT['coeffUCI'] = mdl.conf_int()[:, 1]
    mdlExportT['OR'] = np.exp(mdl.params)
    mdlExportT['ORLCI'] = np.exp(mdl.conf_int())[:, 0]
    mdlExportT['ORUCI'] = np.exp(mdl.conf_int())[:, 1]
    mdlExportT['p'] = mdl.pvalues
    mdlExportT['llrp'] = pValArray

    # Assess trained model predictive capacity
    trainTrue = outcome[trainIdx]
    trainScore = mdl.predict()
    evalTrue = outcome[evalIdx]
    evalScore = mdl.predict(feats[evalIdx])
    modelSummaryInfo.update(performanceMetrics([trainTrue, trainScore], [evalTrue, evalScore]))

    # Store common info and model objectin model summary output dict.
    modelSummaryInfo['nTotal'] = len(outcome)
    modelSummaryInfo['nTrain'] = len(trainIdx)
    modelSummaryInfo['nEval'] = len(evalIdx)
    modelSummaryInfo['partitionRatio'] = len(trainIdx)/(len(evalIdx)+len(trainIdx))
    modelSummaryInfo['outcomeVar'] = outcome
    modelSummaryInfo['outcomeLabels'] = featMetaData[outcomeVar]['truncvaluelabels']
    modelSummaryInfo['modelType'] = 'logisticRegression'
    modelSummaryInfo['featureSet'] = feats
    modelSummaryInfo['nFeatures'] = len(feats)
    return mdlExportT, mdl, modelSummaryInfo

def multiGroupLogisticRegression(df, featureSet, outcomeVar, featMetaData, featDataTypeDict, dummyExceptionDict,
                                 groupVar, multiGroupException, MV):
    groupVarDict = dict(zip(featMetaData[groupVar]['values'],
                            featMetaData[groupVar]['valuelabels']))
    byGroupLinearMdlExportT = {}
    byGroupLinearMdls = {}
    byGroupMSI = {}
    for G in df[groupVar].unique().astype(int):
        # If undertaking logistic regresion on subgroups and dummy variables change accordingly.
        dummyExceptionDict['Diagnosis'] = multiGroupException[G]
        if MV:
            mdlExportT, mdl, msi = MVLogisticRegression_v2(df=df[df[groupVar] == G],
                                                         featureSet=featureSet,
                                                         outcomeVar=outcomeVar,
                                                         featMetaData=featMetaData,
                                                         featDataTypeDict=featDataTypeDict,
                                                         dummyExceptionDict=dummyExceptionDict)
        else:
            mdlExportT, mdl, msi = UVLogisticRegression_v2(df=df[df[groupVar] == G],
                                                      featureSet=featureSet,
                                                      outcomeVar=outcomeVar,
                                                      featMetaData=featMetaData,
                                                      featDataTypeDict=featDataTypeDict,
                                                      dummyExceptionDict=dummyExceptionDict)
        byGroupLinearMdlExportT[groupVarDict[G]] = mdlExportT
        byGroupLinearMdls[groupVarDict[G]] = mdl
        byGroupMSI[groupVarDict[G]] = msi
    return byGroupLinearMdlExportT, byGroupLinearMdls, byGroupMSI


def logitCoeffForestPlot(predictorTables, mdl, tag, groupVar, returnPlot=False):
    """ Forest plot of univariate coefficients """
    # Takes a table of regression coefficients arranged in a dictionary by group and with:
    # values for: coeff, coeffLCI, coeffUCI, OR, ORLCI, ORUCI, p

    #  Rough adaptation of figure size accoridng to number of variables.
    figHeight = round(predictorTables[list(predictorTables.keys())[0]].shape[0]/3.3)
    fig = plt.figure(num=1, figsize=(5, figHeight), dpi=200, frameon=False)
    ax = fig.add_subplot(1, 1, 1)
    yLabels = []
    yTicks = []
    legendLines = []
    legendStr = []
    # Calculate spacing requirements..
    groupBuffer = 0.3
    nGroups = len(predictorTables)
    varBuffer = groupBuffer * (nGroups-1)
    GTick = np.linspace(start=0, stop=varBuffer, num=nGroups)-(0.5*varBuffer)

    # Set colour scheme:
    if groupVar == 'ExpGroups':
        pltCols = [Color('#950000'), Color('#00336E')]
    else:
        pltCols = [Color('#42f4b3'), Color('#4f1260'),
                   Color('#273e89'), Color('#3f6334'),
                   Color('#797f26'), Color('#b21a78')]

    g = 0
    for G in predictorTables.keys():
        print('Plotting ' + G)
        if 'Not Explained' in G:
            pltCols = [Color('#950000'), Color('#950000')]
        else:
            pltCols = [Color('#00336E'), Color('#00336E')]
        i = 0
        T = predictorTables[G]
        for var in T.iterrows():
            label = str(var[0][1])
            if 'constant' in label:
                varCol = [0.8, 0.8, 0.8]
                i = i-1
            else:
                varCol = pltCols[g].rgb

            yLabels.append(label)
            y = GTick[g] + i
            yTicks.append(y)
            ax.plot(var[1].coeff, y,
                    ls='None', marker='.', ms=4, color=varCol)
            ax.text(var[1].coeff, y, str(np.round(var[1].OR, 2)), fontsize=6)
            ax.plot([var[1].coeffLCI, var[1].coeffUCI], [y]*2,
                    ls='solid', lw=1.5, marker='None', color=varCol)
            i = i-nGroups/2
        legendLines.append(Line2D([0], [0], color=pltCols[g].rgb, lw=1.5))
        legendStr.append(G)
        g = g+1

    # Plot 0 Reference line...
    ax.plot([0, 0], [5, -800], ls='--', lw=0.5,
            marker='None', color=[0.8, 0.8, 0.8],
            alpha=0.9)
    ax.set_ylim([min(yTicks)-1, 1])

    ax.set_yticks(yTicks[0:int(len(yTicks)/nGroups)])
    ax.set_yticklabels(yLabels[0:int(len(yLabels)/nGroups)],
                       fontsize=6)
    # Add title
    fig.suptitle('SNSS Regression coefficients for ' + tag[2] + ' by ' + groupVar, fontsize=12)
    # Add legend
    ax.legend(legendLines, legendStr,
              bbox_to_anchor=(1.1, 1), loc=1)

    fig.savefig('output/' + tag[0] + '_' + tag[1] +
                'PredictorsForestPlot' + tag[2] + '.pdf', dpi=200,
                format='pdf', pad_inches=0.1, bbox_inches='tight')
    plt.close()
    if returnPlot:
        return fig
    else:
        print('Plot Saved')


def SNSSSecondaryMeasuresvsPrimaryOutcome(df):
    """Speculative scatter plots assessing correlation between
    reported CGI and IPS and reported SF12 HADS and symptom counts"""
    outcomes = ['T0_PHQNeuro28_Total',
                'T0_HADS',
                'T0_SF12_NormedMCS',
                'T0_SF12_NormedPCS',
                'T0_SIMD04_score',
                'T2_poorCGI']

    # plt.ioff()
    # fig1 = sb.pairplot(df[outcomes][df.ExpGroups == 1].dropna(),
    #                    diag_kind='kde',
    #                    hue='T2_poorCGI',
    #                    palette=sb.cubehelix_palette(2, start=0, reverse=True),
    #                    kind='scatter', plot_kws={"marker": '.', "s": 5, "edgecolors": None})
    # fig1.savefig('output/self_report_outcomes/1_FunctionalSecondaryOutcomesPairPlot.pdf', dpi=300,
    #              format='pdf', pad_inches=0.1, bbox_inches='tight')
    # plt.close()
    # plt.ioff()
    # fig2 = sb.pairplot(df[outcomes][df.ExpGroups == 2].dropna(),
    #                    diag_kind='kde',
    #                    hue='T2_poorCGI',
    #                    palette=sb.cubehelix_palette(2, start=2, reverse=True),
    #                    kind='scatter', plot_kws={"marker": '.', "s": 5, "edgecolors": None})
    # fig2.savefig('output/self_report_outcomes/1_StructuralSecondaryOutcomesPairPlot.pdf', dpi=300,
    #              format='pdf', pad_inches=0.1, bbox_inches='tight')
    # plt.close()
    plt.ioff()
    fig3 = sb.pairplot(df[outcomes].dropna(),
                       diag_kind='kde',
                       hue='T2_poorCGI',
                       kind='scatter', plot_kws={"marker": '.', "s": 8, "edgecolors": None},
                       corner=True)
    fig3.savefig('output/self_report_outcomes/1_SecondaryOutcomesPairPlot.pdf', dpi=300,
                 format='pdf', pad_inches=0.1, bbox_inches='tight')
    plt.close()
    return


def NNDummyEncoder(rawFeatures, outcome, df, featMetaData, k_dummy=False):
    rDat = df[df[outcome].notna()].dropna(subset=rawFeatures)
    # Drop missing observations from relevant subset.
    rDat = rDat.dropna(subset=rawFeatures)

    # Assign outcome array.
    if len(np.unique(rDat[outcome])) == 2:
        y = np.asarray(rDat[outcome]).astype(int)
    else:
        y = pd.get_dummies(rDat[outcome], drop_first=False)

    # Initialise arrays
    x = np.ones([len(rDat), 1]).astype(int)
    varLabels = []
    varNames = []

    if k_dummy:
        # For every variable dummy encode and append to feature array.
        for P in rawFeatures:
            varDat = rDat[P]  # Select data for encoding.
            if varDat.dtype.name != 'category':  # If not a categorical, convert...
                varDat = pd.Categorical(varDat)

            # Convert to k-dummy matrix..
            X = pd.get_dummies(varDat, drop_first=False)

            # Translate categorical series labels into SNSS var value labels..
            varDict = dict(zip(featMetaData[P]['values'],
                               featMetaData[P]['valuelabels']))

            # For each new dummy assign label, some cat labels amenable to dictionary
            # assigment others are not so include exception..
            vn = 1
            for col in range(X.shape[1]):
                try:
                    varLabels.append(featMetaData[P]['label'] + ' - ' +
                                     varDict[X.columns.
                                             categories[X.columns.codes].values.tolist()[col]])
                    varNames.append(P + '_' + str(vn))
                except (KeyError) as err:
                    varLabels.append(featMetaData[P]['label'] + ' - ' +
                                     str(X.columns.
                                         categories[X.columns.codes].values.tolist()[col]))
                    varNames.append(P + '_' + str(vn))
                vn += 1
            # Append dummy encoded variable to x array...
            x = np.append(x, X, axis=1)
    else:
        for P in rawFeatures:
            varDat = rDat[P]  # Select data for encoding.
             # As data is no longer dummy and unit variance, normalise. Not necessarily required for all NN applicaitons
             # in this case use of the SELY activation functino warrants a general usage.
            if varDat.dtype.name == 'category':  # If a categorical string, convert...
                varDat=varDat.cat.codes
            X = normalise(np.array(varDat))
            varLabels.append('')
            varNames.append(P)
            x = np.append(x, X.reshape(-1,1), axis=1)

    # Delete constant
    x = np.delete(x, 0, 1)
    return y, x, varNames, varLabels


def NNStructurePlot(kerasModel):
    fig = plt.figure(num=1, figsize=(6, 6), dpi=200, frameon=False)
    ax = fig.add_subplot(1, 1, 1)

    i = 0
    for L in kerasModel.layers:
        if i == 0:
            print('Input: ' + str(L.input_shape[1]))
        print('\nLayer: ' + str(i))
        if i == len(kerasModel.layers)-1:
            print('(Output Layer)')
        print('Activation Fx: ' + L.activation.__name__)
        print('Size: ' + str(L.output_shape[1]))

        print('Plotting...')

        bias = L.get_weights()[1]
        ii = 0
        iinorm = -1*(len(L.get_weights()[0]))/2
        for input in L.get_weights()[0]:
            wi = 0
            winorm = -1*(len(input))/2
            if i == 0:
                ax.text(i-0.05, iinorm+ii, varNames[ii],
                        fontsize=3, ha='right', va='center')
            for w in input:
                # print(w)
                if w < 0:
                    wcol = 'xkcd:azure'
                elif w > 0:
                    wcol = 'xkcd:light red'
                else:
                    wcol = 'xkcd:grey'
                ax.plot([i, i+1], [iinorm+ii, winorm+wi], linestyle='-',
                        c=wcol, linewidth=(w*w)/2, alpha=logistic.cdf(abs(w))-0.3)
                wi += 1
            ii += 1
        i += 1
    ax.axis('off')
    fig.savefig('output/6_' + tag + '_modelStructure.pdf', dpi=300,
                format='pdf', pad_inches=0.1, bbox_inches='tight')
    plt.close()


def hiddenLayerNGen(nHiddenLayers, nFeats, convergeFun, minLayerSize, maxLayerSize, nOutput):
    if (convergeFun == 'linDesc'):
        maxCoeff = 0.8
        c = min([maxLayerSize, (maxCoeff*nFeats)])
        grad = (minLayerSize-c)/(nHiddenLayers-1)
        layerSizes = [nFeats] + list(map(lambda x: round(grad*x + c, 0),
                                         list(range(nHiddenLayers)))) + [nOutput]
    elif (convergeFun == 'const'):
        maxCoeff = 0.5
        c = min([maxLayerSize, (maxCoeff*nFeats)])
        layerSizes = [nFeats] + list(map(lambda x: round(0*x + c, 0),
                                         list(range(nHiddenLayers)))) + [nOutput]
    elif (convergeFun == 'stepDesc'):
        maxCoeff = 0.5
        L1 = min([0.5*nFeats, maxLayerSize])
        layerSizes = [nFeats] + [L1] + [20]*(nHiddenLayers-2) + [minLayerSize] + [nOutput]
    return layerSizes


def MLPAnalysis(df, features, outcome, featMetaData, params):
    verbose = 0
    modelSummaryInfo = {}
    # Extract key model features from param dict
    cpDir = 'output/NNoutput/modelCheckpoints/'
    nHiddenLayers = params['nHiddenLayers']
    maxHiddenLayerSize = params['maxHiddenLayerSize']
    modelType = params['modelType']
    nEpochs = params['nEpochs']
    dropoutP = params['dropoutP']
    batchSize = params['batchSize']
    k_dummy = params['k_dummy']
    storePredictHist = params['storePredictHist']
    mdlStr = params['mdlStr']
    # classWeights = params['classWeights']
    # Limit dataset to obs with complete predictor sets
    NNdf = df[df[outcome].notna()]
    # Determine no. of required output units for task.
    outcomeLabels = np.unique(NNdf[outcome])
    if len(outcomeLabels) == 2:
        nOutput = 1
        outputFun = 'sigmoid'
        lossFun = 'binary_crossentropy'
        metrics = ['binary_accuracy']
    else:
        nOutput = len(outcomeLabels)
        outputFun = 'softmax'
        lossFun = 'categorical_crossentropy'
        top2_acc = functools.partial(top_k_categorical_accuracy, k=2)
        top2_acc.__name__ = 'top2_acc'
        metrics = ['categorical_accuracy']

    # Data Preparation: Set up random train & test partition
    # Compile list of all relevant variables.
    # varGroups.keys()

    # Encode relevent features in k-dummy format and return label array. (NOTE not k-1)
    endog, exog, varNames, varLabels =\
        NNDummyEncoder(df=NNdf, rawFeatures=features,
                       featMetaData=featMetaData,
                       outcome=outcome,
                       k_dummy=k_dummy)

    if verbose:
        print('Classification Task:')
        print(modelType + ' network predicting ' + outcome + ' from ' + str(NNdf.shape[0]) +
              ' observations on ' + str(len(features)) + ' features.')
        print('Dummy encoding categorical features:')
        print(str(len(features)) + ' categorical features encoded into ' +
              str(exog.shape[1]) + ' dummy variables.')
    # Assign feature and label arrays.
    allFeatures = pd.DataFrame(data=exog, columns=varNames)
    allLabels = pd.DataFrame(data=endog)

    # Partition data into training and evaluation sets.
    if len(params['trainIdx']) == 0:
        partitionRatio = params['partitionRatio']
        seed = 5
        print('No partition index provided \n \
              Creating new partition with seed '+str(seed)+'...')
        rand = np.random.RandomState(seed=seed)
        randIdx = rand.randint(0, len(allFeatures), len(allFeatures))
        trainIdx = randIdx[0:round(len(allFeatures)*partitionRatio)]
        evalIdx = randIdx[round(len(allFeatures)*(partitionRatio)):len(randIdx)]
    else:
        trainIdx = params['trainIdx']
        evalIdx = params['evalIdx']
        partitionRatio = params['partitionRatio']

    # Assign feature and label arrays based on partiion index
    trainFeats = allFeatures.iloc[trainIdx].values
    trainLabels = allLabels.iloc[trainIdx].values
    evalFeats = allFeatures.iloc[evalIdx].values
    evalLabels = allLabels.iloc[evalIdx].values
    nFeats = trainFeats.shape[1]

    # Keras Model Definition
    nLayers = nHiddenLayers + 2  # No. of layers derived from hidden
    kerasModel = Sequential()
    if modelType == 'selfNormalising':
        dropoutFun = AlphaDropout(dropoutP)
        modelSummaryInfo['dropoutFun'] = 'AlphaDropout'
        modelSummaryInfo['dropoutP'] = dropoutP
        layerSizes = hiddenLayerNGen(nHiddenLayers, nFeats=nFeats,
                                     convergeFun='const', minLayerSize=20,
                                     maxLayerSize=maxHiddenLayerSize, nOutput=nOutput)
        # print(layerSizes)
        for L in range(0, nLayers):
            if L == 0:
                '''Input layer added as part of layer 1 syntax'''
                # print('Adding Input Layer of size: ' + str(nFeats))
            elif L == 1:
                # print('Adding Hidden Layer ' + str(L) + ' of size: ' + str(layerSizes[L]))
                kerasModel.add(Dense(units=layerSizes[L], input_shape=(nFeats,), activation='selu',
                                     kernel_initializer='lecun_normal'))
                kerasModel.add(dropoutFun)
            elif L == nLayers - 1:
                # print('Adding Output Layer of size: ' + str(nOutput))
                kerasModel.add(Dense(units=nOutput, activation=outputFun))
            else:
                # print('Adding Hidden Layer ' + str(L) + ' of size: ' + str(layerSizes[L]))
                kerasModel.add(Dense(units=layerSizes[L], activation='selu',
                               kernel_initializer='lecun_normal'))
                kerasModel.add(dropoutFun)

    elif modelType == 'deepFeedForwardLeakyReLU':
        dropoutFun = Dropout(dropoutP)
        modelSummaryInfo['dropoutFun'] = 'Dropout'
        modelSummaryInfo['dropoutP'] = dropoutP
        layerSizes = hiddenLayerNGen(nHiddenLayers, nFeats=nFeats,
                                     convergeFun='const', minLayerSize=20,
                                     maxLayerSize=maxHiddenLayerSize, nOutput=nOutput)
        for L in range(0, nLayers):
            if L == 0:
                '''Input layer added as part of layer 1 syntax'''
                # print('Adding Input Layer of size: ' + str(nFeats))
            elif L == 1:
                # print('Adding Hidden Layer ' + str(L) + ' of size: ' + str(layerSizes[L]))
                kerasModel.add(Dense(units=layerSizes[L], input_shape=(nFeats,), activation='linear',
                                     kernel_initializer='TruncatedNormal'))
                kerasModel.add(LeakyReLU(alpha=0.3))
                kerasModel.add(dropoutFun)
            elif L == nLayers - 1:
                # print('Adding Output Layer of size: ' + str(nOutput))
                kerasModel.add(Dense(units=nOutput, activation=outputFun))
            else:
                # print('Adding Hidden Layer ' + str(L) + ' of size: ' + str(layerSizes[L]))
                kerasModel.add(Dense(units=layerSizes[L], activation='linear',
                                     kernel_initializer='TruncatedNormal'))
                kerasModel.add(LeakyReLU(alpha=0.3))
                kerasModel.add(dropoutFun)

    elif modelType == 'deepFeedForwardTanH':
        dropoutFun = Dropout(dropoutP)
        modelSummaryInfo['dropoutFun'] = 'Dropout'
        modelSummaryInfo['dropoutP'] = dropoutP
        layerSizes = hiddenLayerNGen(nHiddenLayers, nFeats=nFeats,
                                     convergeFun='const', minLayerSize=20,
                                     maxLayerSize=maxHiddenLayerSize, nOutput=nOutput)
        for L in range(0, nLayers):
            if L == 0:
                '''Input layer added as part of layer 1 syntax'''
                # print('Adding Input Layer of size: ' + str(nFeats))
            elif L == 1:
                # print('Adding Hidden Layer ' + str(L) + ' of size: ' + str(layerSizes[L]))
                kerasModel.add(Dense(units=layerSizes[L], input_shape=(nFeats,), activation='tanh',
                                     kernel_initializer='TruncatedNormal'))
                kerasModel.add(dropoutFun)
            elif L == nLayers - 1:
                # print('Adding Output Layer of size: ' + str(nOutput))
                kerasModel.add(Dense(units=nOutput, activation=outputFun))
            else:
                # print('Adding Hidden Layer ' + str(L) + ' of size: ' + str(layerSizes[L]))
                kerasModel.add(Dense(units=layerSizes[L], activation='tanh',
                                     kernel_initializer='TruncatedNormal'))
                kerasModel.add(dropoutFun)

    elif modelType == 'logisticRegressionNN':
        layerSizes = np.nan
        kerasModel.add(Dense(units=nOutput, input_shape=(nFeats,), activation=outputFun))

    # csv_logger = CSVLogger('log_dir/training.csv', separator=',', append=False)
    callbacks = []

    if storePredictHist:
        class prediction_history(Callback):
            def __init__(self):
                self.trainPredHist = []
                self.evalPredHist = []

            def on_epoch_end(self, epoch, logs={}):
                self.trainPredHist.append(kerasModel.predict(trainFeats))
                self.evalPredHist.append(kerasModel.predict(evalFeats))
        predictions = prediction_history()
        callbacks.append(predictions)

    if nHiddenLayers > 3:
        patience = 3500
    else:
        patience = 1000

    stoppingMetric = 'val_loss'  #val_loss
    earlyStop = EarlyStopping(monitor=stoppingMetric, mode='min', verbose=1, patience=patience)
    callbacks.append(earlyStop)

    modelCheckpoint = ModelCheckpoint(cpDir+mdlStr+'.h5', monitor=stoppingMetric, mode='min', save_best_only=True, verbose=0)
    callbacks.append(modelCheckpoint)

    kerasModel.compile(optimizer='sgd', loss=lossFun, metrics=metrics)
    # kerasModel.metrics_tensors += kerasModel.outputs
    # kerasModel.metrics_names += 'predictions'

    startTime = datetime.now()
    # batch_sizes: len(trainLabels), 1, 16, 32
    history = kerasModel.fit(trainFeats, trainLabels, batch_size=batchSize, epochs=nEpochs,
                             verbose=0, validation_data=(evalFeats, evalLabels),
                             shuffle=True, steps_per_epoch=None, validation_steps=None,
                             callbacks=callbacks)
    # classWeights=classWeights #For inclusion in above fit if needed
    stopTime = datetime.now()
    # Calculate time taken for training.
    timeElapsed = stopTime-startTime
    timeElapsedList = list(map(int, divmod(timeElapsed.total_seconds(), 60)))

    # Compute sigmoid output scores for train and eval datasets.
    trainScore = kerasModel.predict(trainFeats, batch_size=None, verbose=0, steps=None)
    evalScore = kerasModel.predict(evalFeats, batch_size=None, verbose=0, steps=None)

    # Reshape outputs for performance assessment
    if nOutput == 1:
        trainTrue = np.reshape(trainLabels, [len(trainLabels), ])
        trainScore = np.reshape(trainScore, [len(trainScore), ])
        evalTrue = np.reshape(evalLabels, [len(evalLabels), ])
        evalScore = np.reshape(evalScore, [len(evalScore), ])
    else:
        trainTrue = np.reshape(np.argmax(trainLabels, axis=1), [len(trainLabels), ])
        evalTrue = np.reshape(np.argmax(evalLabels, axis=1), [len(evalLabels), ])
    modelSummaryInfo.update(performanceMetrics([trainTrue, trainScore], [evalTrue, evalScore]))

    # Store Input data
    modelSummaryInfo['trainInput'] = trainFeats
    modelSummaryInfo['evalInput'] = evalFeats

    # Store common info and model objectin model summary output dict.
    modelSummaryInfo['nTotal'] = len(allLabels)
    modelSummaryInfo['nTrain'] = len(trainLabels)
    modelSummaryInfo['nEval'] = len(evalLabels)
    modelSummaryInfo['partitionRatio'] = partitionRatio
    modelSummaryInfo['outcomeVar'] = outcome
    modelSummaryInfo['outcomeLabels'] = featMetaData[outcome]['truncvaluelabels']
    modelSummaryInfo['modelType'] = modelType
    modelSummaryInfo['featureSet'] = str(features)
    modelSummaryInfo['nFeatures'] = len(features)
    modelSummaryInfo['nInputs'] = nFeats
    modelSummaryInfo['nOutputs'] = nOutput
    modelSummaryInfo['nLayers'] = nLayers
    modelSummaryInfo['nUnits'] = layerSizes
    modelSummaryInfo['nEpochs'] = nEpochs
    modelSummaryInfo['timeToTrain'] = timeElapsedList
    modelSummaryInfo['lossFun'] = lossFun
    modelSummaryInfo['kerasHistory'] = history
    modelSummaryInfo['kerasModelObject'] = kerasModel
    modelSummaryInfo['paramDict'] = params
    modelSummaryInfo['mdlStr'] = mdlStr
    if storePredictHist:
        modelSummaryInfo['kerasTrainPredictionHistory'] = predictions.trainPredHist
        modelSummaryInfo['kerasEvalPredictionHistory'] = predictions.evalPredHist

    if verbose:
        print('\nNN training and preliminary evaluation complete!\n')
    clear_session()
    return modelSummaryInfo


def MLPHyperParameterSweep(problemDict, hpDict, featMetaData):
    # Extract problem definition from problem dict.
    featureSet = problemDict['featureSet']['value']
    outcome = problemDict['outcome']['value']
    df = problemDict['df']['value']

    # Calculate total number of hyperparamter combinations from hpDict.
    totalHpParams = np.prod([len(hpDict[k]['value']) for k in hpDict.keys()])
    args = [hpDict[k]['value'] for k in hpDict.keys()]

    modelDict = {}
    print('Beginning parameter sweep with '+str(totalHpParams)+' combination(s)...')
    with tqdm.tqdm(total=totalHpParams) as pbar:
        for HpComb in itertools.product(*args):
            # Drop Nas from feature set and outcome
            df = df.dropna(subset=featureSet + [outcome])
            # State model identifier string.
            mdlStr = '_'.join([problemDict['df']['label'],
                               problemDict['featureSet']['label'],
                               problemDict['outcome']['label']]+[str(s) for s in HpComb])
    # Partition Data (can be done randomely by MLP function but importnatn for consistancy here.)
            trainIdx, evalIdx = partitionData(df,
                                    partitionRatio=HpComb[hpDict['partitionRatioArray']['order']])

            NNModelParams = {'modelType': HpComb[hpDict['modelTypeArray']['order']],
                             'partitionRatio': HpComb[hpDict['partitionRatioArray']['order']],
                             'nEpochs': HpComb[hpDict['nEpochArray']['order']],
                             'nHiddenLayers': HpComb[hpDict['nHiddenLayerArray']['order']],
                             'maxHiddenLayerSize': HpComb[hpDict['maxHiddenLayerSizeArray']['order']],
                             'trainIdx': trainIdx,
                             'evalIdx': evalIdx,
                             'dropoutP': 0.5,
                             'batchSize': 128,
                             'k_dummy': True,
                             'storePredictHist': False,
                             'mdlStr': mdlStr}
            mdl = MLPAnalysis(df=df, features=featureSet,
                              outcome=outcome,
                              featMetaData=featMetaData,
                              params=NNModelParams)

            # Add identifying information to modelDict before return
            mdl['outcomeLabels'] = featMetaData[problemDict['outcome']['label']]['truncvaluelabels']
            mdl['featureSetTitle'] = problemDict['featureSet']['label']
            mdl['group'] = problemDict['df']['label']
            NNModelSummaryPlot(mdl, dir='output/NNoutput/paramSweepsModels/')
            # Add model summary info to model dict.
            modelDict[mdlStr] = mdl
            # Update progress bar.
            pbar.update(1)
    # pckl_out = open('output/NNoutput/paramSweeps/paramSweep_' +
    #                 datetime.now().strftime('%d-%m-%Y %H%M') +
    #                 '.pickle', 'wb')
    # pickle.dump(modelDict, pckl_out)
    # pckl_out.close()
    return modelDict


def kFoldPartition(df, k):
    randIdx = np.linspace(0, df.shape[0]-1, df.shape[0]).astype(int)
    np.random.shuffle(randIdx)
    slices = (len(randIdx)*np.linspace(0, 1, k+1)).astype(int)
    folds = []
    for i in range(len(slices)-1):
        evalSlice = range(slices[i], slices[i+1])
        evalIdx = randIdx[evalSlice]
        trainSlice = np.invert(np.in1d(randIdx, evalIdx))
        trainIdx = randIdx[trainSlice]
        folds.append([trainIdx, evalIdx])
    return folds


def crossValidatedMLPAnalysis(problemDict, paramDict, featMetaData, k=10):
    featureSet = problemDict['featureSet']['value']
    outcome = problemDict['outcome']['value']
    df = problemDict['df']['value']
    df = df.dropna(subset=featureSet + [outcome])

    modelDict = {}

    folds = kFoldPartition(df, k=k)
    with tqdm.tqdm(total=len(folds)) as pbar:
        for i, fold in enumerate(folds):
            paramDict['trainIdx'] = fold[0]
            paramDict['evalIdx'] = fold[1]
            # State model identifier string.
            mdlStr = '_'.join([problemDict['df']['label'],
                               problemDict['featureSet']['label'],
                               problemDict['outcome']['label']]+['_CV_'+str(i)])
            paramDict['mdlStr'] = mdlStr
            # paramDict['classWeights'] = {0: weight_for_0, 1: weight_for_1} #?important
            mdl = MLPAnalysis(df=df, features=featureSet,
                              outcome=outcome, featMetaData=featMetaData,
                              params=paramDict)
            mdl['outcomeLabels'] = featMetaData[problemDict['outcome']['label']]['truncvaluelabels']
            mdl['featureSetTitle'] = problemDict['featureSet']['label']
            mdl['group'] = problemDict['df']['label']
            NNModelSummaryPlot(mdl, dir='output/NNoutput/cvModels/')
            # Add model summary info to model dict.
            modelDict[mdlStr] = mdl
            # Update progress bar.
            pbar.update(1)
    return modelDict


def NNModelSummaryPlot(msi, dir='output/NNoutput/'):
    # NNSummary Plot
    def moving_average(signal, period):
        buffer = [np.nan] * period
        for i in range(period, len(signal)):
            buffer.append(np.mean(signal[i-period:i]))
        return buffer

    figParams = {'num': 1,
                 'figsize': (24, 10),
                 'dpi': 100,
                 'frameon': False}
    textParams = {'fontsize': 8,
                  'color': 'xkcd:black',
                  'verticalalignment': 'top',
                  'horizontalalignment': 'left'}
    fig = plt.figure(**figParams)
    # imp.reload(plt)
    # Subplot 1: Model Details
    subPltI = 1
    txtSumm = fig.add_subplot(1, 5, subPltI)
    strArray = 'Diagnostic Class: '+msi['group']+'. ' +\
               'Outcome Variable: '+msi['outcomeVar']+'. ' +\
               'N: '+str(msi['nTotal'])+'. ' +\
               'Partition Ratio: '+str(msi['partitionRatio'])+'. ' +\
               'Training N: '+str(msi['nTrain'])+'. ' +\
               'Evaluation N: '+str(msi['nEval'])+'. ' +\
               'Model Type: '+msi['modelType']+'. ' +\
               'N Features: '+str(msi['nFeatures'])+'. ' +\
               'N Input Units: ' + str(msi['nInputs'])+'. ' +\
               'N Layers: '+str(msi['nLayers']-2)+'. ' +\
               'N Units in each Layer' + str(msi['nUnits'])+'. ' +\
               'N Epochs: '+str(msi['nEpochs'])+'. ' +\
               'Time Taken to Train: '+str(msi['timeToTrain'][0])+' min(s) ' +\
               str(msi['timeToTrain'][1])+' sec(s). '

    if msi['nOutputs'] == 1:
        # Add binary measures of performanceMetrics
        strArray = strArray + 'Binary Accuracy: '+str(msi['evalAcc'])+'. ' +\
                              'Binary AUROC: '+str(msi['evalAuroc'])+'. ' +\
                              'Binary R2: '+str(msi['evalR2'])+'. ' +\
                              'Binary Precision: '+str(msi['evalPrecision'])+'. ' +\
                              'Binary Recall: '+str(msi['evalRecall'])+'. ' +\
                              'Binary Sensitivity: '+str(msi['evalSens'])+'. ' +\
                              'Binary Specificity: '+str(msi['evalSpec'])+'. ' +\
                              'Binary Optimal Operating Point: '+str(msi['evalOperatingThreshold'])+'. '
    else:
        strArray = strArray + 'Multiclass Accuracy: '+str(msi['evalAcc'])+'. ' +\
                              'Multiclass Precision: '+str(msi['evalPrecision'])+'. ' +\
                              'Multiclass Recall: '+str(msi['evalRecall'])+'. '
    strArray = txtwrp.fill(strArray+'Feature Set: '+str(msi['featureSet']), 40)

    txtSumm.text(0, 0, strArray, **textParams)
    txtSumm.set_ylim([-20, 0])
    txtSumm.set_axis_off()
    subPltI += 1
    # Subplot 4/5: Training History
    if msi['nOutputs'] == 1:
        histKeys = ['binary_accuracy', 'loss',
                    'val_binary_accuracy', 'val_loss']
    else:
        histKeys = ['categorical_accuracy', 'loss',
                    'val_categorical_accuracy', 'val_loss']
    titles = ['Test Accuracy', 'Test '+msi['lossFun'],
              'Evaluation Accuracy', 'Evaluation '+msi['lossFun']]
    metricCol = ['xkcd:aqua green', 'xkcd:greyish green',
                 'xkcd:light magenta', 'xkcd:purplish pink']
    i = 0
    for metric in histKeys:
        histAx = fig.add_subplot(2, 5, subPltI)
        histAx.grid(b=None, which='major', axis='both', alpha=0.6)
        y = msi['kerasHistory'].history[metric]
        x = range(len(y))
        histAx.plot(x, y, c=metricCol[i], linewidth=0.2, alpha=0.5)
        smoothY = moving_average(y, 10)
        smoothX = range(len(smoothY))
        histAx.plot(smoothX, smoothY, c=metricCol[i], linewidth=1, alpha=1)
        histAx.plot([0, max(x)], [0.5, 0.5], linestyle='--',
                    c='xkcd:light grey', linewidth=0.2)
        histAx.set_ylim((0, max([1, max(y)])))
        histAx.set_xlim((0, msi['nEpochs']))
        histAx.set_title(titles[i])
        histAx.set_ylabel(metric, fontsize=10)
        histAx.set_xlabel('Training Epoch', fontsize=10)
        if 'accuracy' in metric:
            maxAcc = str(round(np.nanmax(y), 2))
            histAx.scatter(x[y.index(np.nanmax(y))], np.nanmax(y), marker='o', facecolors='None', edgecolors='xkcd:black')
            histAx.text(x[y.index(np.nanmax(y))]+0.06, np.nanmax(y)-0.06, 'Max = '+maxAcc, fontsize=8)
        elif 'loss' in metric:
            minLoss = str(round(np.nanmin(y), 2))
            histAx.scatter(x[y.index(np.nanmin(y))], np.nanmin(y), marker='o', facecolors='None', edgecolors='xkcd:black')
            histAx.text(x[y.index(np.nanmin(y))]+0.06, np.nanmin(y)-0.06, 'Min = '+minLoss, fontsize=8)
        i += 1
        subPltI += 1
    subPltI += 1

    if msi['nOutputs'] == 1:
        # Subplot 2: Histogram
        ax2 = fig.add_subplot(2, 5, subPltI)
        for g in np.unique(msi['evalTrue']):
            ax2.hist(msi['evalScore'][msi['evalTrue'] == g], density=False, alpha=0.5)
            # ax2.hist([s[g] for s in msi['evalScore'][msi['evalTrue'] == g]], bins=20, range=(0, 1),
            #          density=False, alpha=0.5)
        ax2.set_xlabel('NN Score', fontsize=10)
        ax2.set_ylabel('f', fontsize=10)
        ax2.set_title('Histogram of NN Eval Outputs')
        ax2.tick_params(axis='both', labelsize=10)
        subPltI += 1

        # Subplot 3: ROC Curve
        ax3 = fig.add_subplot(2, 5, subPltI)
        x, y, t = roc_curve(msi['evalTrue'], msi['evalScore'])
        ax3.plot(x, y, linewidth=1)
        ax3.plot([0, 1], [0, 1], linestyle='--',
                 c='xkcd:light grey', linewidth=1)
        ax3.set_xlabel('1 - Sensitivity', fontsize=10)
        ax3.set_ylabel('Specificity', fontsize=10)
        ax3.set_ylim((0, 1))
        ax3.set_xlim((0, 1))
        ax3.set_title('ROC of NN Eval Scores')
        ax3.text(0.5, 0.3, 'AUC: ' + str(round(msi['evalAuroc'], 2)), fontsize=10)
        ax3.tick_params(axis='both', labelsize=10)
        subPltI += 1

    # Subplot 3: Confusion matrix heatmap
    ax4 = fig.add_subplot(2, 5, subPltI)
    CMNorm = round(pd.crosstab(msi['evalTrue'], msi['evalPred'], normalize='index'), 2)
    CM = pd.crosstab(msi['evalTrue'], msi['evalPred'], normalize=False)
    for y in range(CM.shape[0]):
        for x in range(CM.shape[1]):
            ax4.text(x+0.5, -y-0.5, str(CM.iloc[y, x]), verticalalignment='center',
                     horizontalalignment='center', fontsize=8)
            if x == y:
                patchCol = sb.color_palette("Blues", 101)[int(CMNorm.iloc[y, x]*100)]
            else:
                patchCol = sb.color_palette("Reds", 101)[int(CMNorm.iloc[y, x]*100)]
            ax4.add_patch(patches.Rectangle((x, -y-1), 1, 1, color=patchCol))
    ax4.text((x+1)*0.5, (-y-1)-0.5, 'Predicted Class', horizontalalignment='center', fontsize=10)
    ax4.text(-0.5, (-y-1)*0.5, 'True Class', horizontalalignment='center',
             verticalalignment='center', rotation=90, fontsize=10)
    ax4.set_xticks([x+0.5 for x in range(x+1)])
    ax4.set_yticks([-y-0.5 for y in range(y+1)])
    ax4.set_xticklabels(msi['outcomeLabels'], rotation=45, ha='right', fontsize=10)
    ax4.set_yticklabels(msi['outcomeLabels'], rotation=45, fontsize=10)
    # ax4.xaxis.tick_top()
    ax4.set_xlim([0, x+1])
    ax4.set_ylim([-y-1, 0])
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['bottom'].set_visible(False)
    ax4.spines['left'].set_visible(False)
    ax4.tick_params(axis='both', pad=0.1)

    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.close()
    dirStr = dir
    if not os.path.exists(dirStr):
        os.makedirs(dirStr)

    idStr = '_'.join(['6']+[msi['mdlStr']]+['summary.pdf'])
    fig.savefig(dirStr + idStr, dpi=200, format='pdf', pad_inches=0.1, bbox_inches='tight')

    # # Plot Keras output model structure
    # plot_model(msi['kerasModelObject'], show_shapes=True, show_layer_names=True,
    #            to_file='output/NNoutput/' + msi['outcomeVar'] + '/' + msi['featureSetTitle'] + '/' +
    #            msi['modelType'] + '/6_' + msi['groupVar'] + '_' + msi['featureSetTitle'] +
    #            '_' + msi['outcomeVar'] + '_' + str(msi['nLayers']) + 'Layers_' + msi['modelType'] +
    #            '_kerasModelPlot.pdf')
    """Plotting Functions"""
    # NNStructurePlot(kerasModel)


def modelComparisonPlot(modelDict, metrics):
    fig = plt.figure(num=1, figsize=(4, 4), dpi=300, frameon=False)
    # Subplot 1: Model Details
    legendLines = []
    legendStr = []
    ax = fig.add_subplot(1, 1, 1)
    metricAlpha = 1
    lineWidth = 0.6
    for k in modelDict.keys():
        if 'NN' in k:
            metricLS = '--'
        else:
            metricLS = '-'

        x, y, t = roc_curve(modelDict[k]['evalTrue'], modelDict[k]['evalScore'])
        ax.plot(x, y, linewidth=lineWidth, ls=metricLS, alpha=metricAlpha)
        ax.plot([0, 1], [0, 1], linewidth=0.5, c=[.8, .8, .8], ls='--', alpha=0.4)
        legendLines.append(Line2D([0], [0], lw=lineWidth, alpha=metricAlpha,
                           linestyle=metricLS))
        legendStr.append(k)

    ax.legend(legendLines, legendStr, loc=4, fontsize=4)
    ax.set_xlabel('1 - Sensitivity', fontsize=6)
    ax.set_ylabel('Specificity', fontsize=6)
    ax.set_ylim((0, 1))
    ax.set_xlim((0, 1))
    ax.set_title('ROC Scores', fontsize=6)
    ax.tick_params(axis='both', labelsize=4)
    fig.savefig('output/NNoutput/paramSweeps/9_ModelROCComparison_' +
                datetime.now().strftime('%d-%m-%Y %H%M') + '.pdf', dpi=200,
                format='pdf', pad_inches=0.1, bbox_inches='tight')
    plt.close()

    fig = plt.figure(num=2, figsize=(8, 4), dpi=300, frameon=False)
    # Subplot 1: Model Details
    xTickStr = []
    xTicks = []
    ax = fig.add_subplot(1, 1, 1)
    metricAlpha = 1
    barWidth = 0.9
    i = 0
    for m in metrics:
        for k in modelDict.keys():
            if 'NN' in k:
                metricHatch = '//'
            else:
                metricHatch = ''

            ax.bar(x=i, height=modelDict[k][m], width=barWidth,
                   alpha=metricAlpha, hatch=metricHatch)
            chance = len(np.unique(modelDict[k]['evalTrue']))
            ax.plot([i-1, i+1], [1/chance, 1/chance], linewidth=0.5, c=[.8, .8, .8],
                    ls='--', alpha=0.4)
            ax.text(x=i, y=modelDict[k][m]+0.01,
                    s=str(round(modelDict[k][m], 2)), fontsize=3, ha='center')
            xTickStr.append(k)
            xTicks.append(i)
            i += 1
        ax.text(x=(i-1)-(len(modelDict.keys())/2), y=0.95, s=m, fontsize=6)
        i += 1
    ax.set_xticklabels(xTickStr, ha='right', rotation=45)
    ax.set_xticks(xTicks)
    ax.set_xlabel('Model', fontsize=6)
    ax.set_ylim((0, 1))
    ax.set_xlim((-0.5, i-0.5))
    ax.set_title('Evaluation Metrics', fontsize=6)
    ax.tick_params(axis='both', labelsize=4)
    fig.savefig('output/NNoutput/paramSweeps/9_ModelevalAccComparison_' +
                datetime.now().strftime('%d-%m-%Y %H%M') + '.pdf', dpi=200,
                format='pdf', pad_inches=0.1, bbox_inches='tight')
    plt.close()


def exportTrainingGif(msi):
    figParams = {'num': 1,
                 'figsize': (8, 3),
                 'dpi': 100,
                 'frameon': False}
    imgs = []
    nImg = 200
    nBins = None
    with tqdm.tqdm(total=nImg) as pbar:
        for i in [int(round(n)) for n in np.linspace(0, msi['nEpochs']-1, nImg)]:
            fig = plt.figure(**figParams)
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)
            # Training plot
            ax1.hist(msi['kerasTrainPredictionHistory'][i][msi['trainTrue'] == 1],
                     bins=nBins, color='xkcd:crimson', alpha=0.3)
            ax1.hist(msi['kerasTrainPredictionHistory'][i][msi['trainTrue'] == 0],
                     bins=nBins, color='xkcd:azure', alpha=0.3)
            ax1.set_ylim([0, 70])
            ax1.set_xlim([0, 1])
            ax1.set_title('Training Predictions')
            # Evaluation Plot
            ax2.hist(msi['kerasEvalPredictionHistory'][i][msi['evalTrue'] == 1],
                     bins=nBins, color='xkcd:crimson', alpha=1)
            ax2.hist(msi['kerasEvalPredictionHistory'][i][msi['evalTrue'] == 0],
                     bins=nBins, color='xkcd:azure', alpha=1)
            ax2.set_ylim([0, 70])
            ax2.set_xlim([0, 1])
            ax2.set_title('Evaluation Predictions')
            fig.subplots_adjust(wspace=0.3)
            fig.suptitle('Epoch - ' + str(i), fontsize=16)

            # Save Jpeg
            fig.savefig('output/NNoutput/histGIF/'+str(i)+'.jpg', dpi=100,
                        format='jpg', pad_inches=0.1, bbox_inches='tight')

            # Load Jpeg from dir into Gif constructor
            imgs.append(imgio.imread('output/NNoutput/histGIF/'+str(i)+'.jpg'))
            plt.clf()
            pbar.update(1)
    gifParams = {'duration': 0.1,
                 'loop': 1,
                 'palettesize': 256}
    print('Saving GIF...')
    imgio.mimsave('output/NNoutput/withOUTCOMEhistGIF.gif', imgs, 'GIF', **gifParams)
    plt.close()


def findOptimalCheckpoint(checkpointDict, checkpointDir, comparisonMetric='evalAuroc'):
    # For saved model in parameter sweep, load each optimal model state compare performance and return optimal model.
    modelKeys = [k for k in checkpointDict.keys()]
    cpPerfArray = {}
    comparisonMetricArray = []
    with tqdm.tqdm(total=len(modelKeys)) as pbar:
        for mdlKey in modelKeys:
            saved_model = load_model(checkpointDir+mdlKey+'.h5')

            evalInput = checkpointDict[mdlKey]['evalInput']
            evalTrue = checkpointDict[mdlKey]['evalTrue']
            evalScore = saved_model.predict(evalInput)

            trainInput = checkpointDict[mdlKey]['trainInput']
            trainTrue = checkpointDict[mdlKey]['trainTrue']
            trainScore = saved_model.predict(trainInput)

            cpPerfArray[mdlKey] = performanceMetrics([trainTrue, trainScore], [evalTrue, evalScore])

            try:
                comparisonMetricArray.append(cpPerfArray[mdlKey][comparisonMetric])
            except (KeyError):
                warningMsg = 'Not all models are suitable for current performance metric, changing to accuracy...'
                warnings.warn(warningMsg)
                comparisonMetric = 'evalAcc'
                comparisonMetricArray.append(cpPerfArray[mdlKey][comparisonMetric])
            pbar.update(1)

    # Find optimal model key
    optimalModelKey = modelKeys[np.argmax(comparisonMetricArray)]
    optimalNNParams = checkpointDict[optimalModelKey]['paramDict']
    # Clear partition indices used in hyperparamter sweep before handing params to
    # final model evaluation.
    optimalNNParams['trainIdx'] = []
    optimalNNParams['evalIdx'] = []
    return optimalNNParams, cpPerfArray


def exportPerfTable(performanceArray, metrics=['evalAuroc', 'evalAcc', 'evalSens', 'evalSpec']):
    mdlKeys = performanceArray.keys()
    summaryTable = pd.DataFrame(columns=metrics)
    for metric in metrics:
        metricArray = []
        for key in mdlKeys:
            summaryTable.loc[key, metric] = performanceArray[key][metric]
            metricArray.append(performanceArray[key][metric])
        summaryTable.loc['mean', metric] = round(np.nanmean(metricArray), 3)
        summaryTable.loc['CI', metric] = str(list(DescrStatsW(metricArray).tconfint_mean()))
    return summaryTable


def fullNNAnalysis(problemDict, hpDict, featMetaData):
    hpRawModelDict = MLPHyperParameterSweep(problemDict, hpDict, featMetaData)
    modelComparisonPlot(hpRawModelDict, metrics=['evalAcc', 'evalSens', 'evalSpec', 'evalAuroc'])  # , 'evalAcc', 'evalSens', 'evalSpec'

    # Using the saved optimal checkpoints from parameter sweep assess most accurate model architecture.
    print('Finding optimal point in parameter space')
    checkpointDir = 'output/NNoutput/modelCheckpoints/'
    optimalNNParams, sweepPerformanceArray = findOptimalCheckpoint(hpRawModelDict, checkpointDir, comparisonMetric='evalAuroc')

    # State output file path...
    outputFilePath = 'output/NNoutput/fullNNAnalysis/'+problemDict['df']['label']+'_'+problemDict['featureSet']['label']+'_'+problemDict['outcome']['label']+'_paramSweep'
    # Save cross validation performance array for later reference if required...
    with open(outputFilePath+'.pickle', 'wb') as handle:
        pickle.dump(sweepPerformanceArray, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # Construct export table from performance array...
    summaryTable = exportPerfTable(sweepPerformanceArray)
    summaryTable.to_csv(outputFilePath+'.tsv', sep='\t')

    # Using optimal model cross-validate performance...
    paramDict = optimalNNParams
    # Compute k-means cross-validation
    print('Conducting Cross-validation of optimal model..')
    cvModelDict = crossValidatedMLPAnalysis(problemDict, paramDict, featMetaData, k=10)
    modelComparisonPlot(cvModelDict, metrics=['evalAcc', 'evalSens', 'evalSpec', 'evalAuroc'])  # , 'evalAcc', 'evalSens', 'evalSpec'

    # Find optimal performance of cross validated models
    print('Assessing performance of optimal cross-validation states..')
    optimalNNParams, CVPerformanceArray = findOptimalCheckpoint(cvModelDict, checkpointDir, 'evalAuroc')

    # State output file path...
    outputFilePath = 'output/NNoutput/fullNNAnalysis/'+problemDict['df']['label']+'_'+problemDict['featureSet']['label']+'_'+problemDict['outcome']['label']+'_CV'
    # Save cross validation performance array for later reference if required...
    with open(outputFilePath+'.pickle', 'wb') as handle:
        pickle.dump(CVPerformanceArray, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # Construct export table from performance array...
    summaryTable = exportPerfTable(CVPerformanceArray)
    summaryTable.to_csv(outputFilePath+'.tsv', sep='\t')
    print('Model selection and cross-validation successful, output saved to:\n'+outputFilePath)
    return


def logDirManager():
    """ Directory manager for TensorFlow logging """
    print('Cleaning and initialising logging directory... \n')
    # Ensure function is starting from project root..
    if os.getcwd() != "/Users/Oliver/AnacondaProjects/SNSS_TF":
        os.chdir("/Users/Oliver/AnacondaProjects/SNSS_TF")
    os.chdir("tb_log")  # Go to logging folder..
    stdout = subprocess.check_output(["ls", "-a"])  # Send ls command to terminal
    # Decode output from terminal
    folders = DictReader(stdout.decode('ascii').splitlines(),
                         delimiter=' ', skipinitialspace=True,
                         fieldnames=['name'])
    # For every folder in ordered dict...
    for f in folders:
        path = f.get('name')  # Get path
        if (path != ('.')) & (path != ('..')) & (path != '.DS_Store'):  # Ignore parent dirs
            cDate = datetime.fromtimestamp(os.stat(os.getcwd() + '/' + f.get('name')).st_ctime)
            delta = datetime.today() - cDate  # Get age of folder.
            if delta.days > 6:  # If older than 1 week...
                rmtree(path)  # Delete folder.
                print('Removed old folder: "' + path + '" \n')  # Log deletion to console.
            # print('Name: ' + str + ' Created on: ' + cDate.isoformat())  # Debugging
    logDir = "log_dir/" + date.today().isoformat() + "/" +\
             datetime.now().time().isoformat(timespec='minutes').replace(':', '')
#  Create todays folder for logging
    print('Tensorflow logging to : ~/' + logDir + '\n')
    os.chdir('..')
    return logDir  # Return path to log dir
