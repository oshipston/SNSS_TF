#!/usr/bin/env python3.6
""" This is a drafting script for SNSS_TF"""
import os
os.chdir("/Users/oss/documents/code/SNSS_TF")
import SNSS_Prognosis_functions as oss
import importlib as imp
import numpy as np
import matplotlib.pyplot as plt, mpld3
import pandas as pd
import csv
import scipy
from statsmodels.discrete.discrete_model import Logit
import scipy.stats as stats
from sklearn.metrics import confusion_matrix, \
    accuracy_score, roc_auc_score, roc_curve, \
    classification_report, precision_score, recall_score, explained_variance_score, r2_score, f1_score
# //////////////////////////////////////////////////////////////////////////////
"""Config"""
# //////////////////////////////////////////////////////////////////////////////
imp.reload(oss)
oss.moduleInit()
cfg = oss.loadJSON("config.json")
varGroups = cfg['varGroups']
featMetaData = oss.loadJSON("raw_data/SNSS_vars.json")

""" Data Pipeline 1. Import dataset from UofE SMB Datashare of local file """
raw = oss.import_SNSS(usr=input("User: "), pwd=input("Password: "), local_file=1)

# //////////////////////////////////////////////////////////////////////////////
""" Data Pipeline 2. Quantify those lost to follow up, add binary follow up variables"""
# //////////////////////////////////////////////////////////////////////////////
SNSSDf, retentionTable = oss.SNSSNullity(raw)

# //////////////////////////////////////////////////////////////////////////////
""" Data Pipeline 3. Preprocess raw data"""
# //////////////////////////////////////////////////////////////////////////////
# Compute compund variables and and sum scores for individual scales.
# Also responsible for mapping SIMD 2004 data to patient postcodes.
SNSSDf = oss.SNSSCompoundVariables(SNSSDf)

# //////////////////////////////////////////////////////////////////////////////
""" Data Pipeline 4. Declare SNSS specific feature sets and dummy vars"""
# //////////////////////////////////////////////////////////////////////////////
# Declare relevant feature set lists
wholeSet = varGroups['T0_Demographics'] + varGroups['T0_Socioeconomic'] + \
    varGroups['T0_HADS'] + varGroups['T0_SF12'] + varGroups['T0_IPQ'] + \
    varGroups['T0_Whiteley'] + varGroups['T0_Cause'] + varGroups['T0_Satisfaction'] + \
    varGroups['T0_PHQSymptoms'] + varGroups['T0_NeuroSymptoms']

T1_wholeSet = varGroups['T1_HADS'] + varGroups['T1_SF12'] + varGroups['T1_IPQ'] + \
    varGroups['T1_Whiteley'] + varGroups['T1_Satisfaction'] + \
    varGroups['T1_PHQSymptoms'] + varGroups['T1_NeuroSymptoms']

T1_outcomes = varGroups['T1_Outcomes']

sharpe2010Set = ['AgeBins', 'Gender', 'Diagnosis', 'T0_PHQ13_Binned',
                 'T0_SF12_PF', 'T0_HADS_Binned', 'T0_NegExpectation',
                 'T0_PsychAttribution', 'T0_IllnessWorry',
                 'T0_IncapacityBenefitorDLA',
                 'T0_SIMD04_bin']

notExplainedReduced = varGroups['T0_0.001NotExplainedReduced']
explainedReduced = varGroups['T0_0.001ExplainedReduced']

# When coding categorical variables as dummy variables the k-1 dummy procedure automatically chooses
# the first in the answer order, the below dict specifies the answer value to drop.
dummyExceptionDict = {'T0_SF12_PF': 100.0,
                      'T0_PsychAttribution': 1.0,
                      'T0_C1': 5.0,
                      'T0_C2': 5.0,
                      'T0_C3': 5.0,
                      'T0_C4': 5.0,
                      'T0_C5': 5.0,
                      'T0_C6': 5.0,
                      'T0_C7': 5.0,
                      'T0_C8': 5.0,
                      'T0_C9': 5.0,
                      'T0_C10': 5.0,
                      'T0_C11': 5.0,
                      'T0_C12': 5.0,
                      'T0_Sat1': 5.0,
                      'T0_Sat2': 5.0,
                      'T0_Sat3': 5.0,
                      'T0_Sat4': 5.0,
                      'T0_Sat5': 5.0,
                      'T0_Sat6': 5.0,
                      'T0_Sat7': 5.0,
                      'T0_Sat8': 5.0,
                      'ExpGroups': 2.0}

# //////////////////////////////////////////////////////////////////////////////
"""Analysis 0. Compare lost to follow up and follow up groups"""
# //////////////////////////////////////////////////////////////////////////////
# Do functional and structural/lost to follow up and followed up groups differ?
imp.reload(oss)
oss.FollowUpandBaselineComparison(SNSSDf)

# Which of these predict loss to follow up in a multivariate model??
featureSet = sharpe2010Set
sharpe2010SetTypeDict = {'AgeBins': 'nominal',
                         'Gender': 'binary',
                         'Diagnosis': 'nominal',
                         'T0_PHQ13_Binned': 'nominal',
                         'T0_SF12_PF': 'nominal',
                         'T0_HADS_Binned': 'nominal',
                         'T0_NegExpectation': 'binary',
                         'T0_PsychAttribution': 'binary',
                         'T0_IllnessWorry': 'nominal',
                         'T0_IncapacityBenefitorDLA': 'binary',
                         'T0_SIMD04_bin': 'nominal'}
outcome = 'T2_HCData'
groupVar = 'ExpGroups'  # State which variable to group by.
multiGroupException = {1: 1.0,   # State which dummy values to drop for each group value
                       2: 3.0}
byGroupMdlT, byGroupMdls, byGroupMSI = oss.multiGroupLogisticRegression(df=SNSSDf, featureSet=featureSet,
                                                            outcomeVar=outcome, featMetaData=featMetaData,
                                                            featDataTypeDict=sharpe2010SetTypeDict,
                                                            dummyExceptionDict=dummyExceptionDict,
                                                            groupVar=groupVar,
                                                            multiGroupException=multiGroupException,
                                                            MV=1)
# Results Export...
byGroupMdlT['Not Explained'].to_csv('output/0_MVAnalysis_NotExplained'+outcome+'.tsv', sep='\t')
byGroupMdlT['Explained'].to_csv('output/0_MVAnalysis_Explained'+outcome+'.tsv', sep='\t')
oss.logitCoeffForestPlot(byGroupMdlT, mdl=[], tag=['0', 'MV', outcome],
                         groupVar=groupVar, returnPlot=False)
# //////////////////////////////////////////////////////////////////////////////
""" Analysis 1. Compare outcomes between functional category"""
# //////////////////////////////////////////////////////////////////////////////
# Do functional patients report different outcomes?
outcome = ['T2_poorCGI', 'T2_poorIPS', 'T2_HealthChange', 'T2_SymptomsChange']
groupVar = 'ExpGroups'
_ = oss.SNSSPrimaryOutcomeMeasures(SNSSDf)
oss.primaryOutcomePlot(outcome, groupVar, SNSSDf, featMetaData, style='stacked')

# //////////////////////////////////////////////////////////////////////////////
"""Analysis 2. Assess secondary outcomes"""
# //////////////////////////////////////////////////////////////////////////////
# Do SF12, HADS and symptom counts describe different trajectories
# over the study period?
outcome = [[['T0_SF12_NormedMCS', 'T1_SF12_NormedMCS', 'T2_SF12_NormedMCS'],
            ['T0_SF12_NormedPCS', 'T1_SF12_NormedPCS', 'T2_SF12_NormedPCS']],
            [['T0_PHQNeuro28_Total', 'T1_PHQNeuro28_Total', 'T2_PHQNeuro28_Total'],
            ['T0_HADS', 'T1_HADS', 'T2_HADS']]]
groupVar = 'ExpGroups'
_ = oss.SNSSSecondaryOutcomeMeasures(SNSSDf)
oss.secondaryOutcomePlot(outcome, groupVar, SNSSDf, featMetaData, style='line')

# //////////////////////////////////////////////////////////////////////////////
"""Analysis 3. Validate Scottish Index of Multiple Deprivation 2004 Interactions"""
# //////////////////////////////////////////////////////////////////////////////
# Given differences in receipt of benefit does living in a deprived area
# explain the observed effect?
_ = oss.SNSSSocioeconomicAssessment(SNSSDf)

# //////////////////////////////////////////////////////////////////////////////
"""Analysis 4. Compute univariate odds ratios for poor outcome:"""
# //////////////////////////////////////////////////////////////////////////////
# Which variables predict poor outcome in functional vs structural patients?
featureSet = sharpe2010Set
outcome = 'T2_poorCGI'
groupVar = 'ExpGroups'  # State which variable to group by.
multiGroupException = {1: 1.0,   # State which dummy values to drop for each group value
                       2: 3.0}
imp.reload(oss)
byGroupMdlT, byGroupMdls, byGroupMSI = oss.multiGroupLogisticRegression(df=SNSSDf, featureSet=featureSet,
                                                            outcomeVar=outcome, featMetaData=featMetaData,
                                                            featDataTypeDict=sharpe2010SetTypeDict,
                                                            dummyExceptionDict=dummyExceptionDict,
                                                            groupVar=groupVar,
                                                            multiGroupException=multiGroupException,
                                                            MV=0)
# Results Export...
byGroupMdlT['Not Explained'].to_csv('output/4_UVAnalysis_NotExplained'+outcome+'.tsv', sep='\t')
byGroupMdlT['Explained'].to_csv('output/4_UVAnalysis_Explained'+outcome+'.tsv', sep='\t')
oss.logitCoeffForestPlot(byGroupMdlT, mdl=[], tag=['4', 'UV', outcome],
                         groupVar=groupVar, returnPlot=False)
"""Addendum to Analysis 4: Combined analysis with diagnosis as predictor"""
sharpe2010SetAdapted = ['AgeBinInt', 'Gender_bin', 'ExpGroups_bin', 'T0_PHQNeuro28_BinInt',
                        'T0_SF12_PF_BinInt', 'T0_HADS_BinInt', 'T0_NegExpectation',
                        'T0_LackofPsychAttribution', 'T0_IllnessWorry',
                        'T0_IncapacityBenefitorDLA', 'T0_SIMD04_bin',
                        'ExpGroups_bin*T0_IncapacityBenefitorDLA',
                        'ExpGroups_bin*T0_LackofPsychAttribution',
                        'ExpGroups_bin*T0_SIMD04_bin',
                        'ExpGroups_bin*T0_NegExpectation']

sharpe2010SetAdaptedTypeDict = {'AgeBinInt': 'continuous',
                                'Gender': 'binary',
                                'Gender_bin': 'binary',
                                'ExpGroups_bin': 'binary',
                                'Diagnosis': 'binary',
                                'T0_PHQNeuro28_BinInt': 'continuous',
                                'T0_SF12_PF_BinInt': 'ordinal',
                                'T0_HADS_BinInt': 'continuous',
                                'T0_NegExpectation': 'binary',
                                'T0_LackofPsychAttribution': 'binary',
                                'T0_IllnessWorry': 'ordinal',
                                'T0_IncapacityBenefitorDLA': 'binary',
                                'T0_SIMD04_bin': 'binary',
                                'ExpGroups_bin*T0_IncapacityBenefitorDLA': 'binary',
                                'ExpGroups_bin*T0_LackofPsychAttribution': 'binary',
                                'ExpGroups_bin*T0_SIMD04_bin': 'binary',
                                'ExpGroups_bin*T0_NegExpectation': 'binary',
                                'ExpGroups_bin*T0_SF12_PF_BinInt': 'ordinal'}

outcomeVar = 'T2_poorCGI'
featureSet = sharpe2010SetAdapted

UVMdlExportT, mdlArray, mdlMSI = oss.UVLogisticRegression_v2(df=SNSSDf,
                                                     featureSet=featureSet,
                                                     outcomeVar=outcomeVar,
                                                     featMetaData=featMetaData,
                                                     featDataTypeDict=sharpe2010SetAdaptedTypeDict,
                                                     dummyExceptionDict=dummyExceptionDict)

# Results Export...
UVMdlExportT.to_csv('output/4adapted_UVAnalysis_All'+outcome+'.tsv', sep='\t')
oss.logitCoeffForestPlot({'All': UVMdlExportT}, mdl=[], tag=['4adapted', 'UV', outcomeVar],
                         groupVar='All', returnPlot=False)
# //////////////////////////////////////////////////////////////////////////////
"""Analysis 5. Compute multivariate odds ratios for poor outcome"""
# //////////////////////////////////////////////////////////////////////////////
# Which variables predict poor outcome in functional vs structural patients when
# accounting for all others?
featureSet = sharpe2010Set
# featureSet = sharpe2010Set[0:len(sharpe2010Set)-4]
outcomeVar = 'T2_poorCGI'
groupVar = 'ExpGroups'  # State which variable to group by.
multiGroupException = {1: 1.0,   # State which dummy values to drop for each group value
                       2: 3.0}
byGroupMdlT, byGroupMdls, byGroupMSI = oss.multiGroupLogisticRegression(df=SNSSDf, featureSet=featureSet,
                                                            outcomeVar=outcomeVar, featMetaData=featMetaData,
                                                            featDataTypeDict=sharpe2010SetTypeDict,
                                                            dummyExceptionDict=dummyExceptionDict,
                                                            groupVar=groupVar,
                                                            multiGroupException=multiGroupException,
                                                            MV=1)
# Results Export...
byGroupMdlT['Not Explained'].to_csv('output/5_MVAnalysis_NotExplained'+outcomeVar+'.tsv', sep='\t')
byGroupMdlT['Explained'].to_csv('output/5_MVAnalysis_Explained'+outcomeVar+'.tsv', sep='\t')
oss.logitCoeffForestPlot(byGroupMdlT, mdl=[], tag=['5', 'MV', outcomeVar],
                         groupVar=groupVar, returnPlot=False)

byGroupMSI['Explained']
"""Addendum to Analysis 5: Combined analysis with diagnosis as predictor"""
imp.reload(oss)

outcomeVar = 'T2_poorCGI'
featureSet = sharpe2010SetAdapted
# _dummyExceptionDict = dummyExceptionDict
# _dummyExceptionDict['Diagnosis'] = 2.0
MVdf = SNSSDf
mdlExportT, mdl, msi = oss.MVLogisticRegression_v2(df=MVdf,
                                                 featureSet=featureSet,
                                                 outcomeVar=outcomeVar,
                                                 featMetaData=featMetaData,
                                                 featDataTypeDict=sharpe2010SetAdaptedTypeDict,
                                                 dummyExceptionDict=dummyExceptionDict)
# Results Export...
mdlExportT.to_csv('output/5adapted_MVAnalysis_All'+outcomeVar+'.tsv', sep='\t')
oss.logitCoeffForestPlot({'All': mdlExportT}, mdl=[], tag=['5adapted', 'MV', outcomeVar],
                         groupVar='All', returnPlot=False)
                         msi
"""Analysis 6a. Reduce data set with UV regression coefficients/p-vals"""
# Feature Selection with UV Regression
featureSet = wholeSet
outcome = 'T2_poorCGI'
groupVar = 'ExpGroups'  # State which variable to group by.
multiGroupException = {1: 1.0,   # State which dummy values to drop for each group value
                       2: 3.0}
# NEED TO ADD AUTOMATED TYPE ASSIGNMENT FOR UVREGV_2 AS NO TYPE DICT FOR AUTOMATED
# VARIABLE SELECTION IN BIG DATASETS
# byGroupMdlT, byGroupMdls, byGroupMSI = oss.multiGroupLogisticRegression(df=SNSSDf, featureSet=featureSet,
#                                                             outcomeVar=outcome, featMetaData=featMetaData,
#                                                             dummyExceptionDict=dummyExceptionDict,
#                                                             groupVar=groupVar,
#                                                             featDataTypeDict=sharpe2010SetTypeDict
#                                                             multiGroupException=multiGroupException,
#                                                             MV=0)
byGroupMdlT['Not Explained'].to_csv('output/6a_WholeSet_UVAnalysis_NotExplained'+outcome+'.tsv', sep='\t')
byGroupMdlT['Explained'].to_csv('output/6a_WholeSet_UVAnalysis_Explained'+outcome+'.tsv', sep='\t')
oss.logitCoeffForestPlot(byGroupMdlT, mdl=[], tag=['6a', 'UV', outcome],
                         groupVar=groupVar, returnPlot=False)

alpha = 0.001
groupVar = 'ExpGroups'
outcome = 'T2_poorCGI'
reducedfeatures = {}
reducedMVTables = {}
reducedMVModels = {}
groupVarDict = dict(zip(featMetaData[groupVar]['valuelabels'],
                        featMetaData[groupVar]['values']))
for G in ['Not Explained', 'Explained']:
    reducedfeatures[G] = []
    for P in set(byGroupMdlT[G].index.get_level_values(0)):
        if byGroupMdlT[G].loc[P].llrp[1] < alpha:
            reducedfeatures[G].append(P)
    print(str(len(reducedfeatures[G])) + ' Variables with llrp < ' + str(alpha) + ' for ' + G + ' group.')
    # Compute logistic regression with reduced dataset..
    reducedMVTables[G], reducedMVModels[G], _ =\
        oss.MVLogisticRegression(df=SNSSDf[SNSSDf[groupVar] == groupVarDict[G]],
                                 featureSet=reducedfeatures[G], outcomeVar=outcome,
                                 featMetaData=featMetaData, dummyExceptionDict=dummyExceptionDict)
    oss.logitCoeffForestPlot(predictorTables={'All': reducedMVTables[G]}, mdl=[],
                             tag=['6a_Reduced_' + G, 'MV', outcome],
                             groupVar='All', returnPlot=False)

reducedfeatures['Not Explained']
# //////////////////////////////////////////////////////////////////////////////
"""Analysis 7. NN Assessment of SNSS Prognosis"""
# //////////////////////////////////////////////////////////////////////////////
imp.reload(oss)
# Declare the groups/datasets of interest...
F_OIdx = (SNSSDf['ExpGroups'] == 1)
S_OIdx = (SNSSDf['ExpGroups'] == 2) # SNSSD f[F_OIdx], SNSSDf[S_OIdx],

# Declare the hyperparameter space...
hpDict = {'nHiddenLayerArray': {'value': [1, 3, 5], 'order': 0},
          'modelTypeArray': {'value': ['selfNormalising', 'deepFeedForwardLeakyReLU', 'deepFeedForwardTanH'], 'order': 1},
          'partitionRatioArray': {'value': [0.7], 'order': 2},
          'maxHiddenLayerSizeArray': {'value': [20, 80], 'order': 3},
          'nEpochArray': {'value': [7000], 'order': 4}}
# Experiment 1: Functional with whole set
# Declare the problem parameters for each experiment.
imp.reload(oss)

problemDict = {'featureSet': {'value': wholeSet, 'label': 'wholeSet'},
               'outcome': {'value': 'T2_poorCGI' , 'label': 'T2_poorCGI'},
               'df': {'value': SNSSDf[F_OIdx], 'label': 'Functional'}}
oss.fullNNAnalysis(problemDict, hpDict, featMetaData)

# Experiment 2: Structural with whole set
problemDict = {'featureSet': {'value': wholeSet, 'label': 'wholeSet'},
               'outcome': {'value': 'T2_poorCGI' , 'label': 'T2_poorCGI'},
               'df': {'value': SNSSDf[S_OIdx], 'label': 'Structural'}}
oss.fullNNAnalysis(problemDict, hpDict, featMetaData)

# Experiment 3: Functional with notExplainedReduced
problemDict = {'featureSet': {'value': notExplainedReduced, 'label': 'notExplainedReduced'},
               'outcome': {'value': 'T2_poorCGI' , 'label': 'T2_poorCGI'},
               'df': {'value': SNSSDf[F_OIdx], 'label': 'Functional'}}
oss.fullNNAnalysis(problemDict, hpDict, featMetaData)

# Experiment 4: Structural explainedReduced
problemDict = {'featureSet': {'value': explainedReduced, 'label': 'explainedReduced'},
               'outcome': {'value': 'T2_poorCGI' , 'label': 'T2_poorCGI'},
               'df': {'value': SNSSDf[S_OIdx], 'label': 'Structural'}}
oss.fullNNAnalysis(problemDict, hpDict, featMetaData)

# Experiment 5: Functional with sharpe2010Set
problemDict = {'featureSet': {'value': sharpe2010Set, 'label': 'sharpe2010Set'},
               'outcome': {'value': 'T2_poorCGI' , 'label': 'T2_poorCGI'},
               'df': {'value': SNSSDf[F_OIdx], 'label': 'Functional'}}
oss.fullNNAnalysis(problemDict, hpDict, featMetaData)

# Experiment 6: Structural with sharpe2010Set
problemDict = {'featureSet': {'value': sharpe2010Set, 'label': 'sharpe2010Set'},
               'outcome': {'value': 'T2_poorCGI' , 'label': 'T2_poorCGI'},
               'df': {'value': SNSSDf[S_OIdx], 'label': 'Structural'}}
oss.fullNNAnalysis(problemDict, hpDict, featMetaData)

# Experiment 7: Functional with wholeSet+T1_wholeSet
problemDict = {'featureSet': {'value': wholeSet+T1_wholeSet, 'label': 'wholeSet+T1_wholeSet'},
               'outcome': {'value': 'T2_poorCGI' , 'label': 'T2_poorCGI'},
               'df': {'value': SNSSDf[F_OIdx], 'label': 'Functional'}}
oss.fullNNAnalysis(problemDict, hpDict, featMetaData)

# Experiment 8: Structural with wholeSet+T1_wholeSet
problemDict = {'featureSet': {'value': wholeSet+T1_wholeSet, 'label': 'wholeSet+T1_wholeSet'},
               'outcome': {'value': 'T2_poorCGI' , 'label': 'T2_poorCGI'},
               'df': {'value': SNSSDf[S_OIdx], 'label': 'Structural'}}
oss.fullNNAnalysis(problemDict, hpDict, featMetaData)

# Experiment 9: Functional with wholeSet+T1_wholeSet+T1_outcomes
problemDict = {'featureSet': {'value': wholeSet+T1_wholeSet+T1_outcomes, 'label': 'wholeSet+T1_wholeSet+T1_outcomes'},
               'outcome': {'value': 'T2_poorCGI' , 'label': 'T2_poorCGI'},
               'df': {'value': SNSSDf[F_OIdx], 'label': 'Functional'}}
oss.fullNNAnalysis(problemDict, hpDict, featMetaData)

# Experiment 10: Structural with wholeSet+T1_wholeSet+T1_outcomes
problemDict = {'featureSet': {'value': wholeSet+T1_wholeSet+T1_outcomes, 'label': 'wholeSet+T1_wholeSet+T1_outcomes'},
               'outcome': {'value': 'T2_poorCGI' , 'label': 'T2_poorCGI'},
               'df': {'value': SNSSDf[S_OIdx], 'label': 'Structural'}}
oss.fullNNAnalysis(problemDict, hpDict, featMetaData)

# Experiment 11: Functional with wholeSet+T1_wholeSet+T1_outcomes
problemDict = {'featureSet': {'value': wholeSet+T1_wholeSet+T1_outcomes, 'label': 'wholeSet+T1_wholeSet+T1_outcomes'},
               'outcome': {'value': 'T0T2_SF12_binaryNormedPCS' , 'label': 'T0T2_SF12_binaryNormedPCS'},
               'df': {'value': SNSSDf[F_OIdx], 'label': 'Functional'}}
oss.fullNNAnalysis(problemDict, hpDict, featMetaData)

# Experiment 12: Structural with wholeSet+T1_wholeSet+T1_outcomes
problemDict = {'featureSet': {'value': wholeSet+T1_wholeSet+T1_outcomes, 'label': 'wholeSet+T1_wholeSet+T1_outcomes'},
               'outcome': {'value': 'T0T2_SF12_binaryNormedPCS' , 'label': 'T0T2_SF12_binaryNormedPCS'},
               'df': {'value': SNSSDf[S_OIdx], 'label': 'Structural'}}
oss.fullNNAnalysis(problemDict, hpDict, featMetaData)
