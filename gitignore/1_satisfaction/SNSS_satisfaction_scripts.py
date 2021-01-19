#!/usr/bin/env python3.6
""" This is a drafting script for SNSS_TF Satisfaction Analyses"""
import SNSS_Prognosis_functions as oss
import SNSS_satisfaction_functions as sat_oss
import importlib as imp
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels as sm
from collections import Counter
from sklearn import preprocessing, decomposition, manifold
import seaborn as sb
import os
from statsmodels.tools.tools import add_constant
from statsmodels.discrete.discrete_model import Logit
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW
import math
import missingno as msno


pd.options.display.max_columns = None
pd.options.display.max_rows = 50

cfg = oss.loadJSON("config.json")
SNSSvars = oss.loadJSON("raw_data/SNSS_vars.json")

""" Import dataset """
raw = oss.import_SNSS(usr='s1302553', pwd=input("Password: "))

""" Return dataframes for every time point and nullity """
df, T0df, T1df, T2df, T1and2df, retentionTable = oss.SNSSNullity(raw)  # Remove NAs from T0 dat
# fig = msno.bar(df.filter(regex='T0'), inline=False, fontsize=8, figsize=(48, 5))

"""Construct compound T0 Measures, bins and outcomes:
Bins continous variables and computes various compound scores
"""
df = oss.SNSSCompoundVariables(df)  # Add compound measures
# fig = msno.bar(df, inline=False, fontsize=8, figsize=(48, 5))

"""Compare lost to follow up and follow up groups
Do functional and structural/lost to follow up and followed up groups differ?
"""
sat_oss.SatisfactionFollowUpandBaselineComparison(df)

"""Construct tables for satisfaction questions
Do functional and structural groups differ in Satisfaction?
"""
sat_oss.ExplainedSatisfactionComparison(df, 'ExpGroups')

"""Export barplot of satisfaction means for T0 and T1"""
sat_oss.plotSatisfactionMeanBars(df, SNSSvars)

"""Construct tables for export to tsv for victor"""
sat_oss.exportForVictor(df)


# satVars = ['T0_Sat1', 'T0_Sat2',
#            'T0_Sat3', 'T0_Sat4',
#            'T0_Sat5', 'T0_Sat6',
#            'T0_Sat7', 'T0_Sat8']
satVars = ['T0_Sat1_Poor_Bin', 'T0_Sat2_Poor_Bin',
           'T0_Sat3_Poor_Bin', 'T0_Sat4_Poor_Bin',
           'T0_Sat5_Poor_Bin', 'T0_Sat6_Poor_Bin',
           'T0_Sat7_Poor_Bin', 'T0_Sat8_Poor_Bin']


def SNSSSatisfactionOutcomePredictionAnalysis(df):
    """ Multivariate analysis of predictors of poor outcome """
    SNSSvars = oss.loadJSON('raw_data/SNSS_vars.json')
    MVPredictorTables = {}
    logitModels = {}
    GLabels = ['Functional', 'Structural']
    satVars = ['T0_Sat1', 'T0_Sat2',
               'T0_Sat3', 'T0_Sat4',
               'T0_Sat5', 'T0_Sat6',
               'T0_Sat7', 'T0_Sat8']
    for S in satVars:
        MVPredictors = ['AgeBins', 'Gender', 'T0_SF12_PF', 'T0_SIMD04_bin'] + [S]
        MVPredictorTables[S] = {}
        logitModels[S] = {}
        for G in [1, 2]:
            rDat = df[(df.ExpGroups == G) & (df.T1_poorCGI.notna())]
            rDat = rDat.dropna(subset=MVPredictors)
            endog = np.asarray(rDat.T1_poorCGI).astype(int)
            exog = np.ones([len(rDat), 1]).astype(int)
            mdlName = S + '_' + GLabels[G-1]
            varNames = ['constant']
            sigTestIdx = {}
            for P in MVPredictors:
                # P = MVPredictors[1]
                varDat = rDat[P]
                if varDat.dtype.name != 'category':  # If not a categorical convert...
                    varDat = pd.Categorical(varDat)

                if (P == 'T0_SF12_PF'):
                    X = pd.get_dummies(varDat, drop_first=False).drop(axis=1, columns=100.0)
                elif (P == 'T0_PsychAttribution'):
                    X = pd.get_dummies(varDat, drop_first=False).drop(axis=1, columns=1.0)
                elif 'Sat' in P:
                    X = pd.get_dummies(varDat, drop_first=False).drop(axis=1, columns=5.0)
                elif (P == 'Diagnosis'):
                    if G == 1:
                        X = pd.get_dummies(varDat, drop_first=False).drop(axis=1, columns=1.0)
                    elif G == 2:
                        X = pd.get_dummies(varDat, drop_first=False).drop(axis=1, columns=3.0)
                else:
                    X = pd.get_dummies(varDat, drop_first=True)

                # Translate categorical series labels into SNSS var value labels..
                varDict = dict(zip(SNSSvars[P]['values'],
                                   SNSSvars[P]['valuelabels']))

                for col in range(X.shape[1]):
                    try:
                        varNames.append(SNSSvars[P]['label'] + ' - ' +
                                        varDict[X.columns.
                                                categories[X.columns.codes].values.tolist()[col]])
                    except (KeyError) as err:
                        varNames.append(SNSSvars[P]['label'] + ' - ' +
                                        str(X.columns.
                                            categories[X.columns.codes].values.tolist()[col]))

                # Save column indices in dict for significance testing later...
                sigTestIdx[P] = range(exog.shape[1], exog.shape[1]+X.shape[1])
                # Append dummy encoded variable to exog array...
                exog = np.append(exog, X, axis=1)

            # Construct Logistic model from all variable array...
            lr = Logit(endog=endog, exog=exog)
            mdl[mdlName] = lr.fit(disp=0)
            coeffT = pd.DataFrame(mdl[mdlName].params, index=[[P]*len(varNames), varNames],
                                  columns=['coeff'])
            coeffT['coeffLCI'] = mdl[mdlName].conf_int()[:, 0]
            coeffT['coeffUCI'] = mdl[mdlName].conf_int()[:, 1]
            coeffT['OR'] = np.exp(mdl[mdlName].params)
            coeffT['ORLCI'] = np.exp(mdl[mdlName].conf_int())[:, 0]
            coeffT['ORUCI'] = np.exp(mdl[mdlName].conf_int())[:, 1]
            coeffT['p'] = mdl[mdlName].pvalues

            # Variable significance testing...
            pValT = []
            for P in MVPredictors:
                testLr = Logit(endog=endog, exog=np.delete(exog, sigTestIdx[P], axis=1))
                testMdl = testLr.fit(disp=0)
                Chi2p = 1 - stats.chi2.cdf(2*(mdl[mdlName].llf - testMdl.llf), df=len(sigTestIdx[P]))
                pValT.append(pd.DataFrame(Chi2p, index=[P],
                                          columns=['llrp']))

            # Save individual coefficient details...
            coeffT.to_csv('output/sat/SatLR/3_' + S + '_' + GLabels[G-1] + 'MVAnalysis.tsv', sep='\t')
            MVPredictorTables[S].update({GLabels[G-1]: coeffT})

            # Save significance tests...
            pd.concat(pValT).to_csv('output/sat/SatLR/3_' + S + '_' + GLabels[G-1] + 'MVpVals.tsv', sep='\t')
            logitModels[S].update({GLabels[G-1]: mdl})
    return MVPredictorTables, mdl


T, mdl = SNSSSatisfactionOutcomePredictionAnalysis(df)
T['T0_Sat1'].keys()
mdl['T0_Sat_Structural'].prsquared
for S in ['T0_Sat1', 'T0_Sat2',
          'T0_Sat3', 'T0_Sat4',
           'T0_Sat5', 'T0_Sat6',
           'T0_Sat7', 'T0_Sat8']:
    oss.SNSSLogitCoeffForestPlot(T[S], tag=['/sat/SatLR/3_' + S, 'UV'], returnPlot=False)

outcome = 'T1_poorCGI'
T, mdl = sat_oss.SNSSSatisfactionRegressionAnalysis(df, outcome)

oss.SNSSLogitCoeffForestPlot(T, mdl, tag=['/sat/3', outcome + '_MV'], returnPlot=False)


imp.reload(oss)
