#!/usr/bin/env python3.6
"""This module describes functions for analysis of the SNSS Dataset"""
import subprocess
import os
from datetime import datetime, date
from csv import DictReader
from shutil import rmtree
import pandas as pd
import tensorflow as tf
from sas7bdat import SAS7BDAT
import tkinter as tk
import numpy as np
import time
from json import load as jsonLoad
from colour import Color
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW
from statsmodels.stats.proportion import proportion_confint

__author__ = "Oliver Shipston-Sharman"
__copyright__ = "Copyright 2017, Oliver Shipston-Sharman"
__credits__ = ["Oliver Shipston-Sharman", "Ingrid Hoeritzauer", "Alan Carson", "Jon Stone"]

__license__ = "Apache-2.0"
__version__ = "0.1.0"
__maintainer__ = "Oliver Shipston-Sharman"
__email__ = "s1302553@ed.ac.uk"


def loadJSON(fname):
    # Load configuration
    f = open(fname)  # Open config file...
    cfg = jsonLoad(f)  # Load data...
    f.close()  # Close config file...
    return cfg


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


def import_ENS():
    """ Import ENS_OSS.

    Note you must have access permissions to specific share.
    Keyword arguments:
    usr = Edinburgh University matriculation number
    pwd = Edinburgh University Password

    Location of data is specified in a JSON config file not included.
    """
    df = pd.read_csv('ENS/raw_data/ENS_OSS.csv', index_col=0)
    return df


def ENSNullity(raw):
    """ Drops NAs and returns patient retention data """
    dxDat = raw.dropna(subset=['functional'])
    _, dxDatCounts = np.unique(dxDat.functional, return_counts=True)
    dxCounts = np.hstack([dxDatCounts,
                          sum(dxDatCounts[0:2]),
                          sum(dxDatCounts[2:4]),
                          sum(dxDatCounts)])

    df = dxDat.dropna()
    _, dfDatCounts = np.unique(df.functional, return_counts=True)
    dfCounts = np.hstack([dfDatCounts,
                          sum(dfDatCounts[0:2]),
                          sum(dfDatCounts[2:4]),
                          sum(dfDatCounts)])

    retentionTable = pd.DataFrame(np.vstack([dxCounts, dfCounts]),
                                  columns=['NA', 'S', 'L', 'C',
                                  'NA/S', 'L/C', 'N'],
                                  index=['FullDiagnosisDat', 'FullENSDat'])
    return df, retentionTable


def ENSCompoundVariables(df):
    df['functional_pool'] = df.functional > 1
    df['ens11_op_no_bin'] = df.ens11_op_no > 1
    return df

def multivariateAnalysis(df, featureSet, tag, outcome='T2_poorCGI', groupVar='ExpGroups'):
    """ Multivariate analysis of predictors of poor outcome """
    SNSSVars = loadJSON('raw_data/SNSS_vars.json')
    groups = df[groupVar].unique()
    GLabels = SNSSVars[groupVar]['valuelabels']

    MVPredictorTables = {}
    mdl = {}
    Gi = 0
    for G in groups:
        rDat = df[(df[groupVar] == G) & (df[outcome].notna())]
        rDat = rDat.dropna(subset=featureSet)
        endog = np.asarray(rDat[outcome]).astype(int)
        exog = np.ones([len(rDat), 1]).astype(int)
        varNames = ['constant']
        sigTestIdx = {}
        for P in featureSet:
            # P = MVPredictors[1]
            varDat = rDat[P]
            if varDat.dtype.name != 'category':  # If not a categorical convert...
                varDat = pd.Categorical(varDat)

            if (P == 'T0_SF12_PF'):
                X = pd.get_dummies(varDat, drop_first=False).drop(axis=1, columns=100.0)
            elif (P == 'T0_PsychAttribution'):
                X = pd.get_dummies(varDat, drop_first=False).drop(axis=1, columns=1.0)
            elif (P == 'Diagnosis'):
                if G == 1:
                    X = pd.get_dummies(varDat, drop_first=False).drop(axis=1, columns=1.0)
                elif G == 2:
                    X = pd.get_dummies(varDat, drop_first=False).drop(axis=1, columns=3.0)
            elif 'Causal Attribution' in SNSSVars[P]['label']:
                X = pd.get_dummies(varDat, drop_first=False).drop(axis=1, columns=5.0)
            else:
                X = pd.get_dummies(varDat, drop_first=True)

            # Translate categorical series labels into SNSS var value labels..
            varDict = dict(zip(SNSSVars[P]['values'],
                               SNSSVars[P]['valuelabels']))

            for col in range(X.shape[1]):
                try:
                    varNames.append(SNSSVars[P]['label'] + ' - ' +
                                    varDict[X.columns.
                                            categories[X.columns.codes].values.tolist()[col]])
                except (KeyError) as err:
                    varNames.append(SNSSVars[P]['label'] + ' - ' +
                                    str(X.columns.
                                        categories[X.columns.codes].values.tolist()[col]))

            # Save column indices in dict for significance testing later...
            sigTestIdx[P] = range(exog.shape[1], exog.shape[1]+X.shape[1])
            # Append dummy encoded variable to exog array...
            exog = np.append(exog, X, axis=1)

        # Construct Logistic model from all variable array...
        lr = Logit(endog=endog, exog=exog)
        mdl[GLabels[Gi]] = lr.fit(disp=0)
        coeffT = pd.DataFrame(mdl[GLabels[Gi]].params, index=[[P]*len(varNames), varNames],
                              columns=['coeff'])
        # coeffT = pd.DataFrame(mdl[G].params, index=[varNames, varVals],
        #                       columns=['coeff'])
        coeffT['coeffLCI'] = mdl[GLabels[Gi]].conf_int()[:, 0]
        coeffT['coeffUCI'] = mdl[GLabels[Gi]].conf_int()[:, 1]
        coeffT['OR'] = np.exp(mdl[GLabels[Gi]].params)
        coeffT['ORLCI'] = np.exp(mdl[GLabels[Gi]].conf_int())[:, 0]
        coeffT['ORUCI'] = np.exp(mdl[GLabels[Gi]].conf_int())[:, 1]
        coeffT['p'] = mdl[GLabels[Gi]].pvalues

        # Variable significance testing...
        pValT = []
        for P in featureSet:
            testLr = Logit(endog=endog, exog=np.delete(exog, sigTestIdx[P], axis=1))
            testMdl = testLr.fit(disp=0)
            Chi2p = 1 - stats.chi2.cdf(2*(mdl[GLabels[Gi]].llf - testMdl.llf), df=len(sigTestIdx[P]))
            pValT.append(pd.DataFrame(Chi2p, index=[P],
                                      columns=['llrp']))

        # Save individual coefficient details...
        coeffT.to_csv('output/' + tag[0] + '_' + GLabels[Gi] +
                      'MVAnalysis_' + outcome + '.tsv', sep='\t')
        MVPredictorTables[GLabels[Gi]] = coeffT

        # Save significance tests...
        pd.concat(pValT).to_csv('output/' + tag[0] + '_' + GLabels[Gi] +
                                'MVpVals_' + outcome + '.tsv', sep='\t')
        Gi += 1
    return MVPredictorTables, mdl


def univariateAnalysis(df, featureSet, tag, outcome='T2_poorCGI', groupVar='ExpGroups'):
    """ Univariate analysis of predictors of poor outcome """
    SNSSVars = loadJSON('raw_data/SNSS_vars.json')
    groups = df[groupVar].unique()
    GLabels = SNSSVars[groupVar]['valuelabels']

    UVPredictorTables = {}
    UVpValsTables = {}
    mdlT = {}

    Gi = 0
    for G in groups:
        rDat = df[(df[groupVar] == G) & (df[outcome].notna())]
        coeffFrames = []
        countFrames = []
        pValFrames = []
        mdlT[GLabels[Gi]] = {}
        for P in featureSet:
            print('Assessing ' + P)
            varDict = dict(zip(SNSSVars[P]['values'],
                               SNSSVars[P]['valuelabels']))

            # Calculate number of patients included in each Uinvariate analysis.
            countContingency = pd.crosstab(index=rDat[P], columns=rDat[outcome],
                                           margins=False, normalize=False,
                                           dropna=True)
            counts = countContingency.values
            varDat = rDat[P]
            if varDat.dtype.name != 'category':
                varDat = pd.Categorical(varDat)
                varLabels = varDat.categories.values.tolist()
            else:
                varLabels = varDat.cat.categories.values.tolist()

            countT = pd.DataFrame(np.sum(counts, axis=1),
                                  index=[[P]*len(varLabels), varLabels], columns=['N'])
            countT['N_poorCGI'] = countContingency.iloc[:, 1].values
            countT['P_poorCGI'] = (countContingency.iloc[:, 1].values)/np.sum(counts, axis=1)
            countT['Total'] = [np.sum(counts)]*len(varLabels)
            countT['N_Total_poorCGI'] = [np.sum(countContingency.iloc[:, 1].
                                                values)]*len(varLabels)
            countT['P_Total_poorCGI'] = [np.sum(countContingency.iloc[:, 1].
                                                values)/np.sum(counts)]*len(varLabels)

            countFrames.append(countT)

            # Custom dummy encoding..
            if (P == 'T0_SF12_PF'):
                X = pd.get_dummies(varDat, drop_first=False).drop(axis=1, columns=100.0)
            elif (P == 'T0_PsychAttribution'):
                X = pd.get_dummies(varDat, drop_first=False).drop(axis=1, columns=1.0)
            elif (P == 'Diagnosis'):
                if G == 1:
                    X = pd.get_dummies(varDat, drop_first=False).drop(axis=1, columns=1.0)
                elif G == 2:
                    X = pd.get_dummies(varDat, drop_first=False).drop(axis=1, columns=3.0)
            elif 'Sat' in P:
                X = pd.get_dummies(varDat, drop_first=False).drop(axis=1, columns=5.0)
            elif 'SF12' in P:
                dropVar = SNSSVars[P]['values'][len(SNSSVars[P]['values'])-1]
                X = pd.get_dummies(varDat, drop_first=False).drop(axis=1, columns=dropVar)
            elif 'Causal Attribution' in SNSSVars[P]['label']:
                X = pd.get_dummies(varDat, drop_first=False).drop(axis=1, columns=5.0)
            else:
                X = pd.get_dummies(varDat, drop_first=True)

            # Construct var labels including constant and labelvalues from SNSS_vars.
            # NOTE: Some dummy variable column codes are created as integers so attempt to call
            # label from dict but if not just use the value.
            try:
                varNames = [SNSSVars[P]['label'] + ' (constant)'] + [varDict[v] for v in X.columns.
                                                                     categories[X.columns.codes].
                                                                     values.tolist()]
            except (KeyError) as err:
                varNames = [SNSSVars[P]['label'] + ' (constant)'] + [v for v in X.columns.
                                                                     categories[X.columns.codes].
                                                                     values.tolist()]

            exog = np.asarray(add_constant(X), dtype=float)
            endog = np.asarray(rDat[outcome], dtype=float)
            lr = Logit(endog=endog, exog=exog)
            # lr1 = Logit(endog=endog, exog=exog[:, 0:exog.shape[1]-1])
            mdl = lr.fit(disp=0)
            # mdl1 = lr1.fit(disp=0)
            pValT = pd.DataFrame(mdl.llr_pvalue, index=[P],
                                 columns=['llrp'])
            # Log likelihood ratio test
            # 1 - stats.chi2.cdf(2*(mdl.llf-mdl1.llf), mdl.df_model)
            coeffT = pd.DataFrame(mdl.params, index=[[P]*len(varNames), varNames],
                                  columns=['coeff'])
            coeffT['coeffLCI'] = mdl.conf_int()[:, 0]
            coeffT['coeffUCI'] = mdl.conf_int()[:, 1]
            coeffT['OR'] = np.exp(mdl.params)
            coeffT['ORLCI'] = np.exp(mdl.conf_int())[:, 0]
            coeffT['ORUCI'] = np.exp(mdl.conf_int())[:, 1]
            coeffT['dummyp'] = mdl.pvalues
            coeffT['varp'] = mdl.pvalues
            coeffFrames.append(coeffT)
            pValFrames.append(pValT)
            mdlT[GLabels[Gi]][P] = mdl
        UVCoeffT = pd.concat(coeffFrames)
        UVCountT = pd.concat(countFrames)
        UVpValT = pd.concat(pValFrames)

        UVCoeffT.to_csv('output/' + tag[0] + '_' + GLabels[Gi] +
                        '_UVAnalysis_' + outcome + '.tsv', sep='\t')
        UVCountT.to_csv('output/' + tag[0] + '_' + GLabels[Gi] +
                        '_UVCounts_' + outcome + '.tsv', sep='\t')
        UVpValT.to_csv('output/' + tag[0] + '_' + GLabels[Gi] +
                       '_UVpVals_' + outcome + '.tsv', sep='\t')

        UVPredictorTables[GLabels[Gi]] = UVCoeffT
        UVpValsTables[GLabels[Gi]] = UVpValT
        Gi += 1
    return UVPredictorTables, UVpValsTables, mdlT


def SNSSPredictorClustering(df, featureSet):
    predictDat = df[featureSet].dropna()
    # outcomeDat['T2_HealthChange'] = outcomeDat['T2_HealthChange'].astype(int)
    predictDat['T0_SIMD04_score'] = np.log(predictDat['T0_SIMD04_score'])
    predictNorm = preprocessing.scale(predictDat[featureSet[0:len(featureSet)-2]])
    PCA = decomposition.PCA(n_components=3, svd_solver='full')
    predictPCA = PCA.fit(predictNorm).transform(predictNorm)
    tsne = manifold.TSNE(n_components=2, perplexity=80, n_iter=1000).fit_transform(predictNorm)

    plt.ioff()
    fig = plt.figure(num=1, figsize=(10, 10), dpi=200, frameon=False)
    colVar = 'T2_poorCGI'
    ax1 = fig.add_subplot(2, 2, 1)
    sb.scatterplot(x=predictPCA[:, 0], y=predictPCA[:, 1],
                   alpha=1, data=predictDat,
                   hue=colVar, s=8,
                   marker='.', edgecolor='none',
                   palette=sb.xkcd_palette(['faded green', 'dusty purple']),
                   ax=ax1)
    ax1.legend_.remove()
    ax1.set_xlabel('Principal Component 1')
    ax1.set_ylabel('Principal Component 2')
    ax1.set_title('PCA by Outcome')
    ax2 = fig.add_subplot(2, 2, 2)
    sb.scatterplot(x=tsne[:, 0], y=tsne[:, 1],
                   alpha=1, data=predictDat,
                   hue=colVar, s=8,
                   marker='.', edgecolor='none',
                   palette=sb.xkcd_palette(['faded green', 'dusty purple']),
                   ax=ax2)
    handles, _ = ax2.get_legend_handles_labels()
    ax2.legend(handles=handles,
               labels=['12 Month CGI', 'MW, W or NC', 'B or MB'],
               bbox_to_anchor=(1.05, 1), loc=1)
    ax2.set_xlabel('tSNE 1')
    ax2.set_ylabel('tSNE 2')
    ax2.set_title('tSNE by Outcome')

    colVar = 'ExpGroups'
    ax3 = fig.add_subplot(2, 2, 3)
    sb.scatterplot(x=predictPCA[:, 0], y=predictPCA[:, 1],
                   alpha=1, data=predictDat,
                   hue=colVar, s=8,
                   marker='.', edgecolor='none',
                   palette=sb.xkcd_palette(['windows blue', 'amber']),
                   ax=ax3)
    ax3.legend_.remove()
    ax3.set_xlabel('Principal Component 1')
    ax3.set_ylabel('Principal Component 2')
    ax3.set_title('PCA by Explained Group')
    ax4 = fig.add_subplot(2, 2, 4)
    sb.scatterplot(x=tsne[:, 0], y=tsne[:, 1],
                   alpha=1, data=predictDat,
                   hue=colVar, s=8,
                   marker='.', edgecolor='none',
                   palette=sb.xkcd_palette(['windows blue', 'amber']),
                   ax=ax4)
    handles, _ = ax4.get_legend_handles_labels()
    ax4.legend(handles=handles,
               labels=['Diagnostic Category', 'Not Explained', 'Explained'],
               bbox_to_anchor=(1.05, 1), loc=1)
    ax4.set_xlabel('tSNE 1')
    ax4.set_ylabel('tSNE 2')
    ax4.set_title('tSNE by Explained Group')
    fig.savefig('output/self-report-outcomes/2_PredictorClustering.pdf', dpi=200,
                format='pdf', pad_inches=0.1, bbox_inches='tight')
    plt.close()
    return
# def ENSrmSQNA(df):


# # //////////////////////////////////////////////////////////////////////////////
# """Analysis 6b. Reduce data set with Factor Analysis"""
# # //////////////////////////////////////////////////////////////////////////////
# featureSet = wholeSet
# groupVar = 'ExpGroups'
# nLatentVars = 20
# F_OIdx = SNSSDf[SNSSDf['ExpGroups'] == 1]
# S_OIdx = SNSSDf[SNSSDf['ExpGroups'] == 2]
# B_OIdx = SNSSDf
# # Below changes AgeBins to numerical, THIS NEEDS TO CHANGE!
# # F_OIdx['AgeBins'] = F_OIdx['AgeBins'].cat.codes
# # S_OIdx['AgeBins'] = S_OIdx['AgeBins'].cat.codes
# # B_OIdx['AgeBins'] = B_OIdx['AgeBins'].cat.codes
#
# datasets = [F_OIdx, S_OIdx, B_OIdx]
# ds_tag = ['functional', 'structural', 'all']
# covDict = {'Functional': {'ds': F_OIdx, 'cov': [], 'corr': [], 'label': 'Functinonal'},
#            'Functional': {'ds': S_OIdx, 'cov': [], 'corr': [], 'label': 'Functinonal'},
#            'Functional': {'ds': B_OIdx, 'cov': [], 'corr': [], 'label': 'Functinonal'},
#            'Functional': {'ds': F_OIdx, 'cov': [], 'corr': [], 'label': 'Functinonal'}}
# for ds in datasets:
#
#
# for dsi, ds in enumerate(datasets):
#     fig = plt.figure(figsize=[20, 10])
#     data = ds[wholeSet].dropna(subset=wholeSet)
#     # for var in wholeSet:
#     #     if not isinstance(data[var].values[0], np.float):
#     #         data[var] = data[var].cat.codes
#     #         print('Converted categorical into float')
#     # data.shape
#     cov = np.cov(data, rowvar=0)
#     corr = np.corrcoef(data,rowvar=0)
#     titles = ['Covariance Matrix', 'Correlation Matrix']
#     labels = data[wholeSet].columns
#     for i, m in enumerate([cov, corr]):
#         ax = fig.add_subplot(1, 2, i+1)
#         sb.heatmap(m,
#                    vmin=-1, vmax=1, center=0,
#                    cmap=sb.diverging_palette(20, 220, n=200),
#                    square=True,
#                    ax=ax)
#         ax.set_title(titles[i])
#         ax.set_xticks(range(len(m))+np.ones(len(m))*0.5)
#         ax.set_xticklabels(labels, ha='right', rotation=45, fontsize=3)
#         ax.set_yticks(range(len(m))+np.ones(len(m))*0.5)
#         ax.set_yticklabels(labels, ha='right', rotation=0, fontsize=3)
#     fig.savefig('output/6b_'+ds_tag[dsi]+'_covarianceMatrix.pdf', dpi=300,
#                 format='pdf', pad_inches=0.1, bbox_inches='tight')
#     plt.close()
#
# # Also plot difference incovariance and correlation matrices between F & S
# FCov = np.cov(F_OIdx[wholeSet].dropna(subset=wholeSet), rowvar=0)
# SCov = np.cov(S_OIdx[wholeSet].dropna(subset=wholeSet), rowvar=0)
# FCorr = np.corrcoef(F_OIdx[wholeSet].dropna(subset=wholeSet), rowvar=0)
# SCorr = np.corrcoef(S_OIdx[wholeSet].dropna(subset=wholeSet), rowvar=0)
# FvsSCov = FCov-SCov
# FvsSCorr = FCorr-SCorr
#
# fig = plt.figure(figsize=[20, 10])
# titles = ['Covariance Matrix', 'Correlation Matrix']
# labels = F_OIdx[wholeSet].columns
#
# for i, m in enumerate([FvsSCov, FvsSCorr]):
#     ax = fig.add_subplot(1, 2, i+1)
#     sb.heatmap(m,
#                vmin=-1, vmax=1, center=0,
#                cmap=sb.diverging_palette(20, 220, n=200),
#                square=True,
#                ax=ax)
#     ax.set_title(titles[i])
#     ax.set_xticks(range(len(m))+np.ones(len(m))*0.5)
#     ax.set_xticklabels(labels, ha='right', rotation=45, fontsize=3)
#     ax.set_yticks(range(len(m))+np.ones(len(m))*0.5)
#     ax.set_yticklabels(labels, ha='right', rotation=0, fontsize=3)
# fig.savefig('output/6b_FvsS_covarianceMatrix.pdf', dpi=300,
#             format='pdf', pad_inches=0.1, bbox_inches='tight')
#
# nComp = 20
# labels = [featMetaData[var]['label'] for var in F_OIdx[wholeSet].columns]
# efaData = B_OIdx[wholeSet].dropna(subset=wholeSet)
# efa = decomposition.FactorAnalysis(n_components=20,
#                                    copy=True,
#                                    tol=1e-30)
# efaFit = efa.fit_transform(efaData)
# efaFitCorr = np.corrcoef(efaData, efaFit, rowvar=0)
# factorCorrM = efaFitCorr[0:efaData.shape[1], efaData.shape[1]:efaData.shape[1]+nComp]
# fig = plt.figure(figsize=[10, 20])
# ax = fig.add_subplot(111)
# sb.heatmap(factorCorrM,
#            vmin=-1, vmax=1, center=0,
#            cmap=sb.diverging_palette(20, 220, n=200),
#            square=True,
#            ax=ax)
#
# ax.set_title('Factor Correlation Matrix')
# ax.set_yticks(range(factorCorrM.shape[0])+np.ones(factorCorrM.shape[0])*0.5)
# ax.set_yticklabels(labels, ha='right', rotation=0, fontsize=5)
# ax.set_xticks(range(factorCorrM.shape[1])+np.ones(factorCorrM.shape[1])*0.5)
# ax.set_xticklabels(['EFA'+str(f) for f in range(factorCorrM.shape[1])], ha='right', rotation=45, fontsize=5)
# fig.savefig('output/6b_Factor_CorrelationMatrix.pdf', dpi=300,
#             format='pdf', pad_inches=0.1, bbox_inches='tight')
# plt.close()
#
#
# efaDf = pd.DataFrame(efaFit, columns=['EFA'+str(f) for f in range(factorCorrM.shape[1])])
# efaDf['ExpGroups'] = B_OIdx.dropna(subset=wholeSet)['ExpGroups'].values
# efaDf['Diagnosis'] = B_OIdx.dropna(subset=wholeSet)['Diagnosis'].values
# efaDf['T2_HealthChange'] = B_OIdx.dropna(subset=wholeSet+['T2_HealthChange'])['T2_HealthChange'].values
# efaDf['T0T2_SF12_binaryNormedMCS'] = B_OIdx.dropna(subset=wholeSet)['T0T2_SF12_binaryNormedMCS'].values
#
# tsne = manifold.TSNE(n_components=2, perplexity=80)
# tsneFit = tsne.fit_transform(efaDf.iloc[:, 0:2])
# efaDf['tsne0'], efaDf['tsne1'] = tsneFit[:,0], tsneFit[:,1]
#
# fig = plt.figure(figsize=[10, 10])
# ax = fig.add_subplot(111)
# sb.scatterplot(x='EFA0', y='EFA1', hue='ExpGroups', data=efaDf, ax=ax)
# sb.scatterplot(x='tsne0', y='tsne1', hue='Diagnosis', data=efaDf, ax=ax)

# NEED TO WORK OUT HOW TO PLOT A SCREE PLOT AND FINISH EFA
#
# fig
# factorCorrM
# w, v = np.linalg.eig(factorCorrM)
# w
