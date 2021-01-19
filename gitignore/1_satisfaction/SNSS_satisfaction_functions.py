#!/usr/bin/env python3.6
"""This module describes functions for analysis of the SNSS Dataset"""

import SNSS_Prognosis_functions as oss
import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW
from statsmodels.discrete.discrete_model import Logit
from textwrap import wrap
import seaborn as sb
import matplotlib.pyplot as plt


def SatisfactionFollowUpandBaselineComparison(df):
    def sigTest(G, varList, vType, df):
        sigDict = {}
        if vType == 'cat':
            for v in varList:
                T = pd.crosstab(index=df[G], columns=df[v],
                                margins=False, normalize=False)
                chi2Stat, chi2p, _, _ = stats.chi2_contingency(T, correction=True)
                cats = np.unique(df[v].dropna())
                if len(cats) == 2:
                    OR, _ = stats.fisher_exact(T)
                else:
                    OR = np.nan
                sigDict[v] = [chi2p, chi2Stat, OR]
        elif vType == 'cont':
            for v in varList:
                if G == 'ExpGroups':
                    Gi = [1, 2]
                elif (G == 'T2HCData') | (G == 'T1HCData') | (G == 'T1_Satisfaction_Bool'):
                    Gi = [0, 1]
                cm = CompareMeans.from_data(df[v][(df[G] == Gi[0]) & (df[v].notna())],
                                            df[v][(df[G] == Gi[1]) & (df[v].notna())])
                tStat, tp, _ = cm.ttest_ind()
                cohend = oss.cohen_d(cm.d1.data, cm.d2.data)
                sigDict[v] = [tp, tStat, cohend]
        sigT = pd.DataFrame.from_dict(sigDict, orient='index', columns=['p', 'stat', 'effect'])
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

    contVars = ['Age', 'T0_PHQ13_Total', 'T0_SF12_MCS', 'T0_SF12_PCS', 'T0_HADS', 'T0_Satisfaction_Total']
    catVars = ['Gender', 'T0_NegExpectation', 'Diagnosis', 'T0_SIMD04_bin']

    groupVar = 'T1_Satisfaction_Bool'
    catT = varTables(G=groupVar, varList=catVars, vType='cat', df=df)
    catStats = sigTest(G=groupVar, varList=catVars, vType='cat', df=df)
    catT.transpose().to_csv('output/sat/0_FollowUpCategoricalTable.tsv', sep='\t')
    catStats.transpose().to_csv('output/sat/0_FollowUpCategoricalStats.tsv', sep='\t')

    contT = varTables(G=groupVar, varList=contVars, vType='cont', df=df)
    contStats = sigTest(G=groupVar, varList=contVars, vType='cont', df=df)
    contT.transpose().to_csv('output/sat/0_FollowUpContinuousTable.tsv', sep='\t')
    contStats.transpose().to_csv('output/sat/0_FollowUpContinuousStats.tsv', sep='\t')

    groupVar = 'ExpGroups'
    catT = varTables(G=groupVar, varList=catVars, vType='cat', df=df[df.T2HCData == 1])
    catStats = sigTest(G=groupVar, varList=catVars, vType='cat', df=df[df.T2HCData == 1])
    catT.transpose().to_csv('output/sat/0_BaselineCategoricalTable.tsv', sep='\t')
    catStats.transpose().to_csv('output/sat/0_BaselineCategoricalStats.tsv', sep='\t')

    contT = varTables(G=groupVar, varList=contVars, vType='cont', df=df[df.T2HCData == 1])
    contStats = sigTest(G=groupVar, varList=contVars, vType='cont', df=df[df.T2HCData == 1])
    contT.transpose().to_csv('output/sat/0_BaselineContinuousTable.tsv', sep='\t')
    contStats.transpose().to_csv('output/sat/0_BaselineContinuousStats.tsv', sep='\t')
    return


def ExplainedSatisfactionComparison(df, groupVar):
    def sigTest(G, varList, vType, df):
        sigDict = {}
        if vType == 'cat':
            for v in varList:
                T = pd.crosstab(index=df[G], columns=df[v],
                                margins=False, normalize=False)
                chi2Stat, chi2p, _, _ = stats.chi2_contingency(T, correction=True)
                if (T.shape[0] == 2) & (T.shape[1] == 2):
                    OR, _ = stats.fisher_exact(T)
                else:
                    OR = np.nan
                sigDict[v] = [chi2p, chi2Stat, OR]
        elif vType == 'cont':
            for v in varList:
                if G == 'ExpGroups':
                    Gi = [1, 2]
                elif (G == 'T2HCData') | (G == 'T1HCData') | (G == 'T1_Satisfaction_Bool'):
                    Gi = [0, 1]
                elif G == 'Diagnosis':
                    Gi = [1, 2, 3, 4]
                if len(Gi) == 2:
                    cm = CompareMeans.from_data(df[v][(df[G] == Gi[0]) & (df[v].notna())],
                                                df[v][(df[G] == Gi[1]) & (df[v].notna())])
                    tStat, tp, _ = cm.ttest_ind()
                    cohend = oss.cohen_d(cm.d1.data, cm.d2.data)
                    sigDict[v] = [tp, tStat, cohend]
                elif len(Gi) > 2:
                    anovaStat, anovap = stats.f_oneway(*[df[v][(df[G] == i) &
                                                         (df[v].notna())] for i in Gi])
                    sigDict[v] = [anovap, anovaStat, np.nan]
        sigT = pd.DataFrame.from_dict(sigDict, orient='index', columns=['p', 'stat', 'effect'])
        return sigT

    def varTables(G, varList, vType, df):
        if vType == 'cont':
            T = df.groupby(G)[varList].agg([('N', 'count'), ('Mean', 'mean'),
                                            ('SD', 'std'), ('CI', lambda x:
                                            tuple(np.round(
                                             DescrStatsW(x.dropna()).
                                             tconfint_mean(), 2)))])
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

    catVars = ['T0_Sat1', 'T0_Sat2', 'T0_Sat3', 'T0_Sat4',
               'T0_Sat5', 'T0_Sat6', 'T0_Sat7', 'T0_Sat8',
               'T0_IPQ3', 'T0_IPQ4', 'T0_IPQ5', 'T0_IPQ6', 'T0_Wbelieve',
               'T1_Sat1', 'T1_Sat2', 'T1_Sat3', 'T1_Sat4',
               'T1_Sat5', 'T1_Sat6', 'T1_Sat7', 'T1_Sat8',
               'T1_IPQ3', 'T1_IPQ4', 'T1_IPQ5', 'T1_IPQ6', 'T1_Wbelieve']

    contVars = ['T0_Sat1', 'T0_Sat2', 'T0_Sat3', 'T0_Sat4',
                'T0_Sat5', 'T0_Sat6', 'T0_Sat7', 'T0_Sat8',
                'T1_Sat1', 'T1_Sat2', 'T1_Sat3', 'T1_Sat4',
                'T1_Sat5', 'T1_Sat6', 'T1_Sat7', 'T1_Sat8']

    catT = varTables(G=groupVar, varList=catVars, vType='cat', df=df)
    catStats = sigTest(G=groupVar, varList=catVars, vType='cat', df=df)
    catT.transpose().to_csv('output/sat/1_ExplainedSatisfactionTables_Cat_'+groupVar+'.tsv', sep='\t')
    catStats.transpose().to_csv('output/sat/1_ExplainedSatisfactionStats_Cat_'+groupVar+'.tsv', sep='\t')

    contT = varTables(G=groupVar, varList=contVars, vType='cont', df=df)
    contStats = sigTest(G=groupVar, varList=contVars, vType='cont', df=df)
    contT.transpose().to_csv('output/sat/1_ExplainedSatisfactionTables_Cont_'+groupVar+'.tsv', sep='\t')
    contStats.transpose().to_csv('output/sat/1_ExplainedSatisfactionStats_Cont_'+groupVar+'.tsv', sep='\t')
    return


def plotSatisfactionMeanBars(df, SNSSvars):
    satVars = ['T0_Sat1', 'T0_Sat2', 'T0_Sat3', 'T0_Sat4',
               'T0_Sat5', 'T0_Sat6', 'T0_Sat7', 'T0_Sat8',
               'T1_Sat1', 'T1_Sat2', 'T1_Sat3', 'T1_Sat4',
               'T1_Sat5', 'T1_Sat6', 'T1_Sat7', 'T1_Sat8']

    ylabels = [SNSSvars[S]['label'] for S in satVars]
    ylabels = ['\n'.join(wrap(l, 40)) for l in ylabels]

    fig1 = plt.figure(num=1, figsize=(16, 6), dpi=200, frameon=False)
    i = 0
    ii = 0
    ax = []
    for o in satVars:
        if (i < 8) & (ii == 0):
            ax.append(fig1.add_subplot(2, 4, i+1))
            pal = sb.cubehelix_palette(4, start=2.6, rot=.1, reverse=True)
        elif (i == 8) & (ii == 0):
            i = 0
            ii = 1
            pal = sb.cubehelix_palette(4, start=1, rot=.3, reverse=True)
        elif (i > 8) & (ii == 1):
            pal = sb.cubehelix_palette(4, start=1, rot=.3, reverse=True)
        sb.barplot(x='Diagnosis', y=o,
                   data=df, ax=ax[i],
                   palette=pal,
                   **{'alpha': 1})
        ax[i].set_xticklabels(labels=['NA', 'S', 'L', 'C'], fontsize=6)
        ax[i].set_ylim([1, 5])
        ax[i].set_ylabel(ylabels[i], fontsize=8, wrap=True)
        i = i+1

    fig1.subplots_adjust(wspace=0.4, hspace=0.5)
    fig1.savefig('output/sat/1_ExplainedSatisfactionBarsMeans.pdf', dpi=300,
                 format='pdf', pad_inches=0.1, bbox_inches='tight')
    plt.close()
    return True


def SNSSSatisfactionRegressionAnalysis(df, outcome):
    """ Multivariate analysis of predictors of poor outcome """
    SNSSvars = oss.loadJSON('raw_data/SNSS_vars.json')
    MVPredictorTables = {}
    GLabels = ['Functional', 'Structural']
    MVPredictors = ['AgeBins', 'Gender', 'Diagnosis', 'T0_SIMD04_bin', 'T0_SF12_PF',
                    'T0_Sat1_Poor_Bin', 'T0_Sat2_Poor_Bin',
                    'T0_Sat3_Poor_Bin', 'T0_Sat4_Poor_Bin',
                    'T0_Sat5_Poor_Bin', 'T0_Sat6_Poor_Bin',
                    'T0_Sat7_Poor_Bin', 'T0_Sat8_Poor_Bin']
    mdl = {}
    for G in [1, 2]:
        rDat = df[(df.ExpGroups == G) & (df[outcome].notna())]
        rDat = rDat.dropna(subset=MVPredictors)
        endog = np.asarray(rDat[outcome]).astype(int)
        exog = np.ones([len(rDat), 1]).astype(int)
        varNames = ['constant']
        sigTestIdx = {}
        for P in MVPredictors:
            # P = MVPredictors[1]
            varDat = rDat[P]
            if varDat.dtype.name != 'category':  # If not a categorical convert...
                varDat = pd.Categorical(varDat)

            # if 'Sat' in P:
            #     X = pd.get_dummies(varDat, drop_first=False).drop(axis=1, columns=5.0)
            if (P == 'T0_SF12_PF'):
                X = pd.get_dummies(varDat, drop_first=False).drop(axis=1, columns=100.0)
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
        mdl[G] = lr.fit(disp=0)
        coeffT = pd.DataFrame(mdl[G].params, index=[[P]*len(varNames), varNames],
                              columns=['coeff'])
        # coeffT = pd.DataFrame(mdl[G].params, index=[varNames, varVals],
        #                       columns=['coeff'])
        coeffT['coeffLCI'] = mdl[G].conf_int()[:, 0]
        coeffT['coeffUCI'] = mdl[G].conf_int()[:, 1]
        coeffT['OR'] = np.exp(mdl[G].params)
        coeffT['ORLCI'] = np.exp(mdl[G].conf_int())[:, 0]
        coeffT['ORUCI'] = np.exp(mdl[G].conf_int())[:, 1]
        coeffT['p'] = mdl[G].pvalues

        # Variable significance testing...
        pValT = []
        for P in MVPredictors:
            testLr = Logit(endog=endog, exog=np.delete(exog, sigTestIdx[P], axis=1))
            testMdl = testLr.fit(disp=0)
            Chi2p = 1 - stats.chi2.cdf(2*(mdl[G].llf - testMdl.llf), df=len(sigTestIdx[P]))
            pValT.append(pd.DataFrame(Chi2p, index=[P],
                                      columns=['llrp']))

        # Save individual coefficient details...
        coeffT.to_csv('output/sat/3_' + GLabels[G-1] + '_' + outcome + '_MVAnalysis.tsv', sep='\t')
        MVPredictorTables[GLabels[G-1]] = coeffT

        # Save significance tests...
        pd.concat(pValT).to_csv('output/sat/3_' + GLabels[G-1] + '_' + outcome + '_MVpVals.tsv', sep='\t')
    return MVPredictorTables, mdl


def exportForVictor(df):
    satVars = ['T0_Sat1', 'T0_Sat2', 'T0_Sat3', 'T0_Sat4',
               'T0_Sat5', 'T0_Sat6', 'T0_Sat7', 'T0_Sat8',
               'T1_Sat1', 'T1_Sat2', 'T1_Sat3', 'T1_Sat4',
               'T1_Sat5', 'T1_Sat6', 'T1_Sat7', 'T1_Sat8',
               'T0_IPQ6', 'T1_IPQ6',
               'T0_Wbelieve', 'T1_Wbelieve',
               'T0_IPQ3', 'T1_IPQ3',
               'T0_IPQ4', 'T1_IPQ4',
               'T0_IPQ5', 'T1_IPQ5']
    satDat = df[['Diagnosis'] + satVars]
    satDat.to_csv('output/sat/SNSSSatisfactionData.tsv', sep='\t')
    varDat = pd.read_json('raw_data/SNSS_vars.json')
    varDat[['Diagnosis'] + satVars].transpose().to_csv('output/sat/SNSSSatisfactionVariableLabels.tsv', sep='\t')
    return True
