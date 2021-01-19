"""This module describes functions for analysis of the SNSS Dataset"""
import ENSfunctions as oss
import importlib as imp
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
from colour import Color
import pandas as pd
import missingno as msno
from statsmodels.discrete.discrete_model import Logit
from statsmodels.genmod.generalized_linear_model import GLM
import statsmodels.genmod.families.family as mdlFamily
import statsmodels.genmod.families.links as mdlLink

import statsmodels.tools.tools.add_constant as add_constant
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW
from statsmodels.stats.proportion import proportion_confint
pd.options.display.max_columns = 20
pd.options.display.max_rows = 300
pd.options.mode.chained_assignment = None

imp.reload(oss)
raw = oss.import_ENS()

df, retentionTable = oss.ENSNullity(raw)
df = oss.ENSCompoundVariables(df)

# df = oss.ENSrmSQNA(df)
coreQs = ['ens1', 'ens2', 'ens3', 'ens4',
          'ens5', 'ens6', 'ens7', 'ens8',
          'ens9', 'ens10', 'ens11']

subQDict = {'ens1': {
                'subQs': ['ens1a', 'ens1b', 'ens1c', 'ens1d', 'ens1e', 'ens1f'],
                'formula': 'C(ens1a) + ens1b + ens1c + ens1d + ens1e + ens1f'
                },
            'ens2': {
                'subQs': ['ens2a', 'ens2b', 'ens2c', 'ens2d', 'ens2e'],
                'formula': 'ens2a + ens2b + ens2c + ens2d + ens2e'
                },
            'ens3': {
                'subQs': [],
                'formula': ''
                },
            'ens4': {
                'subQs': ['ens4a', 'ens4b', 'ens4c'],
                'formula': 'C(ens4a) + ens4b + ens4c'
                },
            'ens5': {
                'subQs': ['ens5a', 'ens5b', 'ens5c', 'ens5d'],
                'formula': 'ens5a + ens5b + ens5c + ens5d'
                },
            'ens6': {
                'subQs': ['ens6a'],
                'formula': 'ens6a'
                },
            'ens7': {
                'subQs': ['ens7a'],
                'formula': 'ens7a'
                },
            'ens8': {
                'subQs': [],
                'formula': ''
                },
            'ens9': {
                'subQs': [],
                'formula': ''
                },
            'ens10': {
                'subQs': [],
                'formula': ''
                },
            'ens11': {
                'subQs': ['ens11_ops', 'ens11_op_no_bin'],
                'formula': 'ens11_op_no_bin'
                }
            }
mdl = {}
for cQ in coreQs:
    if subQDict[cQ]['formula'] != '':
        exog = df[subQDict[cQ]['subQs']][df[cQ] == 1]
        endog = df['functional_pool'][df[cQ] == 1]
        print(['Modelling ' + cQ])
        # lr = Logit.from_formula(formula='functional_pool ~ ' + subQDict[cQ]['formula'],
        #                         data=cqDf)
        # mdl[cQ] = lr.fit()
        lr = GLM(endog=endog, exog=exog,
                 family=mdlFamily.Binomial(link=mdlLink.logit))
        xx = lr.fit()
        xx.summary()
    else:
        mdl[cQ] = ''

mdl['ens1'].summary()



lr[cQ] = Logit.from_formula(formula='functional_pool ~ ' + subQDict[cQ]['formula'],
                            data=cqDf)

mdl = lr.fit()
mdl.summary()
np.exp(mdl.params)
np.exp(mdl.conf_int())

msno.matrix(df, inline=False, fontsize=8, figsize=(10, 5))
