#!/usr/bin/env python3.6
""" This is a drafting script for SNSS_TF"""
import os
os.chdir("/Users/oss/documents/code/SNSS_TF")
import SNSS_Prognosis_functions as oss
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


"""Config"""
oss.moduleInit()
cfg = oss.loadJSON("config.json")
SNSSvars = oss.loadJSON("raw_data/SNSS_vars.json")

""" Data Pipeline 1. Import dataset from UofE SMB Datashare """
raw = oss.import_SNSS(usr='Nil', pwd='Nil', local_file=1)

""" Data Pipeline 2. Quantify those lost to follow up, add binary follow up variables"""
SNSSDf, retentionTable = oss.SNSSNullity(raw)  # Retention Table and FU variables added

""" Data Pipeline 2. Preprocess raw data"""
SNSSDf = oss.SNSSCompoundVariables(SNSSDf)

"""Analysis 1. Compare lost to follow up and follow up groups"""
# Do functional and structural/lost to follow up and followed up groups differ?
oss.FollowUpandBaselineComparison(SNSSDf)

"""Predictor Clustering"""
# Do SF12, HADS and symptom counts correlate with either reported outcome or functional group?
# Is there an underlying group structure not represented in the variable set collected?
oss.SNSSSecondaryMeasuresvsPrimaryOutcome(SNSSDf)

imp.reload(oss)
