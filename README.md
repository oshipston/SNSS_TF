# Prognosis in functional and recognised pathophysiological neurological disorders - a shared basis.

**Running head**: Predicting prognosis in functional and pathophysiological neurological disorders.

**Department**: Centre for Clinical Brain Sciences, University of Edinburgh, Edinburgh, United Kingdom

**Authors**: Oliver Shipston-Sharman1, Stoyan Popkirov2, Christian H Hansen3, Jon Stone1, Alan Carson1

**Author Affiliations**
1. Centre for Clinical Brain Sciences, University of Edinburgh, Edinburgh, United Kingdom.
2. Department of Neurology, University Hospital Knappschaftskrankenhaus Bochum, Ruhr University Bochum, Bochum, Germany.
3. Faculty of Epidemiology and Population Health, London School of Hygiene and Tropical Medicine, London, United Kingdom.

## Abstract
**Background:** Predicting outcomes in functional neurological disorders remains a challenge with unique predictors and reliable models elusive. A comparative assessment of prognosis would facilitate identification of specific markers.

**Methods:** Data on 2581 patients from the Scottish Neurological Symptoms Study (SNSS) were used. Baseline data included health-related quality of life, anxiety and depressive symptoms, illness perceptions, consultation satisfaction, physical symptoms and demographics.The prospective cohort included neurology outpatients classified with a functional (reporting symptoms &#39;not at all&#39; or &#39;somewhat explained&#39; by &#39;organic disease&#39;; n = 716) or recognised pathophysiological disorder (&#39;largely&#39; or &#39;completely explained&#39;; n = 1865). Univariate and multivariable logistic regression and deep neural network (DNN) models were used to assess the capacity of self-reported markers at baseline to predict self-reported global clinical improvement (CGI) at 12-months.  
**Results:** Patients with functional and pathophysiological disorders report near identical outcomes at 12-months with 67% and 66% respectively reporting unchanged or worse CGI. In multivariable modelling &#39;negative expectation of recovery&#39; and &#39;disagreement with psychological attribution&#39; predicting same or worse outcome in both groups. Receipt of disability-related state benefit showed a complex relationship with outcome in the functional disorder group (OR = 2.28 (95%-CI: 1.36-3.84) for same or worse CGI in a group-stratified model) and was not related to a measure of economic deprivation. DNN models trained on all 92 baseline questionnaire items predicted poor outcome with area under the receiver-operator curve (AUROC) of only 0.67 in both groups.  
**Conclusions:** Those with functional and pathophysiological neurological disorder share similar outcomes, clinical trajectories, and poor prognostic markers in multivariable models. Predicting outcome was not possible using the baseline data in this study.

## Introduction
This repository contains all necessary functions and script files to run the analysis undertaken in the above manuscript. The directory excludes the raw data necessary to run the analyses due to patient confidentiality reasons but includes all the latest output tables and figures. The analysis pipeline is contained within SNSS\_Prognosis\_scripts.py. This lays out in a hopefully readable fashion the data pre-processing pipeline and analytical approach step-by-step. It imports SNSS\_Prognosis\_functions.py; which contains all the relevant functions required to run. The functions were developed over a significant period by Oliver Shipston-Sharman only and contain some idiosyncracies which I would welcome improvements on should you find any.

## Installation
I would recommend using anaconda to install all required dependancies:
```
conda env create --name snss_tf --file=environment.yml
```

## SNSS\_Prognosis\_scripts.py structure
### Data Pipeline 1. Import dataset from UofE SMB Datashare of local file
Step 1 of the data pipeline imports the original SAS7BDAT file in a pandas dataframe from
the remote UofE datastore. Gaining access to the original patient-level data is at
the discretion of the research group and initial study ethics approval.
### Data Pipeline 2. Quantify those lost to follow up, add binary follow up variables
Step 2 of the pipeline ensures index integrity and assesses attrition over the study period.
A table summarising this is output in the process.  
Output:
1. output/0_SNSS_retention.tsv
### Data Pipeline 3. Preprocess Raw Data
Step 3 adds custom variables to the original dataframe. These include user selected binarisation
of categorical variables or compound scores such as illness worry which is the sum of 3 whitely index variables.  
```python
IllnessWorry = Wworry + Wseriousworry + Wattention
```
### Data Pipeline 4. Declare SNSS specific feature sets and dummy vars
Step 4 is an administrative step of declaring variable groups and custom dictionaries
for later use in the analysis.
### Analysis 0. Compare lost to follow up and follow up groups
This section assesses baseline differences between functional and pathophysiological groups as well as those lost to follow up within this grouping.  
Output:
1. output/0_OutcomeContinuousTable.tsv
2. output/0_OutcomeContinuousStats.tsv
3. output/0_OutcomeCategoricalTable.tsv
4. output/0_OutcomeCategoricalStats.tsv
5. output/0_MVPredictorsForestPlotT2_HCData.pdf
6. output/0_MVAnalysis_NotExplainedT2_HCData.tsv
7. output/0_MVAnalysis_ExplainedT2_HCData.tsv
8. output/0_FollowUpContinuousTable.tsv
9. output/0_FollowUpContinuousStats.tsv
10. output/0_FollowUpCategoricalTable.tsv
11. output/0_FollowUpCategoricalStats.tsv
12. output/0_BaselineContinuousTable.tsv
13. output/0_BaselineContinuousStats.tsv
14. output/0_BaselineCategoricalTable.tsv
15. output/0_BaselineCategoricalStats.tsv

These are summarised in:
1. Tables/Manuscript Table 1 - Baseline Explained Comparison.pdf

### Analysis 1. Compare outcomes between functional category
Output:
1. /Users/oss/Documents/code/SNSS_TF/output/1_PrimaryOutcomebyGroup.tsv
2. /Users/oss/Documents/code/SNSS_TF/output/1_PrimaryOutcomePoolbyGroup.tsv
3. /Users/oss/Documents/code/SNSS_TF/output/1_SNSSPrimaryOutcomeBars.pdf
4. /Users/oss/Documents/code/SNSS_TF/output/1_SNSSPrimaryOutcomeStackedBars.pdf

Tables/Manuscript Table 2 - Primary Outcomes by Group.pdf
### Analysis 2. Assess secondary outcomes
Output:
1. /Users/oss/Documents/code/SNSS_TF/output/2_SecondaryOutcomeMeasures.tsv
2. /Users/oss/Documents/code/SNSS_TF/output/2_SNSSSecondaryOutcomePlot.pdf

### Analysis 3. Validate Scottish Index of Multiple Deprivation 2004 Interactions
Output:
1. /Users/oss/Documents/code/SNSS_TF/output/3_DeprivationGroups.tsv
2. /Users/oss/Documents/code/SNSS_TF/output/3_DeprivationOutcomeAndGroup.tsv
3. /Users/oss/Documents/code/SNSS_TF/output/3_SNSSSocioeconomicAssessment.pdf
4. /Users/oss/Documents/code/SNSS_TF/output/3_SNSSSocioeconomicDomainsAssessment_T0_IncapacityBenefitorDLA.pdf
5. /Users/oss/Documents/code/SNSS_TF/output/3_SNSSSocioeconomicDomainsAssessment_T0_Inemployment.pdf
6. /Users/oss/Documents/code/SNSS_TF/output/3_SNSSSocioeconomicGroups.pdf

### Analysis 4. Compute univariate odds ratios for poor outcome
Output:
1. 4_UVAnalysis_ExplainedT2_poorCGI.tsv
2. 4_UVAnalysis_NotExplainedT2_poorCGI.tsv
3. 4_UVPredictorsForestPlotT2_poorCGI.pdf

### Addendum to Analysis 4: Combined analysis with diagnosis as predictor
Output:
1. 4adapted_UVAnalysis_AllT2_poorCGI.tsv
2. 4adapted_UVPredictorsForestPlotT2_poorCGI.pdf
### Analysis 5. Compute multivariate odds ratios for poor outcome
Output:
1. 5_MVAnalysis_ExplainedT2_poorCGI.tsv
2. 5_MVAnalysis_NotExplainedT2_poorCGI.tsv
3. 5_MVPredictorsForestPlotT2_poorCGI.pdf

### Addendum to Analysis 5: Combined analysis with diagnosis as predictor
Output:
1. 5adapted_MVAnalysis_AllT2_poorCGI.tsv
2. 5adapted_MVAnalysis_AllT2_strictPoorCGI.tsv
3. 5adapted_MVPredictorsForestPlotT2_poorCGI.pdf
4. 5adapted_MVPredictorsForestPlotT2_strictPoorCGI.pdf  
### Analysis 6 Reduce data set with UV regression coefficients/p-vals
Output:
1. 6_UVPredictorsForestPlotT2_poorCGI.pdf
2. 6a_Reduced_Explained_MVPredictorsForestPlotT2_poorCGI.pdf
3. 6a_Reduced_Not Explained_MVPredictorsForestPlotT2_poorCGI.pdf
4. 6a_UVPredictorsForestPlotT2_poorCGI.pdf
5. 6a_WholeSet_UVAnalysis_ExplainedT2_poorCGI.tsv
6. 6a_WholeSet_UVAnalysis_NotExplainedT2_poorCGI.tsv
7. 6b_Factor_CorrelationMatrix.pdf
8. 6b_FvsS_covarianceMatrix.pdf
9. 6b_FvsS_factor_CorrelationMatrix.pdf
10. 6b_all_covarianceMatrix.pdf
11. 6b_functional_covarianceMatrix.pdf
12. 6b_structural_covarianceMatrix.pdf  
### Analysis 7. NN Assessment of SNSS Prognosis
The output comprises
#### cvModels:
The training summary of all 10 cross-validation runs for each experiment. Only network architectures/hyperparameters that performed best as determined by accuracy reached cross-validation stage.  
#### fullNNAnalysis:
Performance metrics of the cross-validation runs with each runs individual and aggregate performance. Both TSV files of the runs and pickles of each run are exported.  
#### modelCheckpoints:
Logging directory used for storing .h5 model checkpoints at the optimal performance point during training. Represented by circles on the accuracy plot in the training summary figures.  
#### paramSweeps:
Bar charts comparing performance metrics of each cross validation run as well as hyperparameter sweeps.  
#### paramSweepsModels:
Core summary figures of the network training runs, there is a figure outlining the evolution of the network over the training epochs. Both training and evaluation accuracy and cross entropy are plotted up until the early stopping point. Performance metrics and histograms of network scores for each label are also shown.



```python

```
