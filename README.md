# oshipston/SNSS\_TF
## Prognosis in functional and pathophysiological neurological disorders: a shared basis.

**Running head**
Predicting prognosis in functional and pathophysiological neurological disorders.

**Department**
Centre for Clinical Brain Sciences, University of Edinburgh, Edinburgh, United Kingdom

**Authors**
Oliver Shipston-Sharman1, Stoyan Popkirov2, Christian H Hansen3, Jon Stone1, Alan Carson1,4

**Author Affiliations**
1. Centre for Clinical Brain Sciences, University of Edinburgh, Edinburgh, United Kingdom.
2. Department of Neurology, University Hospital Knappschaftskrankenhaus Bochum, Ruhr University Bochum, Bochum, Germany.
3. Faculty of Epidemiology and Population Health, London School of Hygiene and Tropical Medicine, London, United Kingdom.
4. Scottish Neurobehavioural Rehabilitation Unit, Royal Edinburgh Hospital, Edinburgh, United Kingdom.

## Abstract
**Objective:** To compare self-reported outcomes, clinical trajectory and utility of baseline questionnaire responses in predicting prognosis in functional and pathophysiological neurological symptoms.
**Methods:** Data on 2581 patients from the Scottish Neurological Symptoms Study (SNSS) were used. Baseline data included health-related quality of life, anxiety and depressive symptoms, illness perceptions, consultation satisfaction, physical symptoms and demographics.The prospective cohort included neurology outpatients classified with a functional (reporting symptoms &#39;not at all&#39; or &#39;somewhat explained&#39; by &#39;organic disease&#39;; n = 716) or recognised pathophysiological disorder (&#39;largely&#39; or &#39;completely explained&#39;; n = 1865). Univariate and multivariable logistic regression and deep neural network (DNN) models were used to assess the capacity of self-reported markers at baseline to predict self-reported global clinical improvement (CGI) at 12-months.
**Results:** Patients with functional and pathophysiological disorders report near identical outcomes at 12-months with 67% and 66% respectively reporting unchanged or worse CGI. In multivariable modelling &#39;negative expectation of recovery&#39; and &#39;disagreement with psychological attribution&#39; predicting same or worse outcome in both groups. Receipt of disability-related state benefit showed a complex relationship with outcome in the functional disorder group (OR = 2.28 (95%-CI: 1.36-3.84) for same or worse CGI in a group-stratified model) and was not related to a measure of economic deprivation. DNN models trained on all 92 baseline questionnaire items predicted poor outcome with area under the receiver-operator curve (AUROC) of only 0.67 in both groups.
**Conclusions:** Those with functional and pathophysiological neurological disorder share similar outcomes, clinical trajectories, and poor prognostic markers in multivariable models. Predicting outcome was not possible using the baseline data in this study.

## Introduction
This repository contains all necessary functions and script files to run the analysis undertaken in the above manuscript. The directory excludes the raw data necessary to run the analyses due to patient confidentiality reasons but includes all the latest output tables and figures. The analysis pipeline is contained within SNSS\_Prognosis\_scripts.py. This lays out in a hopefully readable fashion the data pre-processing pipeline and analytical approach step-by-step. It imports SNSS\_Prognosis\_functions.py; which contains all the relevant functions required to run.

## SNSS\_Prognosis\_scripts.py structure
### Data Pipeline 1. Import dataset from UofE SMB Datashare of local file
Step 1 of the data pipeline imports the original SAS7BDAT file in a pandas dataframe from
the remote UofE datastore. Gaining access to the original patient-level data is at
the discretion of the research group and initial study ethics approval.
### Data Pipeline 2. Quantify those lost to follow up, add binary follow up variables
Step 2 of the pipeline ensures index integrity and assesses attrition over the study period.
A table summarising this is output to "output/0\_SNSS\_retention.tsv"
### Data Pipeline 3. Preprocess Raw Data
### Data Pipeline 4. Declare SNSS specific feature sets and dummy vars
### Analysis 0. Compare lost to follow up and follow up groups
### Analysis 1. Compare outcomes between functional category
### Analysis 2. Assess secondary outcomes
### Analysis 3. Validate Scottish Index of Multiple Deprivation 2004 Interactions
### Analysis 4. Compute univariate odds ratios for poor outcome
### Addendum to Analysis 4: Combined analysis with diagnosis as predictor
### Analysis 5. Compute multivariate odds ratios for poor outcome
### Addendum to Analysis 5: Combined analysis with diagnosis as predictor
### Analysis 6 Reduce data set with UV regression coefficients/p-vals
### Analysis 7. NN Assessment of SNSS Prognosis


'''python

'''
