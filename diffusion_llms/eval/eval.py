"""
Reads the predictions stored in this folder and generates the result table.

Prediction are named:

logit_25
logit_50
logit_75
regression_avg
regression_concat
classification_token
bert_class
bert_regr

.npy

This scripts reads each file and computes results.
"""