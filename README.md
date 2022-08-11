# HoltWinters
# Language: Python
# Input: TXT
# Output: PNG
# Tested with: PluMA 1.1, Python 3.6

PluMA plugin to perform Holt-Winters (Holt, 1978) weather forecasting.

The plugin expectes as input a tab-delimited file of keyword-value pairs:
csvfile: Dataset
trainend: Last row of training data
seasonal: Is data seasonal (True/False)?
trainparam: Parameter for training

Predictions are output in the form of a graph, in PNG format (the user-specified output file).
