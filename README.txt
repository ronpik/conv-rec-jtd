This code is for My NAACL2018 paper: Microblog Conversation Recommendation via Joint Modeling of Topics and Discourse.

This code is implemented based on HFT model(McAuley and Leskovec, 2013), original code can be found in http://cseweb.ucsd.edu/~jmcauley/code/code_RecSys13.tar.gz.


You will need to export liblbfgs to your LD_LIBRARY_PATH to run the code. 
(try:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/liblbfgs-1.10/lib/.libs/
or:
setenv LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:$PWD/liblbfgs-1.10/lib/.libs/)


Compile using "make".


Run using ./train and specifying an input file (e.g. test.data). Input files should be like the following form: 
For each line: 
[Conv ID] + '\t' + [Msg ID] + '\t' + [Parent ID] + '\t' + [Original sentence] + '\t' + [words after preprocessing] + '\t' + [User ID]

Optional parameters:
lambda(default: 0.1)  topic_proportion(default: 0.5)  number_of_topic(default: 10)  number_of_discourse(default: 10)  training_percentage(default: 0.75)  confidence_paramter(default: 200)  running_time(default: 0)

For example:
./train test.data
equals to 
./train test.data 0.1 0.5 10 10 0.75 200 0


Result files:
'model.out': Top 30 words for each topic and discourse
'result.out': Evaluation results as well as model parameters
'score.out': Predicting scores for test data
'learning.out': learning steps during training (to plot learning curve)

CODE AND MODELS
======================================
there are 3 relevant branches:
master in order to run the CT_JTD model
svd++ to run the CR_JTD++ model
structure-user-preferences - to run the CR_JTD-Struct model

DATA
==============================
there are 4 relevant datasets that can be found in the directory 'included-data-and-scripts':
TREC-full.data - TREC dataset
US_Election-full.data - US Election dataset
cmv-random-branches.data - CMV dataset
cmv-freq-convs.data CMV full conversations dataset

RESULTS
=========================
the resultsd of the models over the various datasets can be found in the *-results directories (under 'included-data-and-scripts' directory)

SCRIPTS AND NOTEBOOKS
===========================
graph_embedding.ipynb - create graph embeddings to a given dataset
plor-learning-curve.ipynb - calculate learning curve given a *learning.out file
datasets-stats.ipynb - plot statistics of a given dataset
process_cmv_data.ipynb - create the two cmv datasets (CMV and CMV full conversations) this file 						needs a Reddit data which weight 0.5GB, and cannot be uploaded to Github.

tweets2data.py - takes a set of tweets (in jsonl format) and create a dataset suitable to the 					models with the raw text inside. the outputed dataset should be processed by 					the file 'text_processing.py'
text_processing.py - take a dataset with a raw text, and process the text to created a processed 						version of it, to remove some redundant noise
twokenize.py - a helper module for text_processing.py - used to   tokenize a raw text, specified 				for twitter texts. this script was taken from the python code of TweetNLP 
			   (http://www.cs.cmu.edu/~ark/TweetNLP/)