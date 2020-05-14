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
./train tweets_election.data
equals to 
./train tweets_election.data 0.1 0.5 10 10 0.75 200 0


Result files:
'model.out': Top 30 words for each topic and discourse
'result.out': Evaluation results as well as model parameters
'score.out': Predicting scores for test data
