# piecing together all the different scripts

import train_model          # train model
import gather_scores1~5     # generate score pairs
import build1~5             # build first 5 fims, build second 5 fims, add all 10 fims
import fimsolve             # generate eigs
import sort                 # sort eig values
import plot_data2           # plot data

# is it better to do this as a batch script?
# perhaps
# do I want to parallelise? Probably after submitting one done in series, just in case parallel causes problems 




to do list
1. get other pictures of 7
2. train on them
3. 