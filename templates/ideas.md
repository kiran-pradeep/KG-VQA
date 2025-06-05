# Sub-Question Set Classifier

## Overview
1. We already have ground truth decompositions. These instances will be labeled as 1.
2. Distort the dataset with some noise (add_irrelevant/remove_relevant). These instances will be labeled as 0.
3. Train a classifier
    - `Input`: Precision_Q, Precision_C, Recall_Q, Sim_Q, Sim_C.
    - `Output`: Relevant (1) / Irrelevant (0) - set of sub-questions.