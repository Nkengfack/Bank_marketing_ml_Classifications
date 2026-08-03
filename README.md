# Bank_marketing_ml_Classification
# Bank Term Deposit Subscription Prediction

A comparative performance analysis of machine learning classifiers on the UCI Bank Marketing dataset, built as the MSc Data Science final project (Module 7PAM2002, University of Hertfordshire).

**Research question:** which supervised machine learning model best predicts whether a bank client will subscribe to a term deposit, using only information available *before* the marketing call is made, and how large is the performance cost of enforcing that constraint?

The constraint matters. The strongest predictor in this dataset, call duration, does not exist until the call has ended, so a model that uses it cannot decide who to ring. Every model here is trained twice, with and without duration, to measure exactly what that leakage is worth.


