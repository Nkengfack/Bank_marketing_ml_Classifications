# A comparative analysis of the performance of machine learning models for the prediction of bank term deposit subscription 

MSc Data Science final project. I compare four classifiers on the UCI Bank Marketing dataset to predict whether a client subscribes to a term deposit after a telemarketing call.

Most calls fail. Only 11.3% of the 41,188 contacts ended in a subscription, so the point of the model is to tell a call centre who is worth ringing.

## What I found

LightGBM came out on top, scoring 0.523 F1 and 0.811 ROC-AUC on a test set I only touched once. Clients it flags are about 4.3 times more likely to subscribe than a random client.

But I wouldn't call it a clear win. The gap to Random Forest was 0.0046, and the fold-to-fold variation was 0.0099, so the two are within noise of each other. All four models landed between 0.778 and 0.784 AUC. That says the limit here is the data, not the algorithm. LightGBM gets the nod mainly because it tuned in a quarter of Random Forest's time.

Two other things worth mentioning. Tuning on F1 instead of accuracy pushed the tree models from finding a third of subscribers to over half, and accuracy dropped while doing it. That's the right trade when a missed subscriber costs you revenue and a wasted call costs you a minute. And the Euribor rate dominates every model, which means this thing is really learning the 2008-2010 Portuguese economy. I wouldn't trust it in another period without retesting.

## Decisions I made

**Dropped `duration`.** It correlates 0.405 with the target, the strongest thing in the dataset, but you only know how long a call lasted after it's over. Useless for deciding who to ring. Flip `DROP_DURATION = False` if you want to see the inflated numbers.

**Split before touching anything.** Scaling, encoding and SMOTE all live inside the pipeline, so they only ever see training folds. Test set used once, at the end.

**Rebuilt `pdays`.** It uses 999 to mean "never contacted before", which a model reads as three years. I split it into a yes/no flag and the actual gap, using `-1` for never contacted, because `0` already means "contacted today" in this data.

**Kept one economic indicator.** `emp_var_rate`, `euribor3m` and `nr_employed` correlate at 0.91 to 0.97. Keeping all three would have made the logistic regression coefficients meaningless.

## Running it

Open the notebook and run top to bottom. The data downloads itself, nothing to set up locally.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn
```

Takes about an hour on Colab, almost all of it the hyperparameter search. Drop `N_ITER` from 30 if you're impatient, but you'll search less of the space. Seed is fixed at 42, so you should get my exact numbers back.

## Known limits

SMOTE interpolates across one-hot columns and produces fractional values where there should be 0s and 1s. Fine for trees, not clean. SMOTE-NC would fix it.

The split is random, not chronological, so this doesn't tell you how the model holds up against next quarter's campaign.

Threshold is sitting at the default 0.5. Tuning it against what a call actually costs would make this a decision tool rather than a ranking one.

Predicted probabilities are inflated by SMOTE. Use it to rank clients, not to read off someone's real chance of subscribing.

## Data

UCI Bank Marketing dataset, collected by a Portuguese bank between May 2008 and November 2010, published by Moro, Cortez and Rita (2014). Anonymised and publicly released for research.



