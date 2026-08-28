# A comparative analysis of the performance of machine learning models for the prediction of bank term deposit subscription 


MSc Data Science final project. I compare four classifiers on the UCI Bank Marketing dataset to predict whether a client subscribes to a term deposit after a telemarketing call.

Only 11.3% of the 41,188 contacts ended in a subscription, so the job of the model is to tell a call centre who is worth ringing.

## Results

LightGBM came out top: 0.523 F1 and 0.811 ROC-AUC on a test set I only touched once. Clients it flags are about 4.3 times more likely to subscribe than a random client.

I wouldn't call it a clear win though. The gap to Random Forest was 0.0046 and the fold-to-fold variation was 0.0099, so the two sit inside each other's noise. All four models landed between 0.778 and 0.784 AUC, which says the limit here is the data rather than the algorithm. LightGBM gets the nod mainly because it tuned in a quarter of Random Forest's time.

## Decisions worth knowing

**Dropped `duration`.** It correlates 0.405 with the target, the strongest thing in the dataset, but you only know how long a call lasted once it's over. Useless for deciding who to ring.

**Split before touching anything.** Scaling, encoding and SMOTE all sit inside the pipeline, so they only ever see training folds. Test set used once, at the end.

**Rebuilt `pdays`.** It uses 999 to mean "never contacted before". I split it into a yes/no flag and the actual gap, using `-1` for never contacted, because `0` already means "contacted today" in this data.

**Kept one economic indicator.** `emp_var_rate`, `euribor3m` and `nr_employed` correlate at 0.91 to 0.97. Keeping all three would have made the logistic regression coefficients meaningless.

## Running it

Open the notebook and run top to bottom. The data downloads itself.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn
```

About an hour on Colab, almost all of it the hyperparameter search. Lower `N_ITER` if you're impatient, but you'll search less of the space. Seed is fixed at 42, so you should get my numbers back.

## Limits

SMOTE produces fractional values on one-hot columns. Fine for trees, not clean. SMOTE-NC would fix it.

The split is random rather than chronological, so this says nothing about how the model holds up against next quarter's campaign.

Threshold sits at the default 0.5. Tuning it against what a call actually costs would make this a decision tool rather than a ranking one.

## Data

UCI Bank Marketing dataset, collected by a Portuguese bank between May 2008 and November 2010, published by Moro, Cortez and Rita (2014). Anonymised and publicly released for research.

The model uses age, job and marital status. Anything deployed along these lines would need a fairness audit first.



