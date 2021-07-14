import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
import streamlit as st
import optuna
    
#import lightgbm as lgb
import optuna.integration.lightgbm as lgb
import optuna


train = pd.read_csv('train.csv')
train['train'] = True
test = pd.read_csv('test.csv')
test['train'] = False

alldata = train.append(test).copy()

# Continuous features
alldata['FamilyMembers'] = alldata['SibSp'] + alldata['Parch']
alldata['AdjFare'] = alldata.groupby('Ticket')['Fare'].transform(lambda x: x/len(x))
# Categorical features
alldata['Female'] = alldata['Sex'].map({'male': 0, 'female': 1}).astype(np.int8)
alldata['Embarked'] = alldata['Embarked'].map({'S':2, 'C':1, 'Q': 0, np.NaN: 2}).astype(np.int8) # Missing values assigned to majority class (Southampton)
alldata['Pclass_cat'] = (alldata['Pclass']-1).astype(np.int8)
alldata['Male3rd'] = (alldata['Sex'].map({'male': 1, 'female': 0}) * alldata['Pclass'].map({3:1, 1:0, 2:0})).astype(np.int8)
alldata['Adult'] = (alldata['Age']>16).astype(np.int8)
alldata['Adult'].values[alldata['Age'].isna()] = 1 # Looking at titles of people, I suspect those with missing age are mostly adult
alldata['MissingAge'] = (train['Age'].isna()*1).astype(np.int8)
alldata['NonAdult1st2nd'] = (alldata['Adult'] * alldata['Pclass'].map({3:0, 1:1, 2:1})).astype(np.int8)


alldata['Female1st2nd'] = (alldata['Female'] * alldata['Pclass'].map({3:0, 1:1, 2:1})).astype(np.int8)


# Imputation of age
alldata['Age'] = alldata.groupby(['Pclass', 'Female'])['Age'].transform(lambda x: x.fillna(x.median()))

# Family member by age interaction
alldata['FamilyAge'] = alldata['FamilyMembers'] + alldata['Age']/60
# Taking a guess as to which passengers are parents
alldata['father'] = 1 * (alldata['Age']>=18) * (alldata['Parch']>0) * (alldata['Sex']=='male')
alldata['mother'] = 1 * (alldata['Age']>=18) * (alldata['Parch']>0) * (alldata['Sex']=='female')
alldata['parent'] = alldata['father'] + alldata['mother']


alldata['title_type2'] = [ any([title in Name for title in ['Capt.', 'Col.', 'Major.', 'Rev.']]) for Name in alldata['Name']]
alldata['title_type1'] = [ any([title in Name for title in ['Master.', 'Mme.', 'Dona.', 'Countess.', 'Lady.', 'Miss.', 'Mlle.']]) for Name in alldata['Name']]

alldata['title_type'] = alldata['title_type1']*1 + alldata['title_type2']*2

alldata['AgeGroup'] = 1 * (alldata['Age']<=2) + 1 * (alldata['Age']<=6) + 1 * (alldata['Age']<=17) + 1 * (alldata['Age']<=60)

# Lists of features to be used later
continuous_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'FamilyMembers', 'AdjFare', 'FamilyAge']
discrete_features = ['Female', 'Pclass_cat', 'Male3rd', 'Embarked', 'Adult', 'NonAdult1st2nd', 
                     'Female1st2nd', 'MissingAge', 'father', 'mother', 'parent', 'title_type', 'AgeGroup']
                     
ids_of_categorical = [0,1,2,3,4,5,6,7,8,9,10,11,12]

train = alldata[alldata['train']==True]
test = alldata[alldata['train']==False]

X = np.array( train[discrete_features + continuous_features] )    
y = np.array( train['Survived'] ).flatten()

dtrain = lgb.Dataset(X, label=y)
st.write(dtrain)

# We will track how many training rounds we needed for our best score.
# We will use that number of rounds later.
best_score = 999
training_rounds = 10000

rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
# Declare how we evaluate how good a set of hyperparameters are, i.e.
# declare an objective function.
def objective(trial):
    # Specify a search space using distributions across plausible values of hyperparameters.
    param = {
        "objective": "binary",
        "metric": "binary_error",
        "verbosity": -1,
        "boosting_type": "gbdt",                
        "seed": 42,
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 512),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.1, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.1, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 0, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
        'seed': 1979
    }
    
   # Run LightGBM for the hyperparameter values
    lgbcv = lgb.cv(param,
                   dtrain,
                   categorical_feature=ids_of_categorical,
                   folds=rkf,
                   verbose_eval=False,                   
                   early_stopping_rounds=250,                   
                   num_boost_round=10000,                    
                   callbacks=[lgb.reset_parameter(learning_rate = [0.005]*200 + [0.001]*9800) ]
                  )
    
    cv_score = lgbcv['binary_error-mean'][-1] + lgbcv['binary_error-stdv'][-1]
    if cv_score<best_score:
        training_rounds = len( list(lgbcv.values())[0] )
    
    # Return metric of interest
    return cv_score

# Suppress information only outputs - otherwise optuna is 
# quite verbose, which can be nice, but takes up a lot of space
optuna.logging.set_verbosity(optuna.logging.WARNING) 

# We search for another 4 hours (3600 s are an hours, so timeout=14400).
# We could instead do e.g. n_trials=1000, to try 1000 hyperparameters chosen 
# by optuna or set neither timeout or n_trials so that we keep going until 
# the user interrupts ("Cancel run").
study = optuna.create_study(direction='minimize')  
#study.enqueue_trial(tmp_best_params)
study.optimize(objective, timeout=14400) 


optuna.visualization.plot_optimization_history(study)

st.write(study.best_params)

