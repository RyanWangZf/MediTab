import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import xgboost

from pytrial.tasks.indiv_outcome.tabular import FTTransformer, TransTab, MLP
from pytrial.data.patient_data import TabularPatientBase

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import distilltab.dataset 


def run_xgboost(train_df, val_df, num_features, cat_features, bin_features, label_col):
        # initialize xgbclassifier
        # xgboost.set_config(verbosity=0)
        for col in cat_features + bin_features:
            labelencoder = LabelEncoder().fit(train_df[col])
            train_df[col] = labelencoder.transform(train_df[col])
            val_df[col] = labelencoder.transform(val_df[col])

        xgb_model = xgboost.XGBClassifier(objective="binary:logistic", 
                                    random_state=42,
                                    eval_metric='auc', 
                                    n_estimators=200, 
                                    max_depth=6, 
                                    learning_rate=0.2,
                                    early_stopping_rounds=5,
                                    callbacks=None)

        # train the model
        xgb_model.fit(
            train_df[num_features+cat_features+bin_features], train_df[label_col],
            eval_set=[(train_df[num_features+cat_features+bin_features], train_df[label_col])],
            # verbose=0
        )

        # evaluate the model
        y_pred = xgb_model.predict_proba(val_df[num_features+cat_features+bin_features])[:, 1]
        return y_pred

def run_fttransformer(train_df, val_df, num_features, cat_features, bin_features, label_col):
    cat_cardinalities = [len(np.unique(train_df[col])) for col in cat_features + bin_features]
    model = FTTransformer(
        num_feat=num_features,
        cat_feat=cat_features+bin_features,
        cat_cardinalities=cat_cardinalities,
        output_dim=1,
        mode='binary',
        device='cpu',
        epochs=50,
        )

    # sklearn-like API, train the model
    model.fit({'x': TabularPatientBase(train_df[num_features+cat_features+bin_features]),
               'y': train_df[label_col].values})
    # or make prediction based on the trained model by passing the test_data dict
    with torch.no_grad():
        ypred = model.predict({'x':TabularPatientBase(val_df[num_features+cat_features+bin_features]),
                               'y':val_df[label_col].values})
    # print(ypred, y_pred); quit()    
    return ypred['pred'].cpu().numpy().flatten()

def run_transtab(train_df, val_df, num_features, cat_features, bin_features, label_col):
    model = TransTab(
        categorical_columns=cat_features+bin_features,
        numerical_columns=num_features,
        binary_columns=[],
        contrastive_pretrain=False, # mean this model will get contrastive pretraining
        num_class=2,
        mode='binary',
        device='cpu',
        epochs=50,
        )

    # sklearn-like API, train the model
    model.fit({'x': TabularPatientBase(train_df[num_features+cat_features+bin_features]),
               'y': train_df[label_col]})
    # or make prediction based on the trained model by passing the test_data dict
    with torch.no_grad():
        ypred = model.predict(test_data=TabularPatientBase(val_df[num_features+cat_features+bin_features]))
    # print(ypred, y_pred); quit()    
    return ypred

def run_mlp(train_df, val_df, num_features, cat_features, bin_features, label_col):
    # cat_cardinalities = [len(np.unique(train_df[col])) for col in cat_features + bin_features]
    for col in cat_features + bin_features:
        labelencoder = LabelEncoder().fit(train_df[col])
        train_df[col] = labelencoder.transform(train_df[col])
        val_df[col] = labelencoder.transform(val_df[col])

    model = MLP(
        input_dim=len(cat_features)+len(bin_features)+len(num_features),
        output_dim=1,
        num_layer=2, # number of fully connected layers
        mode='binary',
        device='cpu',
        epochs=50,
        )

    # sklearn-like API, train the model
    # print(train_df[num_features+cat_features+bin_features].shape, sum(cat_cardinalities))
    model.fit({'x': train_df[num_features+cat_features+bin_features],'y': train_df[label_col].values})    # or make prediction based on the trained model by passing the test_data dict
    with torch.no_grad():
        ypred = model.predict({'x': TabularPatientBase(val_df[num_features+cat_features+bin_features]), 'y': val_df[label_col].values})
    # print(ypred); quit()    
    return ypred['pred'].cpu().numpy().flatten()

def run_all_baselines(train_df, val_df, num_features, cat_features, bin_features, label_col):
    output_csv = []
    # load the train test split
    y_pred = run_xgboost(train_df, val_df, num_features, cat_features, bin_features, label_col)
    # print(y_pred.shape)
    output_csv.append(['xgboost', roc_auc_score(val_df[label_col], y_pred), average_precision_score(val_df[label_col], y_pred)])
    # print(output_csv)

    # =========================== run_fttransformer pytrial baseline ===========================
    y_pred = run_fttransformer(train_df, val_df, num_features, cat_features, bin_features, label_col)
    output_csv.append(['fttransformer', roc_auc_score(val_df[label_col], y_pred), average_precision_score(val_df[label_col], y_pred)])

    # =========================== run_transtab pytrial baseline ===========================
    y_pred = run_transtab(train_df, val_df, num_features, cat_features, bin_features, label_col)
    output_csv.append(['transtab', roc_auc_score(val_df[label_col], y_pred), average_precision_score(val_df[label_col], y_pred)])

    # =========================== run_mlp pytrial baseline ===========================
    y_pred = run_mlp(train_df, val_df, num_features, cat_features, bin_features, label_col)
    output_csv.append(['mlp', roc_auc_score(val_df[label_col], y_pred), average_precision_score(val_df[label_col], y_pred)])

    output_df = pd.DataFrame(output_csv, columns=['Model', 'ROCAUC', 'PRAUC'])
    return output_df
    
def run_trial_baselines(trial_path='../data', trial_list=None):
    if trial_list is None:
        trial_list = [
        'breast_cancer_NCT00041119',
        'breast_cancer_NCT00174655', 
        'breast_cancer_NCT00312208', 
        'colorectal_cancer_NCT00079274', 
        'lung_cancer_NCT00003299', 
        'lung_cancer_NCT00694382', 
        'lung_cancer_NCT03041311']
    
    all_output_dfs = []
    for trialname in trial_list:
            
        # load the data
        data = distilltab.dataset.load_trial_data(trial_path=os.path.join(trial_path, trialname))
        num_features = data['num_features']
        cat_features = data['cat_features']
        bin_features = data['bin_features']

        for feature in cat_features + bin_features:
            data['df_train'][feature] = data['df_train'][feature].astype(str)
            data['df_test'][feature] = data['df_test'][feature].astype(str)
        for feature in num_features:
            data['df_train'][feature] = data['df_train'][feature].astype(float)
            data['df_test'][feature] = data['df_test'][feature].astype(float)

        # =========================== Run XGBoost ===========================
        results = run_all_baselines(data['df_train'], data['df_test'], num_features, cat_features, bin_features, label_col='target_label')
        results['trialname'] = trialname
        all_output_dfs.append(results)

        # break
    output_df = pd.concat(all_output_dfs)
    return output_df



if __name__ == '__main__':
    # print("Trialname, Model, ROCAUC, PRAUC")
    output_df = run_trial_baselines(trial_path='../data', trial_list=None)
    print(output_df)
    # run_ehr_baselines()