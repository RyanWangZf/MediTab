import os
import pandas as pd
import numpy as np
import json

# __all__ = ['load_data', 'load_augmented_data', 'load_ehr_data']


def split_paraphrase(paraphrase):
    splits = [_[2:] for _ in paraphrase.split('\n') if _.strip() != '']
    # =========== account for potential errors ===========
    # if paraphrase has random intro e.g.
    # "Sure, here are five possible textual descriptions for this patient based on the provided record:
    # 1. This female patient is post-menopausal ..."
    if len(splits) == 6: 
        return splits[1:]
    
    # =========== account for potential errors ===========
    # if paraphrase just stopped generating
    # 1. This female patient is post-menopausal ..."
    # 2. This female patient is post-menopausal ..."
    # 3. This female patient is"
    if len(splits) < 5 and '5. ' not in paraphrase: 
        return [splits[0]] * 5

    return splits


def load_trial_augmented_data(trial_path='./data/breast_cancer_NCT00041119/', paraphrased_path='./data/'):
    '''
    Load the textual descriptions data augmented by GPT-3.
    '''
    data = load_trial_data(trial_path=trial_path)
    df = data['df']

    filename = [x for x in os.listdir(paraphrased_path) if x.endswith('.csv')][0]
    paraphrased_df = pd.read_csv(os.path.join(paraphrased_path, filename), index_col=0)

    df['paraphrase'] = paraphrased_df['sentence']
    # trialname = os.path.basename(trial_path)
    # df.rename(columns={'target_label': 'label', 'sentence': 'paraphrase'}, inplace=True)
    # df = df.reset_index().rename(columns={'index': 'pid'})
    # df['pid'] = df['pid'].apply(lambda pid: f'{trialname}-{pid}')
    df['paraphrase'] = df['paraphrase'].apply(split_paraphrase)

    # load the split indices
    split_filename = os.path.join(paraphrased_path, 'split_idxs.json')
    with open(split_filename, 'r') as f:
        split_idxs = json.load(f)
        split_idxs = {k: [int(_) for _ in v] for k, v in split_idxs.items()}
    df_train = df.iloc[split_idxs['train']]
    df_test = df.iloc[split_idxs['test']]

    # explode the augmented sentences
    df = df.explode('paraphrase').reset_index(drop=True)
    df_train = df_train.explode('paraphrase').reset_index(drop=True)
    df_test = df_test.explode('paraphrase').reset_index(drop=True)

    return {
        'df': df,
        'df_train': df_train,
        'df_test': df_test,
        'num_features': data['num_features'],
        'bin_features': data['bin_features'],
        'cat_features': data['cat_features'],
    }


def load_trial_data(trial_path='./data/breast_cancer_NCT00041119/'):
    # load the data
    filename = [x for x in os.listdir(trial_path) if x.endswith('.csv')][0]
    df = pd.read_csv(os.path.join(trial_path, filename), index_col=0)
    df.columns = [x.lower() for x in df.columns]

    bin_filename = os.path.join(trial_path, 'binary_feature.txt')
    num_filename = os.path.join(trial_path, 'numerical_feature.txt')
    if os.path.exists(bin_filename):
        with open(bin_filename, 'r') as f:
            bin_features = f.read().splitlines()
        bin_features = [x.lower() for x in bin_features]
    else:
        bin_features = []
    
    if os.path.exists(num_filename):
        with open(num_filename, 'r') as f:
            num_features = f.read().splitlines()
        num_features = [x.lower() for x in num_features]
    else:
        num_features = []
    feat = df.drop(['target_label'], axis=1)
    cat_features = [col.lower() for col in feat.columns if col not in bin_features and col not in num_features]

    cat_feat_str = None
    bin_feat_str = None
    num_feat_str = None

    if len(bin_features) > 0:
        bin_feat = feat[bin_features]
        bin_feat = bin_feat.replace({'Yes': 1, 'No': 0})
        bin_feat_str = bin_feat.apply(lambda x: x.name + ' ') * bin_feat
        bin_feat_str = bin_feat_str.agg('; '.join, axis=1)

    if len(num_features) > 0:
        # round the numerical features to 3 decimal places
        feat[num_features] = feat[num_features].round(3)
        num_feat = feat[num_features].astype(str)
        mask  = (~pd.isna(num_feat)).astype(int)
        num_feat = num_feat.fillna('')
        num_feat = num_feat.apply(lambda x: x.name + ' '+ x) * mask # mask out nan features
        num_feat_str = num_feat.agg('; '.join, axis=1)

    if len(cat_features) > 0:
        cat_feat = feat[cat_features].astype(str)
        mask = (~pd.isna(cat_feat)).astype(int)
        cat_feat = cat_feat.fillna('')
        cat_feat = cat_feat.apply(lambda x: x.name + ' '+ x) * mask # mask out nan features
        cat_feat_str = cat_feat.agg('; '.join, axis=1)
    
    assert (cat_feat_str is not None) or (bin_feat_str is not None) or (num_feat_str is not None), "No feature found!"

    # store the feature string in a dataframe
    df_feat = pd.DataFrame()
    if cat_feat_str is not None:
        df_feat['cat_feat'] = cat_feat_str
    if bin_feat_str is not None:
        df_feat['bin_feat'] = bin_feat_str
    if num_feat_str is not None:
        df_feat['num_feat'] = num_feat_str

    # concatenate all the feature strings
    df['sentence'] = df_feat.apply(lambda x: '; '.join(x.dropna()), axis=1)

    # remove the intermediate ";" which repeats more than one time
    df['sentence'] = df['sentence'].str.replace(r'(; )\1+', r'\1', regex=True)

    # load train test split
    filename = os.path.join(trial_path, 'split_idxs.json')
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            split_idxs = json.load(f)
    else:
        split_idxs = None

    trialname = os.path.basename(trial_path)
    df = df.reset_index().rename(columns={'index': 'pid'})
    df['pid'] = df['pid'].apply(lambda pid: f'{trialname}-{pid}')
    
        
    return {'df': df, 'df_train': df.iloc[split_idxs['train']], 'df_test': df.iloc[split_idxs['test']], 'num_features': num_features, 'bin_features': bin_features, 'cat_features': cat_features}


def load_ehr_data(input_dir='./data/ehr_hf'):
    '''
    Load tabular ehr data and return the naive sentence representation.
    '''
    df_raw = pd.read_csv(os.path.join(input_dir, 'data.csv'))
    text_filename = os.path.join(input_dir, 'data_sentence.csv')

    if os.path.exists(text_filename):
        print("Loading sentence representation from file...")
        text = pd.read_csv(text_filename)
        text = text['sentence']

    else:
        print("No sentence representation found, generating from raw data...")
        df = df_raw.copy()
        mask  = (~pd.isna(df)).astype(int)
        mask.drop(['subject_id', 'mortality_label'], axis=1, inplace=True)
        df.fillna('[]', inplace=True)

        # load icd 2 description
        df_icd_diag = pd.read_csv(os.path.join('./physionet.org/files/mimiciv/2.0/hosp/', 'd_icd_diagnoses.csv.gz'))
        df_icd_diag = df_icd_diag[df_icd_diag['icd_version'] == 10]
        map_icd_diag = dict(zip(df_icd_diag['icd_code'], df_icd_diag['long_title']))
        df_icd_prod = pd.read_csv(os.path.join('./physionet.org/files/mimiciv/2.0/hosp/', 'd_icd_procedures.csv.gz'))
        df_icd_prod = df_icd_prod[df_icd_prod['icd_version'] == 10]
        map_icd_prod = dict(zip(df_icd_prod['icd_code'], df_icd_prod['long_title']))

        # parse list of strings
        df['diagnosis_icd10_codes'] = df['diagnosis_icd10_codes'].apply(lambda x: eval(x))
        # map the icd code to description
        df['diagnosis_icd10_codes'] = df['diagnosis_icd10_codes'].apply(lambda x: [map_icd_diag.get(y, y) for y in x])
        # join the list of strings
        df['diagnosis_icd10_codes'] = df['diagnosis_icd10_codes'].apply(lambda x: ', '.join(x))

        df['medication_names'] = df['medication_names'].apply(lambda x: eval(x))
        df['medication_names'] = df['medication_names'].apply(lambda x: ', '.join(x))

        df['procedure_icd10_codes'] = df['procedure_icd10_codes'].apply(lambda x: eval(x))
        df['procedure_icd10_codes'] = df['procedure_icd10_codes'].apply(lambda x: [map_icd_prod.get(y, y) for y in x])
        df['procedure_icd10_codes'] = df['procedure_icd10_codes'].apply(lambda x: ', '.join(x))

        # rename the columns
        df_text = df.rename(columns={'diagnosis_icd10_codes': 'diagnoses: ', 'medication_names': 'medications: ', 'procedures': 'procedure icd-10 codes: '}).drop(['subject_id', 'mortality_label'], axis=1)

        # transform all column to string
        df_text = df_text.astype(str)
        # lowercase all the text
        df_text = df_text.apply(lambda x: x.str.lower())
        text = df_text.apply(lambda x: x.name.lower() + ' '+ x) * mask.values # mask out nan features

        # concatenate all the feature strings
        text = text.agg('; '.join, axis=1)
        text = text.str.replace(r'(; )\1+', r'\1', regex=True)
        text.name = 'sentence'
        text.to_csv(os.path.join(input_dir, 'data_sentence.csv'), index=False)

    # load train test split
    filename = os.path.join(input_dir, 'split_idxs.json')

    if os.path.exists(filename):
        with open(filename, 'r') as f:
            split_idxs = json.load(f)
    else:
        split_idxs = None

    # str to list
    df_raw.fillna('[]', inplace=True)
    df_raw['diagnosis_icd10_codes'] = df_raw['diagnosis_icd10_codes'].apply(lambda x: eval(x))
    df_raw['medication_names'] = df_raw['medication_names'].apply(lambda x: eval(x))
    df_raw['procedure_icd10_codes'] = df_raw['procedure_icd10_codes'].apply(lambda x: eval(x))

    return {'df': df_raw, 'sentence': text, 'split_idxs': split_idxs}


def load_TOP_data(return_hint=True,
                  hint_path='/srv/local/data/chufan2/clinical-trial-outcome-prediction/data',
                  trial_path='/srv/local/data/chufan2/DistillTab/data/trial_outcome_pred_data.csv',
                  load_ec=False):
    # load HINT data 
    hint_data = []
    for phase in ['I', 'II', 'III']:
        for split in ['train', 'valid', 'test']:
            d = pd.read_csv(os.path.join(hint_path,'phase_{}_{}.csv'.format(phase, split)))
            if split in ['train', 'valid']:
                d['split'] = 'train'
            else:
                d['split'] = 'test'
    #             hint_test_nct_id.update(list(d['nctid']))
            hint_data.append(d)
    hint_data = pd.concat(hint_data)

    d = pd.read_csv(trial_path)
    # replace labels with hint labels
    label_map = {d['nct_id'].iloc[i]: d['label'].iloc[i] for i in range(len(d))} # first populate with initial labels
    label_map.update({hint_data['nctid'].iloc[i]: hint_data['label'].iloc[i] for i in range(len(hint_data))})
    d['label'] = d['nct_id'].apply(lambda x: label_map[x])
    d = d.dropna(subset=['label'])
    d['label'] = d['label'].astype(float)
    
    if return_hint:
        d = d[d['nct_id'].isin(hint_data['nctid'])]
        split_map = {hint_data['nctid'].iloc[i]: hint_data['split'].iloc[i] for i in range(len(hint_data))}
        d['split'] = d['nct_id'].apply(lambda x: split_map[x])
    else: # return trials for pretraining, not in hint
        d = d[~d['nct_id'].isin(hint_data['nctid'])]
        # randomly split into train/test
        np.random.seed(0)
        d['split'] = np.random.choice(['train', 'test'], size=len(d), p=[0.8, 0.2])

    d[['inclusion criteria', 'exclusion criteria']] = d[['inclusion criteria', 'exclusion criteria']].fillna('')
    d['inclusion criteria'] = d['inclusion criteria'].apply(lambda x: " ".join(x.split())) # remove extra spaces
    d['exclusion criteria'] = d['exclusion criteria'].apply(lambda x: " ".join(x.split())) # remove extra spaces
    d['interventions'] = d['interventions'].str.replace(';',',')

    cat_features = ['title', 'study type', 'phase', 'enrollment', 'conditions', 'interventions', 'sponsor', ]

    if load_ec==True:
        text = d[cat_features + ['inclusion criteria', 'exclusion criteria']].astype(str)
    else:
        text = d[cat_features].astype(str)
    mask = (~pd.isna(text[cat_features])).astype(int)
    text = text.fillna('')
    text[cat_features] = text[cat_features].apply(lambda x: x.name + ' '+ x) * mask # mask out nan features
        
    # store the feature string in a dataframe
    d['sentence'] = text.agg('; '.join, axis=1)

    # remove the intermediate ";" which repeats more than one time
    d['sentence']  = d['sentence'].str.replace(r'(. )\1+', r'\1', regex=True)
    
    return {'df': d, 'df_train': d[d['split']=='train'], 'df_test': d[d['split']=='test'], 'cat_features':cat_features, 'num_features':[], 'bin_features':[]}

def load_HINT_augmented_data(hint_path='/srv/local/data/chufan2/clinical-trial-outcome-prediction/data',
                             trial_path='/srv/local/data/chufan2/DistillTab/data/trial_outcome_pred_data.csv',
                             paraphrased_path='/srv/local/data/chufan2/DistillTab/data/HINT_paraphrased_sentences_ensemble.csv'):
    data = load_TOP_data(return_hint=True, hint_path=hint_path, trial_path=trial_path, load_ec=False)
    df = data['df']

    df.rename(columns={'sentence': 'original_sentence'}, inplace=True)
    paraphrased = pd.read_csv(paraphrased_path)
    df['paraphrase'] = paraphrased['paraphrased'].tolist()
    df['paraphrase'] = df['paraphrase'].apply(lambda x: split_paraphrase(x))
    df = df.explode('paraphrase').reset_index(drop=True)
    df['sentence'] = df[['paraphrase', 'inclusion criteria', 'exclusion criteria']].agg(' '.join, axis=1)

    # text_features = ['paraphrase', 'inclusion criteria', 'exclusion criteria',]
    # df[text_features] = df[text_features].astype(str)
    # mask = (~pd.isna(df[text_features])).astype(int)
    # df[text_features] = df[text_features].fillna('')
    # df[text_features] = df[text_features].apply(lambda x: x.name + ' '+ x) * mask # mask out nan features
    # df['paraphrase'] = df[text_features].agg('; '.join, axis=1)

    return {'df': df, 'df_train': df[df['split']=='train'], 'df_test': df[df['split']=='test'], 'cat_features':data['cat_features'], 'num_features':data['num_features'], 'bin_features':data['bin_features']}
