import os
import sys
# import pdb
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
import numpy as np
import datasets
import torch
from transformers import Trainer, TrainingArguments
import json
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
os.environ["WANDB_DISABLED"] = "true"

from distilltab.bert import BertTabClassifier, BertTabTokenizer
import distilltab.dataset

trial_list = [
    'breast_cancer_NCT00041119',
    'breast_cancer_NCT00174655',
    'breast_cancer_NCT00312208',
    'colorectal_cancer_NCT00079274',
    'lung_cancer_NCT00003299',
    'lung_cancer_NCT00694382',
    'lung_cancer_NCT03041311',
]


def get_all_data(trial_list, trial_path='./data', pmc_path='/srv/local/data/PMC-Patients/datasets/PMC-Patients.json'):
    # ==================== load trial data ====================
    train_df = []
    val_df = []
    for trialname in trial_list:
        aug_data = distilltab.dataset.load_trial_augmented_data(trial_path=os.path.join(trial_path, trialname), 
                                                                paraphrased_path=os.path.join(trial_path, '{}_paraphrase'.format(trialname)))
        train_df.append(aug_data['df_train'])
        val_df.append(aug_data['df_test'])

    train_df = pd.concat(train_df).reset_index(drop=True)
    val_df = pd.concat(val_df).reset_index(drop=True)

    # ==================== load PMC_patients_data ====================
    PMC_Patients_json = json.load(open(pmc_path, 'r'))
    pmc_df = pd.DataFrame({'sentence': [PMC_Patients_json[i]['patient'] for i in range(len(PMC_Patients_json))],
                            'pid': ['pmc-{}'.format(i) for i in range(len(PMC_Patients_json))]})

    return train_df, val_df, pmc_df

def collate_function(examples):
    input_ids = [torch.tensor(example['input_ids']) for example in examples]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = [torch.tensor(example['attention_mask']) for example in examples]
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    if 'label' not in examples[0].keys():
        return {'input_ids': input_ids, 'attention_mask': attention_mask}
    else:
        labels = torch.tensor([example['label'] for example in examples], dtype=torch.float)
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

def get_embeddings(bert_tab_model, **kwargs):
    with torch.no_grad():
        bert_tab_model.eval()
        if 'labels' in kwargs.keys():
            del kwargs['labels']
        return bert_tab_model.dropout(bert_tab_model.bert(**kwargs)[1])


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str)
    args = parser.parse_args()
    print(args)
    mode = args.modea
    assert mode in ['train', 'validate']

    pmc_path='/srv/local/data/chufan2/DistillTab/data/PMC-Patients.json'
    # model_path = '/srv/local/data/zifengw2/DistillTab/checkpoints/FMTP_ZeroShot'
    model_path = "dmis-lab/biobert-base-cased-v1.2"
    trial_path='/srv/local/data/chufan2/DistillTab/data/'
    huggingface_cache_dir = "/srv/local/data/chufan2/huggingface/"
    output_dir = './checkpoints/trial_patient_shots={}'


    few_shot_list = [10, 25, 50, 100, 200]


    output_csv = []
    for shots in few_shot_list:
        model = BertTabClassifier.from_pretrained(model_path, cache_dir=huggingface_cache_dir)
        tokenizer = BertTabTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2", cache_dir=huggingface_cache_dir)
        # train_df, val_df, pmc_df = get_all_data(trial_list=trial_list, trial_path=trial_path, pmc_path=pmc_path)
        round = 0

        sampled_data = []
        val_dfs = []

        # sample percentage of data, stratified by trial
        for trialname in trial_list:
            aug_data = distilltab.dataset.load_trial_augmented_data(trial_path=os.path.join(trial_path, trialname), 
                                                                    paraphrased_path=os.path.join(trial_path, '{}_paraphrase'.format(trialname)))
            trial_df = aug_data['df_train']
            val_dfs.append(aug_data['df_test'])

            unique_labels, counts = np.unique(trial_df['label'].values, return_counts=True)

            for label_i in range(len(unique_labels)):
                relevant_pids = trial_df[trial_df['label']==unique_labels[label_i]]['pid'].unique()
                num_ids_to_sample = np.round(counts[label_i] / counts.sum() * shots).astype(int)
                num_ids_to_sample = np.clip(num_ids_to_sample, 5, len(relevant_pids))
                np.random.seed(0)
                pids = np.random.choice(relevant_pids, size=num_ids_to_sample, replace=False).tolist()
                sampled_data.append(trial_df[trial_df['pid'].isin(pids)])
                # print(sampled_data[-1].shape)

            # sampled_data.append(trial_df.sample(random_state=0, replace=False, n=shots))

        raw_datasets = datasets.DatasetDict()
        raw_datasets['train'] = datasets.Dataset.from_pandas(pd.concat(sampled_data).reset_index(drop=True))
        raw_datasets['val'] = datasets.Dataset.from_pandas(pd.concat(val_dfs).reset_index(drop=True))
        
        def tokenize_function(examples):
            return tokenizer(examples['sentence'], padding=True, truncation=True, max_length=512)
        tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, batch_size=2048*2)

        # init the trainer and train the model
        training_args = TrainingArguments(
            output_dir=output_dir.format(shots),          # output directory
            num_train_epochs=2,              # total # of training epochs
            per_device_train_batch_size=64,  # batch size per device during training
            per_device_eval_batch_size=128,   # batch size for evaluation
            warmup_ratio=0.06,                # number of warmup steps for learning rate scheduler
            weight_decay=1e-6,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            evaluation_strategy='steps', 
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            learning_rate=5e-6,
            fp16=False,
            save_total_limit=1,
        )
        trainer = Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=tokenized_datasets['train'],         # training dataset
            eval_dataset=tokenized_datasets['val'],            # evaluation dataset
            data_collator=collate_function,
            compute_metrics=lambda p: {'roc_auc': roc_auc_score(p.label_ids, p.predictions)}
        )
        if mode == 'train':
            trainer.train()
            trainer.save_model(output_dir.format(shots))

        # ======================================== Model evaluation ========================================
        with torch.no_grad():
            model.eval()
            model = BertTabClassifier.from_pretrained(output_dir.format(shots))

            all_preds_dicts = {}
            all_labels = []
            for trialname in tqdm(trial_list):
                aug_data = distilltab.dataset.load_trial_augmented_data(trial_path=os.path.join(trial_path, trialname), 
                                                                        paraphrased_path=os.path.join(trial_path, '{}_paraphrase'.format(trialname)))
                val_df = aug_data['df_test']
                raw_datasets = datasets.DatasetDict()
                raw_datasets['val'] = datasets.Dataset.from_pandas(val_df)
                tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, batch_size=2048)
                
                val_loader = torch.utils.data.DataLoader(tokenized_datasets['val'], 
                    batch_size=512, 
                    shuffle=False,
                    num_workers=8,
                    pin_memory=True, 
                    drop_last=False,
                    collate_fn=collate_function,
                    )
                
                # evaluate the model
                y_val_pred = []
                for batch in tqdm(val_loader, desc='Predicting on val set'):
                    batch = {k: v.to('cuda:0') for k, v in batch.items() if v is not None}
                    outputs = model(**batch)
                        # embeddings = get_embeddings(model, **batch).detach().cpu().numpy()
                    # y_val_embeds.append(embeddings)
                    y_val_pred.extend(outputs.logits.sigmoid().squeeze().cpu().tolist())

                y_pred = pd.DataFrame({'pred':np.array(y_val_pred), 'pid':val_df['pid'].values})
                # # aggreagete the prediction for the same uid
                # y_pred['pred'] = y_pred['pred'].values * val_df['score'].values
                y_pred = y_pred.groupby('pid').agg({'pred':'mean'}).reset_index()
                y_true = val_df[['pid','label']].drop_duplicates().sort_values('pid').reset_index(drop=True)\
                
                all_preds_dicts['FTMP'].append(y_pred['pred'].values)
                all_labels.append(y_true['label'].values)

                # eval baselines
                auc = roc_auc_score(y_true=y_true['label'].values, y_score=y_pred['pred'].values)
                prauc = average_precision_score(y_true=y_true['label'].values, y_score=y_pred['pred'].values)
                output_csv.append([shots, trialname, 'FMTP', auc, prauc])
                print(output_csv[-1])

        
            # final eval ftmp
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)                
            auc = roc_auc_score(y_true=all_labels, y_score=all_preds)
            prauc = average_precision_score(y_true=all_labels, y_score=all_preds)
            output_csv.append([shots, "All Trials", 'FMTP', auc, prauc])
            print(output_csv[-1])


    print(output_csv)
