import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
os.environ["WANDB_DISABLED"] = "true"

# import pdb
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
import numpy as np
import datasets
import torch
from transformers import Trainer, TrainingArguments, AutoConfig
import json

from betashap.ShapEngine import ShapEngine
from distilltab.bert import BertTabClassifier, BertTabTokenizer
from distilltab.dataset import load_augmented_data, load_data
from run_trial_patient_few_shot import get_all_data, load_trial

trial_list = [
    'breast_cancer_NCT00041119',
    'breast_cancer_NCT00174655',
    'breast_cancer_NCT00312208',
    'colorectal_cancer_NCT00079274',
    'lung_cancer_NCT00003299',
    'lung_cancer_NCT00694382',
    'lung_cancer_NCT03041311',
]


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
    round = 0
    mode = 'validate' # in ['get_importance', 'validate', 'train]
    preds_path = '/srv/local/data/chufan2/DistillTab/preds/round_0_pred.csv'
    pmc_path='/srv/local/data/PMC-Patients/datasets/PMC-Patients.json'
    shap_value_path = '/srv/local/data/chufan2/DistillTab/preds/shap_values_round_0.npy'
    percentage_to_add = 2.0
    trial_path='./data'
    output_csv = []

    model = BertTabClassifier.from_pretrained('./checkpoints/BioBertClassifier_trial_patient_pretrain/checkpoint-2600')
    # model = BertTabClassifier.from_pretrained('./checkpoints/BioBertClassifier_trial_patient_pretrain_round_1/')
    # model = BertTabClassifier.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
    tokenizer = BertTabTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2", cache_dir='./share_data/biobert-base-cased-v1.2')
    def tokenize_function(examples):
        return tokenizer(examples['sentence'], padding=True, truncation=True, max_length=512)


    if mode == 'get_importance':
        model.to('cuda')

        train_df, val_df, pmc_df = get_all_data(trial_list, trial_path=trial_path, pmc_path=pmc_path)
        # test_df = test_df.iloc[:3000]
        test_df = pd.concat([train_df, pmc_df])
        print('test_df.shape', test_df)
        test_datasets = datasets.DatasetDict()
        test_datasets['test'] = datasets.Dataset.from_pandas(test_df)
        tokenized_test_datasets = test_datasets.map(tokenize_function, batched=True, batch_size=2048*2)

        test_loader = torch.utils.data.DataLoader(tokenized_test_datasets['test'],
            batch_size=512, 
            shuffle=False,
            num_workers=8,
            pin_memory=True, 
            drop_last=False,
            collate_fn=collate_function,
            )

        model.eval()
        y_test_pred = []
        y_test_embeds = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Predicting on test set'):
                batch = {k: v.to('cuda:0') for k, v in batch.items() if v is not None}
                outputs = model.forward(**batch)
                embeddings = get_embeddings(model, **batch).detach().cpu().numpy()
                y_test_embeds.append(embeddings)
                y_test_pred.extend(outputs.logits.sigmoid().squeeze().cpu().tolist())

        all_pred = pd.DataFrame({'pred':y_test_pred, 'pid':test_df['pid'].values.tolist()})

        # run shap
        shap_engine=ShapEngine(X=embeddings[len(train_df):], 
                            y=y_test_pred[len(train_df):],
                            X_val=embeddings[len(train_df):], 
                            y_val=train_df['label'].values,
                            problem='classification', model_family='logistic',
                            metric='auc', GR_threshold=1.05, max_iters=200, use_parallel_knn=True)
        shap_engine.run(weights_list=None, loo_run=False, knn_run=True)

        np.save('preds/shap_values_round_{}.npy'.format(round), shap_engine.results['KNN'])
        all_pred.to_csv('preds/round_{}_pred.csv'.format(round), index=False)
        np.save('preds/round_{}_pred'.format(round), np.concatenate(y_test_embeds))

    elif mode=='validate': # validate is true
        model.to('cuda')

        # train_df = []
        # val_df = []
        for trialname in trial_list:
            train_df, val_df = load_trial(trialname=trialname, trial_path=trial_path)
        
            raw_datasets = datasets.DatasetDict()
            raw_datasets['train'] = datasets.Dataset.from_pandas(train_df)
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

            model.eval()
            y_val_pred = []
            
            for batch in tqdm(val_loader, desc='Predicting on val set'):
                batch = {k: v.to('cuda:0') for k, v in batch.items() if v is not None}
                with torch.no_grad():
                    outputs = model(**batch)
                y_val_pred.extend(outputs.logits.sigmoid().squeeze().cpu().tolist())

            # # aggreagete the prediction for the same uid
            y_pred = pd.DataFrame({'pred':np.array(y_val_pred), 'pid':val_df['pid'].values})
            y_pred = y_pred.groupby('pid').agg({'pred':'mean'}).reset_index()
            y_true = val_df[['pid','label']].drop_duplicates().sort_values('pid').reset_index(drop=True)
            auc = roc_auc_score(y_true=y_true['label'].values, y_score=y_pred['pred'].values)
            prauc = average_precision_score(y_true=y_true['label'].values, y_score=y_pred['pred'].values)

            print('{} Test ROCAUC: {} PRAUC: {} '.format(trialname, auc, prauc))
            output_csv.append('{} Test ROCAUC: {} PRAUC: {} '.format(trialname, auc, prauc))

        print(output_csv)

    elif mode == 'train':
        # train_df = []
        # val_df = []
        # for trialname in trial_list:
        #     input_dir = os.path.join(trial_path, trialname)
        #     data = load_data(input_dir)
        #     sentence = data['sentence']
        #     label = data['df']['target_label']
        #     split_idxs = data['split_idxs']
        #     train_df_org = pd.DataFrame({'sentence': sentence[split_idxs['train']], 'label': label[split_idxs['train']]})
        #     train_df_org = train_df_org.reset_index().rename(columns={'index': 'pid'})
        #     train_df_org['pid'] = train_df_org['pid'].apply(lambda pid: f'{trialname}-{pid}')

        #     val_df_org = pd.DataFrame({'sentence': sentence[split_idxs['test']], 'label': label[split_idxs['test']]})
        #     val_df_org = val_df_org.reset_index().rename(columns={'index': 'pid'})
        #     val_df_org['pid'] = val_df_org['pid'].apply(lambda pid: f'{trialname}-{pid}')

        #     train_df.append(train_df_org)
        #     val_df.append(val_df_org)

        # for trialname in trial_list:
        #     input_dir = os.path.join(trial_path, f'{trialname}_paraphrase')
        #     data_aug = load_augmented_data(input_dir)
        #     data_aug['df_train']['pid'] = data_aug['df_train']['pid'].apply(lambda pid: f'{trialname}-{pid}')
        #     data_aug['df_test']['pid'] = data_aug['df_test']['pid'].apply(lambda pid: f'{trialname}-{pid}')

        #     train_df.append(data_aug['df_train'])
        #     val_df.append(data_aug['df_test'])

        PMC_Patients_json = json.load(open(pmc_path, 'r'))
        all_preds = pd.read_csv(preds_path)
        shap_value = np.load(shap_value_path)
        num_train = len(all_preds) - len(PMC_Patients_json)
        pmc_preds = all_preds.iloc[num_train:]

        train_df, val_df, pmc_df = get_all_data(trial_list, trial_path=trial_path, pmc_path=pmc_path)

        # print(all_preds.shape, pmc_preds.shape, shap_value.shape, pmc_preds.columns)
        pmc_preds['shap_value'] = shap_value
        pmc_preds['sentence'] = [PMC_Patients_json[i]['patient'] for i in range(len(PMC_Patients_json))]
        # pmc_preds['label'] = (pmc_preds['pred'] > .5).astype(int)
        pmc_preds['label'] = pmc_preds['pred']
            
        pos_label_percentage = pd.concat(train_df)['label'].value_counts(normalize=True)[1]
        cutoff = pmc_preds['shap_value'].quantile(0.5)
        pmc_data = pmc_preds[(pmc_preds['shap_value'] > cutoff)]
        pmc_data['label'] = (pmc_data['pred'] > pos_label_percentage).astype(float)

        # best_pos = pmc_preds[(pmc_preds['pred'] > .5)].sort_values('shap_value', ascending=False)
        # best_neg = pmc_preds[(pmc_preds['pred'] <= .5)].sort_values('shap_value', ascending=False)
        # # get same percentage of positive and negative samples
        # pos_label_percentage = pd.concat(train_df)['label'].value_counts(normalize=True)[1]
        # print('pos_label_percentage', pos_label_percentage)
        # pmc_data = pd.concat([
        #     best_pos.iloc[:int(num_train * percentage_to_add * pos_label_percentage)],
        #     best_neg.iloc[:int(num_train * percentage_to_add * (1-pos_label_percentage))],
        # ])
        pmc_data = pmc_data[['sentence', 'pid', 'label']].reset_index(drop=True)


        # create dataset
        def tokenize_function(examples):
            return tokenizer(examples['sentence'], padding=True, truncation=True, max_length=512)

        raw_datasets = datasets.DatasetDict()
        raw_datasets['train'] = datasets.Dataset.from_pandas(train_df)
        raw_datasets['val'] = datasets.Dataset.from_pandas(val_df)
        raw_datasets['distant'] = datasets.Dataset.from_pandas(pd.concat([train_df, pmc_data]).reset_index(drop=True))
        tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, batch_size=2048*2)

        # # enable full model param fine-tuning     
        # for param in model.bert.embeddings.parameters():
        #     param.requires_grad = True
        # for param in model.bert.encoder.layer[:6].parameters():
        #     param.requires_grad = True

        # init the trainer and train the model
        training_args = TrainingArguments(
            output_dir='./checkpoints/BioBertClassifier_trial_patient_pretrain_round_{}'.format(round+1),          # output directory
            num_train_epochs=2,              # total # of training epochs
            per_device_train_batch_size=64,  # batch size per device during training
            per_device_eval_batch_size=128,   # batch size for evaluation
            warmup_ratio=0.06,                # number of warmup steps for learning rate scheduler
            weight_decay=1e-6,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            logging_steps=10,
            save_steps=64*5,
            eval_steps=64*5,
            evaluation_strategy='steps', 
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            learning_rate=2e-5,
            fp16=False,
            save_total_limit=3,
        )
        trainer = Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=tokenized_datasets['distant'],         # training dataset
            eval_dataset=tokenized_datasets['val'],            # evaluation dataset
            data_collator=collate_function,
            compute_metrics=lambda p: {'roc_auc': roc_auc_score(p.label_ids, p.predictions)}
        )
        trainer.train()

        training_args = TrainingArguments(
            output_dir='./checkpoints/BioBertClassifier_trial_patient_pretrain_round_{}'.format(round+1),          # output directory
            num_train_epochs=1,              # total # of training epochs
            per_device_train_batch_size=64,  # batch size per device during training
            per_device_eval_batch_size=128,   # batch size for evaluation
            warmup_ratio=0.06,                # number of warmup steps for learning rate scheduler
            weight_decay=1e-6,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            logging_steps=10,
            save_steps=64*5,
            eval_steps=64*5,
            evaluation_strategy='steps', 
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            learning_rate=1e-5,
            fp16=False,
            save_total_limit=3,
        )
        trainer = Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=tokenized_datasets['train'],         # training dataset
            eval_dataset=tokenized_datasets['val'],            # evaluation dataset
            data_collator=collate_function,
            compute_metrics=lambda p: {'roc_auc': roc_auc_score(p.label_ids, p.predictions)}
        )
        trainer.train()

        trainer.save_model('./checkpoints/BioBertClassifier_trial_patient_pretrain_round_{}'.format(round+1))
        print('done')
# no training
# ['breast_cancer_NCT00041119 Test ROCAUC: 0.6047405007055872 PRAUC: 0.10895340096058369 ', 
# 'breast_cancer_NCT00174655 Test ROCAUC: 0.8038461538461539 PRAUC: 0.09462937278471259 ', 
# 'breast_cancer_NCT00312208 Test ROCAUC: 0.7605337078651685 PRAUC: 0.47542892276692206 ', 
# 'colorectal_cancer_NCT00079274 Test ROCAUC: 0.6822772197236959 PRAUC: 0.25726733267340146 ', 
# 'lung_cancer_NCT00003299 Test ROCAUC: 0.7567567567567568 PRAUC: 0.9792545396296068 ', 
# 'lung_cancer_NCT00694382 Test ROCAUC: 0.6965113538147246 PRAUC: 0.6826236155250802 ', 
# 'lung_cancer_NCT03041311 Test ROCAUC: 0.8571428571428571 PRAUC: 0.9367346938775509 ']

# epochs = 10
# ['breast_cancer_NCT00041119 Test ROCAUC: 0.6231380337636545 PRAUC: 0.10559955885465311 ', 
# 'breast_cancer_NCT00174655 Test ROCAUC: 0.7897435897435897 PRAUC: 0.06926016926016926 ', 
# 'breast_cancer_NCT00312208 Test ROCAUC: 0.7583099250936329 PRAUC: 0.4749427292895433 ', 
# 'colorectal_cancer_NCT00079274 Test ROCAUC: 0.6821425686047452 PRAUC: 0.25610416891918075 ', 
# 'lung_cancer_NCT00003299 Test ROCAUC: 0.7992277992277992 PRAUC: 0.9852974925051539 ', 
# 'lung_cancer_NCT00694382 Test ROCAUC: 0.6998507110866661 PRAUC: 0.6845132226218208 ', 
# 'lung_cancer_NCT03041311 Test ROCAUC: 0.8571428571428571 PRAUC: 0.9240362811791383 ']

# epochs = 3
# ['breast_cancer_NCT00041119 Test ROCAUC: 0.6198191606125543 PRAUC: 0.10453512861908626 ', 
# 'breast_cancer_NCT00174655 Test ROCAUC: 0.7999999999999999 PRAUC: 0.06024895309423249 ', 
# 'breast_cancer_NCT00312208 Test ROCAUC: 0.758368445692884 PRAUC: 0.474050311392424 ', 
# 'colorectal_cancer_NCT00079274 Test ROCAUC: 0.6809576387579781 PRAUC: 0.25129973161817165 ', 
# 'lung_cancer_NCT00003299 Test ROCAUC: 0.7966537966537967 PRAUC: 0.9851428568087273 ', 
# 'lung_cancer_NCT00694382 Test ROCAUC: 0.7130509939498703 PRAUC: 0.7067953596963268 ', 
# 'lung_cancer_NCT03041311 Test ROCAUC: 0.8214285714285714 PRAUC: 0.9129251700680272 ']

# training from scratch
# 'breast_cancer_NCT00041119 Test ROCAUC: 0.6264830397742122 PRAUC: 0.1003460020358104 ', 
# 'breast_cancer_NCT00174655 Test ROCAUC: 0.7064102564102563 PRAUC: 0.04008792750026972 ', 
# 'breast_cancer_NCT00312208 Test ROCAUC: 0.7478347378277153 PRAUC: 0.48312358467199656 ', 
# 'colorectal_cancer_NCT00079274 Test ROCAUC: 0.7026364689090566 PRAUC: 0.26169968369369695 ', 
# 'lung_cancer_NCT00003299 Test ROCAUC: 0.6113256113256114 PRAUC: 0.9630998812709282 ', 
# 'lung_cancer_NCT00694382 Test ROCAUC: 0.7025221969042195 PRAUC: 0.6742879170744636 ', 
# 'lung_cancer_NCT03041311 Test ROCAUC: 0.8214285714285714 PRAUC: 0.9129251700680272

# no training, reran
# ['breast_cancer_NCT00041119 Test ROCAUC: 0.6064914022892385 PRAUC: 0.10308649866650918 ', 
# 'breast_cancer_NCT00174655 Test ROCAUC: 0.8871794871794871 PRAUC: 0.09798245614035088 ', 
# 'breast_cancer_NCT00312208 Test ROCAUC: 0.7487710674157303 PRAUC: 0.4756274141538407 ', 
# 'colorectal_cancer_NCT00079274 Test ROCAUC: 0.682815824199499 PRAUC: 0.25151806461780557 ', 
# 'lung_cancer_NCT00003299 Test ROCAUC: 0.7760617760617761 PRAUC: 0.9821725967873757 ', 
# 'lung_cancer_NCT00694382 Test ROCAUC: 0.6833503575076609 PRAUC: 0.6792506685695604 ', 
# 'lung_cancer_NCT03041311 Test ROCAUC: 0.8571428571428572 PRAUC: 0.9325396825396823 ']


# with PMC, 1 epochs
# ['breast_cancer_NCT00041119 Test ROCAUC: 0.6184079862018503 PRAUC: 0.1056807706509125 ', 
# 'breast_cancer_NCT00174655 Test ROCAUC: 0.8948717948717949 PRAUC: 0.14195329294508502 ', 
# 'breast_cancer_NCT00312208 Test ROCAUC: 0.7154728464419475 PRAUC: 0.4435688633992354 ', 
# 'colorectal_cancer_NCT00079274 Test ROCAUC: 0.6942342390865268 PRAUC: 0.24137154701377192 ', '
# lung_cancer_NCT00003299 Test ROCAUC: 0.767052767052767 PRAUC: 0.9822071083570476 ', 
# 'lung_cancer_NCT00694382 Test ROCAUC: 0.6869254341164452 PRAUC: 0.6775365689317949 ', 
# 'lung_cancer_NCT03041311 Test ROCAUC: 0.8214285714285714 PRAUC: 0.9129251700680272 ']

# ALL COMBINED, round 1
# Test ROCAUC: 0.8315880533899647 PRAUC: 0.6548092271825946
# ALL COMBINED, round 0
# Test ROCAUC: 0.8424404578943937 PRAUC: 0.6811131570646921