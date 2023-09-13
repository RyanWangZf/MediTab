import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ["WANDB_DISABLED"] = "true"
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sklearn.metrics import roc_auc_score, average_precision_score
import datasets
import torch
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
from transformers import Trainer, TrainingArguments

from distilltab.bert import BertTabClassifier, BertTabTokenizer
from distilltab.dataset import load_TOP_data, load_HINT_augmented_data

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# create the data collator
def collate_function(examples):
    input_ids = [torch.tensor(example['input_ids']) for example in examples]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = [torch.tensor(example['attention_mask']) for example in examples]
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.tensor([example['label'] for example in examples])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

if __name__=='__main__':
    # ==================== Data and Model preparation ====================
    dev_mode = True
    input_dir = '../checkpoints/top_pretrain_no_paraphrase/checkpoint-7000'
    huggingface_cache_dir = "/srv/local/data/chufan2/huggingface/"
    hint_path='/srv/local/data/chufan2/clinical-trial-outcome-prediction/data/'
    trial_path='/srv/local/data/chufan2/DistillTab/data/trial_outcome_pred_data.csv'
    paraphrased_path='/srv/local/data/chufan2/DistillTab/data/HINT_paraphrased_sentences_ensemble.csv'
    num_train_epochs = 10


    if dev_mode==True:
        num_train_epochs = 1

    output_csv = ['Parahrase, Phase, Subset, ROCAUC, PRAUC']
    for use_paraphrase in [False, True]:
        for use_subset in [.25, .5, 1.0]: # float or None
            for phase in ['1','2','3']:
                output_dir = '../checkpoints/top_finetune/'

                # load paraphrased + augmented trial outcome data
                if use_paraphrase==True:
                    data = load_HINT_augmented_data(hint_path=hint_path, trial_path=trial_path, paraphrased_path=paraphrased_path)        
                else:
                    data = load_TOP_data(hint_path=hint_path, trial_path=trial_path, return_hint=True, load_ec=True)
                    
                data['df_train'] = data['df_train'][data['df_train']['phase'].str.contains("Phase "+phase)]
                data['df_test'] = data['df_test'][data['df_test']['phase'].str.contains("Phase "+phase)]
                # combine the data
                print(phase, data['df_train'].shape, data['df_test'].shape) # continue
                train_df = data['df_train']
                val_df = data['df_test']

                # filter the data according to subsets
                if use_subset is not None:
                    if use_subset < 1.0:
                        nct_ids = train_df['nct_id'].unique()
                        np.random.seed(0)
                        ids = np.random.choice(nct_ids, int(len(nct_ids)*use_subset), replace=False)
                        train_df = train_df[train_df['nct_id'].isin(ids)]

                raw_datasets = datasets.DatasetDict()
                raw_datasets['train'] = datasets.Dataset.from_pandas(train_df)
                raw_datasets['val'] = datasets.Dataset.from_pandas(val_df)

                # load the model
                tokenizer = BertTabTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2", cache_dir=huggingface_cache_dir)
                model = BertTabClassifier.from_pretrained("dmis-lab/biobert-base-cased-v1.2", cache_dir=huggingface_cache_dir)
                # model = BertTabClassifier.from_pretrained(input_dir)

                embedding_size = model.get_input_embeddings().weight.shape[0]
                if len(tokenizer) > embedding_size:
                    model.resize_token_embeddings(len(tokenizer))

                def tokenize_function(examples):
                    return tokenizer(examples['sentence'], padding=True, truncation=True, max_length=512)
                tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=['__index_level_0__'])

                # ==================== Model training ====================
                # init the trainer and train the model
                training_args = TrainingArguments(
                    output_dir=output_dir,          # output directory
                    num_train_epochs=num_train_epochs,              # total # of training epochs
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
                    metric_for_best_model='roc_auc',
                    greater_is_better=True,
                    learning_rate=2e-5,
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

                output = trainer.train()
                print('training output', output)
                # trainer.save_model(output_dir)

                # ==================== Model evaluation ====================
                # model = BertTabClassifier.from_pretrained(output_dir)
                val_loader = torch.utils.data.DataLoader(tokenized_datasets['val'], 
                    batch_size=128, 
                    shuffle=False,
                    num_workers=8,
                    pin_memory=True, 
                    drop_last=False,
                    collate_fn=collate_function,
                    )
                
                # evaluate the model
                y_true = []
                y_pred = []
                model.eval()
                model.to('cuda:0')
                for batch in tqdm(val_loader, desc='Predicting'):
                    batch = {k: v.to('cuda:0') for k, v in batch.items()}
                    with torch.no_grad():
                        outputs = model(**batch)
                    logits = outputs.logits
                    y_true.extend(batch['labels'].cpu().tolist())
                    y_pred.extend(logits.sigmoid().cpu().tolist())

                # aggreagete the prediction for the same uid
                y_pred = pd.DataFrame({'pred':np.array(y_pred), 'nct_id':val_df['nct_id'].values})                
                y_pred = y_pred.groupby('nct_id').agg({'pred':'mean'}).reset_index()
                y_true = val_df[['nct_id','label','phase']].drop_duplicates().sort_values('nct_id').reset_index(drop=True)

                y_pred['true'] = y_true['label']
                if dev_mode==True:
                    print(y_pred)
                    quit()    
                y_pred.to_csv(output_dir+'use_paraphrase_{}_phase_{}_subset_{}_pred.csv'.format(use_paraphrase, phase, use_subset), index=False)

                auc = roc_auc_score(y_true=y_true['label'].values, y_score=y_pred['pred'].values)
                prauc = average_precision_score(y_true=y_true['label'].values, y_score=y_pred['pred'].values)

                output_csv.append('{}, {}, {}, {}, {}'.format(use_paraphrase, phase, use_subset, auc, prauc))

    print('\n'.join(output_csv))

# 'Phase, subset, ROCAUC, PRAUC'
# '1, 0.25, 0.6259428571428571, 0.510898185891616 ', 
# '2, 0.25, 0.6873009003683805, 0.4947050736729167 ', 
# '3, 0.25, 0.5855470616340182, 0.20746606808496995 ',
# '1, 0.5, 0.6373079365079366, 0.5365922250384196 ', 
# '2, 0.5, 0.7932635978847149, 0.6974831003792492 ', 
# '3, 0.5, 0.6764612199394807, 0.24022074766281043 ',
# '1, 1.0, 0.6307174603174603, 0.5513512124506206 ', 
# '2, 1.0, 0.8091789020361202, 0.7276170668548181 ', 
# '3, 1.0, 0.7420369485586876, 0.3594137133331343 '