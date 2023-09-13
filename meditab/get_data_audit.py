import numpy as np
import os
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from tqdm import tqdm
import editdistance

from dataset import load_HINT_augmented_data, load_trial_augmented_data, split_paraphrase
from chatgpt_api import chatGPTAPI


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PROMPT_CAT_NUM = "What is the {}? \n{}"
PROMPT_BIN = "Is {} present? (a) yes (b) no \n{}"

def get_qa_model(model_name = "allenai/unifiedqa-v2-t5-3b-1363200", cache_dir='/data/chufan2/huggingface/'):
     # you can specify the model size here
    tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir).to(DEVICE)
    model.eval()
    return model, tokenizer

def run_qa_model(model, tokenizer, prompts_list):
    with torch.no_grad():
        input_ids = tokenizer(prompts_list, return_tensors="pt", max_length=1024, padding=True).to(DEVICE)
        res = model.generate(input_ids['input_ids'], attention_mask=input_ids['attention_mask'], return_dict_in_generate=True, output_scores=True)
        return tokenizer.batch_decode(res['sequences'], skip_special_tokens=True)
        
def normalized_ed(retrieved, orig):
    retrieved = retrieved.lower()
    orig = orig.lower()
    # print(retrieved, orig)
    if orig in retrieved:
        return 1
    else:
        return 1-editdistance.eval(retrieved,orig) / max(len(retrieved), len(orig))

def get_feature_eds(paraphrase: str, patient_row: pd.Series, num_features: list, cat_features: list, bin_features: list, model, tokenizer):
    # paraphrase = dataset.split_paraphrase(paraphrase)[0]

    all_features = num_features + cat_features + bin_features
    prompts_list = [PROMPT_CAT_NUM.format(feature, paraphrase) for feature in num_features + cat_features] + \
        [PROMPT_BIN.format(feature, paraphrase) for feature in bin_features]
    retrieved_list = run_qa_model(model, tokenizer, prompts_list)

    bad_features_list = []
    ed_list = []

    for feat_i, feature in enumerate(all_features):
        edit_dist = normalized_ed(retrieved_list[feat_i], patient_row[feature])
        if edit_dist < 0.1:
            bad_features_list.append(feature)
        ed_list.append(edit_dist)

    return retrieved_list, ed_list, bad_features_list

if __name__=='__main__':
    trial_list = [
        'top',
        # 'breast_cancer_NCT00041119',
        # 'breast_cancer_NCT00174655',
        # 'breast_cancer_NCT00312208',
        # 'colorectal_cancer_NCT00079274',
        # 'lung_cancer_NCT00003299',
        # 'lung_cancer_NCT00694382',
        # 'lung_cancer_NCT03041311',
    ]
    # top_batches = [
    #     'top',
    # ]

    dev_mode = False
    save_interval = 100

    api = chatGPTAPI(system_message="You are a helpful assistant.", keypath='../openai_key.txt', orgid=None)
    # device = torch.device('cpu')
    model, tokenizer = get_qa_model()
    print("device", DEVICE)
    
    for trialname in trial_list:

        if trialname == 'top':
            aug_data = load_HINT_augmented_data(hint_path='/srv/local/data/chufan2/clinical-trial-outcome-prediction/data/',
                                                trial_path='/srv/local/data/chufan2/DistillTab/data/trial_outcome_pred_data.csv',
                                                paraphrased_path='/srv/local/data/chufan2/DistillTab/data/HINT_paraphrased_sentences_ensemble.csv')
            id_col = 'nct_id'   
            base_prompt = 'Add the following info "{}" to the trial description: "{}". Include the numerical values. Rephrase the description in detail 5 times.'

        else:
            # org_data_ = dataset.load_trial_data(os.path.join('../data', trialname))
            aug_data = load_trial_augmented_data(trial_path=os.path.join('../data', trialname),
                                                 paraphrased_path=os.path.join('../data', '{}_paraphrase'.format(trialname)))
            id_col = 'pid'
            base_prompt = 'Add the following info "{}" to the patient description: "{}". Include the numerical values. Rephrase the description in detail 5 times.'


        org_data = aug_data['df']
        all_features = aug_data['num_features'] + aug_data['cat_features'] + aug_data['bin_features']
        # org_data = org_data.sample(n=25, random_state=0, replace=False)
        for feature in aug_data['num_features']:
            org_data[feature] = org_data[feature].apply(lambda x: "{:.2f}".format(x))
        for feature in aug_data['cat_features'] + aug_data['bin_features']:
            org_data[feature] = org_data[feature].astype(str).str.lower()
        # qa_data = pd.read_csv('../qa_output/data_audit_{}.csv'.format(trialname), sep='|')
        print(org_data.columns)

        columns = ['pid', 'i',]
        columns.extend(['original_sentence'] + ['original_{}'.format(feature) for feature in all_features])
        columns.extend(['paraphrased_sentence']+['paraphrased_{}'.format(feature) for feature in all_features]+['paraphrased_ed'])
        columns.extend(['reparaphrased_sentence'] + ['reparaphrased_{}'.format(feature) for feature in all_features] + ['reparaphrased_ed'])
        output_info = []


        unique_ids = np.unique(org_data[id_col])
        if trialname != 'top':
            unique_ids = sorted(unique_ids, key=lambda x: int(x.split('-')[-1])) # sort by patient number

        all_old_eds = []
        all_new_eds = []

        if dev_mode: unique_ids = unique_ids[:20]
        for id in tqdm(unique_ids):
        # for i in tqdm(range(0, 100, 5)):
            patient_df = org_data[org_data[id_col]==id]
            first_paraphrase = patient_df['paraphrase'].values[0]
            orig_ind = patient_df.index[0]

            output_info_ = [id, patient_df.index[0], org_data.iloc[orig_ind]['sentence']]
            output_info_.extend([org_data.iloc[orig_ind][feature] for feature in all_features])

            retrieved_list, ed_list, bad_features_list = get_feature_eds(paraphrase=first_paraphrase, patient_row=org_data.iloc[orig_ind], 
                                                                         num_features=aug_data['num_features'], cat_features=aug_data['cat_features'], bin_features=aug_data['bin_features'], 
                                                                         model=model, tokenizer=tokenizer)
            output_info_.extend([first_paraphrase] + retrieved_list + [np.mean(ed_list)])

            if dev_mode==False and np.mean(ed_list) < 0.5:
                bad_features_sentence = ' '.join(["{} is {};".format(feature, org_data.iloc[orig_ind][feature]) for feature in bad_features_list])
                prompt = base_prompt.format(bad_features_sentence, first_paraphrase)
                reparaphrased_sentence = api.get_response_oneround(prompt)
                first_reparaphrased = split_paraphrase(reparaphrased_sentence)[0]

                retrieved_list2, ed_list2, bad_features_list2 = get_feature_eds(paraphrase=first_reparaphrased, patient_row=org_data.iloc[orig_ind],
                                                                                num_features=aug_data['num_features'], cat_features=aug_data['cat_features'], bin_features=aug_data['bin_features'],
                                                                                model=model, tokenizer=tokenizer)
                output_info_.extend([first_reparaphrased] + retrieved_list2 + [np.mean(ed_list2)])
                print("Prompt:", prompt)
                print("Result:", reparaphrased_sentence)
                print("Old ED:", np.mean(ed_list), "New ED:", np.mean(ed_list2))

                all_old_eds.append(np.mean(ed_list))
                all_new_eds.append(np.mean(ed_list2))
                # print(retrieved_list2, ed_list2, bad_features_list2)

                # quit()
            else:
                output_info_.append(np.nan)                    
                for feature in all_features:
                    output_info_.append(np.nan)
                output_info_.append(np.nan)

            # print(bad_features, reparaphrased_sentence)
            output_info.append(output_info_)

            # save interval
            if (dev_mode==False) and (save_interval is not None) and (len(output_info) % save_interval == 0):
                out_d = pd.DataFrame(data=output_info, columns=columns)
                out_d.to_csv(path_or_buf="../qa_output/data_audit_{}.csv".format(trialname), index=False, sep='|')

        out_d = pd.DataFrame(data=output_info, columns=columns)
        if dev_mode: 
            print(out_d)
            out_d.to_csv(path_or_buf="tmp.csv", index=False, sep='|')
            quit()
        out_d.to_csv(path_or_buf="../qa_output/data_audit_{}.csv".format(trialname), index=False, sep='|')
        print("Old ED: {} +- {}, New ED: {} +- {}".format(np.mean(all_old_eds), np.std(all_old_eds), np.mean(all_new_eds), np.std(all_new_eds)))
        # break

    # for trialname in top_batches:
    #     aug_data = dataset.load_HINT_augmented()
    #     data = dataset.load_TOP_data(return_hint=True, drop_ec=False)

    #     aug_df = pd.concat([aug_data['df_train'], aug_data['df_test']])

    #     columns = ['nct_id', 'i'] + data['cat_features'] 
    #     output_info = []

    #     batch_num = trialname[-1]
    #     n = len(aug_df)
    #     if batch_num == '1': inds = range(0, int(n * 1/4))
    #     # if batch_num == '1': inds = range(0, 5)
    #     elif batch_num == '2': inds = range(int(n * 1/4), int(n * 2/4))
    #     elif batch_num == '3': inds = range(int(n * 2/4), int(n * 3/4))
    #     elif batch_num == '4': inds = range(int(n * 3/4), n)
    #     else:
    #         raise NotImplementedError

    #     # for i in tqdm(range(len(aug_df))):
    #     for i in tqdm(inds):
    #         nct_id = aug_df.iloc[i]['nct_id']
    #         sentence = aug_df.iloc[i]['sentence']
    #         output_info_ = [nct_id,i]

    #         for feature in data['cat_features']:
    #             out = run_model(prompt_cat_num.format(feature, sentence))
    #             output_info_.append(out)
            
    #         output_info.append(output_info_)

    #     out_d = pd.DataFrame(data=output_info, columns=columns)
    #     out_d.to_csv(path_or_buf="qa_output/{}.csv".format(trialname), index=False)
