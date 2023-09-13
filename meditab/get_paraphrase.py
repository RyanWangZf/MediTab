from tqdm import tqdm
import pickle
import argparse
import dataset
from chatgpt_api import chatGPTAPI

def paraphrase(api, text):
    api.reset_message()
    return api.get_response_oneround("Paraphrase the following patient: " + text)

def paraphrase_ensemble(api, text):
    api.reset_message()
    return api.get_response_oneround("Paraphrase the following patient 5 different ways: " + text)

def paraphrase_hint(api, text):
    # api.reset_message()
    return api.get_response_oneround("Describe the following clinical trial 5 different ways: " + text)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-split', required=True)
    args = parser.parse_args()
    assert args.split in ['1', '2', '3']           

    # text = 'The race is White. The treatment is paclitaxel. The tumor laterality is left. The cancer histologic grade is Intermediate. The biopsy type is core needle. estrogen receptor positive . progesterone receptor positive . The number of positive axillary nodes is 0. The tumor size is 2.0'
    # output = 'The patient belongs to the White race and has been administered with paclitaxel treatment for their cancer. The tumor is located on the left side of the body and has an intermediate histologic grade. The biopsy conducted was of a core needle type and the patient is estrogen and progesterone receptor positive. However, there are no positive axillary nodes detected and the tumor size is measured at 2.0.'
    # text = 'race White; treatment paclitaxel; tumor laterality left; cancer histologic grade Intermediate; biopsy type core needle; estrogen receptor positive ; progesterone receptor positive ; number of positive axillary nodes 0; tumor size 2.0'
    # output = 'The patient is Caucasian and is undergoing treatment with paclitaxel. The tumor is located on the left side of the body and has an intermediate cancer histologic grade. The biopsy performed was a core needle biopsy and it showed that the tumor is estrogen and progesterone receptor positive. There are no positive axillary nodes present and the size of the tumor is 2.0.'
    # output = paraphrase(api=api, text=text)
    api = chatGPTAPI(system_message="You are a helpful assistant.", keypath='./openai_key.txt', orgid=None)

    # data = distilltab.dataset.load_data()
    data = dataset.load_TOP_data(return_hint=True, drop_ec=True)
    sentences = data['sentence']

    n = len(sentences)
    # print(len(sentences))
    if args.split == '1':
        sentences = sentences.values[:int(n/3)]
    elif args.split == '2':
        sentences = sentences.values[int(n/3):int(n*2/3)]
    else:
        sentences = sentences.values[int(n*2/3):]

    print(sentences.shape)

    # paraphrased_sentences = []
    # for i in tqdm(range(len(sentences))):
    #     sent = sentences.values[i]
    #     new_sent = paraphrase(api, sent)
    #     paraphrased_sentences.append(new_sent)
    # pickle.dump(paraphrased_sentences, open('paraphrased_sentences.pkl', 'wb'))

    paraphrased_sentences_ensemble = []
    paraphrased_sentences_ensemble = pickle.load(open('HINT_paraphrased_sentences_ensemble_{}.pkl'.format(args.split), 'rb'))
    start_i = len(paraphrased_sentences_ensemble)

    for i in tqdm(range(start_i, len(sentences))):
        sent = sentences[i]
        new_sent = paraphrase_hint(api, sent)
        print(i, new_sent)
        # write to csv file
        paraphrased_sentences_ensemble.append(new_sent)

        if i%10==0:
            pickle.dump(paraphrased_sentences_ensemble, open('HINT_paraphrased_sentences_ensemble_{}.pkl'.format(args.split), 'wb'))

    pickle.dump(paraphrased_sentences_ensemble, open('HINT_paraphrased_sentences_ensemble_{}.pkl'.format(args.split), 'wb'))