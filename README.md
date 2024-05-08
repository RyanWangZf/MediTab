# MediTab

We are published in IJCAI 2024!

Publicly available data can be found in the github releases. You can extract it into the data folder

# TODO:
- [x] load BioBERT and fine-tune it on the raw sentence dataset
- [x] load GPT-3 API and generate diverse paraphrases of the raw sentences as augmentations
- [x] enhance numerical values by adapting the tokenizer and embedding layer of BioBERT (dmis-lab/biobert-base-cased-v1.2)
- [ ] MLM of BioBERT on the augmented data
- [ ] fact checker dataset building with GPT3 API
- [ ] fine-tune BioBERT on the augmented data with fact checker filtering
- [ ] explore extend the raw sentences with new knowledge background texts, e.g., considering the input drug, extend the descriptions of them.
- [ ] extend to trial outcome prediction, three datasets: phase I & II & III.
- [ ] consider transfer learning across databases: 
  - [x] EHR (40K+ patients) -> clinical trial patient data (~1k per dataset); 
  - [ ] clinicaltrials.gov (400K+ trials) -> trial outcome prediction (~5K per dataset)
