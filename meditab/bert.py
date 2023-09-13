import pdb
import re
from typing import List, Optional, Tuple, Union

import torch
from torch.nn import BCEWithLogitsLoss
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

# distilltabtokenizer
# distilltabclassifier

class BertTabClassifier(BertForSequenceClassification):
    def __init__(self, config):
        config.num_labels = 1
        # config.problem_type = "binary_classification"
        config.problem_type = "single_label_classification"
        super().__init__(config)
        print("freeze the first six layers of BioBERT for fine-tuning")
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

        # debug
        for param in self.bert.encoder.layer[:6].parameters():
            param.requires_grad = False

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            # align the shape of logits and labels
            # logits: (batch_size, 1)
            # labels: (batch_size, )
            logits = logits.view(-1)
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class BertTabClassifierWithMLM(BertTabClassifier):
    '''Classification loss + MLM loss during the training.

    TODO: add the MLM loss
    '''
    def __init__(self, config):
        super().__init__(config)


class BertTabTokenizer(BertTokenizer):
    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        do_basic_tokenize=True,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs
        ):
        super().__init__(vocab_file=vocab_file,
                         do_lower_case=do_lower_case,
                         do_basic_tokenize=do_basic_tokenize,
                         never_split=never_split,
                         unk_token=unk_token,
                         sep_token=sep_token,
                         pad_token=pad_token,
                         cls_token=cls_token,
                         mask_token=mask_token,
                         tokenize_chinese_chars=tokenize_chinese_chars,
                         strip_accents=strip_accents,
                         **kwargs)
        
        # add a special number token
        self.add_tokens(['[NUM]','[/NUM]'])

        # replace the tokenizer's processor and post-processor
        # to have better handling of numerical values
        # that means we also need to modify the embedding layer of BioBERT

    def _tokenize(self, text):
        # add a special token [NUM] ahead of numerical values
        # add a special token [/NUM] after numerical values
        # if there is + or - mark ahead the number, add [NUM] ahead of it
        # consider the case of multiple comma in the number
        # using regular expression
        pattern = r'\b([+-]?(?:\d{1,3},)*\d{1,3}(?:\.\d+)?)'
        replacement = r'[NUM]\1[/NUM]'
        new_text = re.sub(pattern, replacement, text)
        return super()._tokenize(new_text)