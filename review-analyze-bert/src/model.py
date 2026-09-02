import torch
from torch import nn
from transformers import AutoModel

import config


class ReviewAnalyzeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config.PRE_TRAINED_DIR/'bert-base-chinese')
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask,token_type_ids):
        # shape: [batch_size, seq_len]
        output = self.bert(input_ids, attention_mask, token_type_ids)

        last_hidden_state = output.last_hidden_state
        # last_hidden_state.shape: [batch_size, seq_len, hidden_size]

        cls_hidden_state = last_hidden_state[:,0,:]
        # cls_hidden_state.shape: [batch_size, hidden_size]

        output = self.linear(cls_hidden_state).squeeze(-1)
        # output.shape: [batch_size]
        return output
