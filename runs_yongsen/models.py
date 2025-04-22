import torch
from torch import nn
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, TFRobertaModel


class SentenceBERTclf(nn.Module):
    def __init__(self):
        super(SentenceBERTclf, self).__init__()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        # for param in self.model.parameters():
        #   param.requires_grad = False
        self.classifier = nn.Sequential(nn.Linear(384, 256),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(256, 1))

    def forward(self, input):
        out = self.model.encode(input)
        out = torch.tensor(out).cuda()
        out = self.classifier(out)
        return out


class BERTclf(nn.Module):
    def __init__(self, num_labels=1, dropout=0.5, bert_model='roberta-base'):
        super(BERTclf, self).__init__()
        self.model = AutoModel.from_pretrained(
            bert_model, add_pooling_layer=True)
        # print(self.model)
        # self.dropout = nn.Dropout(dropout)
        # self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)
        self.classifier = nn.Sequential(nn.Linear(self.model.config.hidden_size, 256),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(256, num_labels))

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
        # dropped = self.dropout(pooled_output)
        # logits = self.classifier(dropped)
        logits = self.classifier(pooled_output)
        return logits
