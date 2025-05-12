import torch.nn as nn
from transformers import AutoModel
from torch.nn import functional as F



class DistilBertClassifier(nn.Module):
    def __init__(self, n_classes):
        super(DistilBertClassifier, self).__init__()
        self.encoder = AutoModel.from_pretrained("distilbert-base-uncased")
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_classes)
        )

    def forward(self, input_ids, attention_mask, targets=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]          # CLS token
        logits = self.classifier(cls_embedding)
        if targets is None:
            loss = None
        else:
            B, C = logits.shape
            logits = logits.view(B, C)
            targets = targets.view(B)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    


class DistilBertRegressor(nn.Module):
    def __init__(self, n_outputs=1):
        super(DistilBertRegressor, self).__init__()
        self.encoder = AutoModel.from_pretrained("distilbert-base-uncased")
        self.regressor = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_outputs)
        )

    def forward(self, input_ids, attention_mask, targets=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]      # CLS token
        preds = self.regressor(cls_embedding).squeeze(-1)
        if targets is None:
            loss = None
        else:
            targets = targets.float().view_as(preds)
            loss = F.mse_loss(preds, targets)
        return preds, loss
