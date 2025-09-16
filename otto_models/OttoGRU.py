import torch
import torch.nn as nn
import torch.nn.functional as F

class OttoGRU(nn.Module):
    def __init__(self, aid_num_embedding, aid_dim_embedding, action_dim, hidden_dim, num_layers):
        self.embedding = nn.Embedding(aid_num_embedding, aid_dim_embedding)
        self.action_dim = action_dim
        self.gru = nn.GRU(aid_dim_embedding + action_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

        self.fcc_aid = nn.Linear(aid_dim_embedding + action_dim, aid_num_embedding)
        self.fcc_action = nn.Linear(aid_dim_embedding + action_dim, action_dim)

    def forward(self, aid, action):
        aid_emb = self.embedding(aid)
        action_emb = F.one_hot(action, num_classes=self.self.action_dim)
        input = torch.concat([aid_emb, action_emb], dim=0)
        _, output = self.gru(input)

        aid_output = self.fcc_aid(output)
        action_output = self.fcc_action(output)

        return aid_output, action_output
