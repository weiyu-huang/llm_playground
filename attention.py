import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionv1(nn.Module):
    def __init__(self, hidden_dim: int = 756) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.Q = nn.Linear(hidden_dim, hidden_dim)
        self.K = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X):
        # X shape is (batch_size, seq_len, hidden_dim)
        Q = self.Q(X)
        K = self.K(X)
        V = self.V(X)

        # shape: batch, seq, seq
        attention_value = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.hidden_dim)

        # shape: batch, seq, seq
        attention_weight = torch.softmax(attention_value, dim=-1)

        # shape: batch, seq, hidden
        output = torch.matmul(attention_weight, V)
        return output


# V2: efficiency improvement to combine Q, K, V
class SelfAttentionV2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, 3 * dim)

    def forward(self, X):  # X shape: (batch_size, seq_len, hidden_dim)
        QKV = self.proj(X)
        Q, K, V = torch.split(QKV, self.dim, dim=-1)

        attention = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dim)
        attention_weights = F.softmax(attention, dim=-1)  # (batch_size, seq_len, seq_len)
        print(attention_weights)

        return torch.matmul(attention_weights, V)  # (batch_size, seq_len, emb_dim)

# X = torch.randn(3, 2, 4)
# self_att_net = SelfAttentionV2(dim=4)
# self_att_net(X)


# V3: add details: dropout, mask,
class SelfAttentionV3(nn.Module):
    def __init__(self, dim, dropout_rate=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

        self.proj = nn.Linear(dim, 3 * dim)
        self.attention_dropout = nn.Dropout(dropout_rate)

        # Optional: output
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, X, attention_mask=None):
        # X shape: (batch, seq, dim)

        QKV = self.proj(X)
        Q, K, V = torch.split(QKV, self.dim, dim=-1)

        # (batch, seq, seq)
        attention_weight = Q @ K.transpose(-1, -2) / math.sqrt(self.dim)
        if attention_mask is not None:
            attention_weight = attention_weight.masked_fill(attention_mask, float('-1e20'))

        print(attention_weight)
        attention_weight = torch.softmax(attention_weight, dim=-1)
        print(attention_weight)
        attention_weight = self.attention_dropout(attention_weight)

        attention_result = attention_weight @ V

        output = self.output_proj(attention_result)
        return output

# X = torch.randn(3, 4, 4)
# mask = torch.tensor([
#     [False, True, True, True],  # First token only attends to itself
#     [False, False, True, True],  # Second token attends to itself and previous
#     [False, False, False, True],
#     [False, False, False, False]  # Last token can attend to all previous
# ])  # shape: batch, seq
# print(f"Before repeat shape: {mask.shape}")
# mask = mask.unsqueeze(dim=0).repeat(3, 1, 1)
#
# print(f"After repeat shape: {mask.shape}")
# self_att_net = SelfAttentionV3(dim=4)
# self_att_net(X, attention_mask=mask)


# V4: Used in interview
class SelfAttentionV4(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, X: torch.Tensor, attention_mask: torch.Tensor = None):
        # X shape is (batch, seq, dim)
        Q = self.query(X)
        K = self.key(X)
        V = self.value(X)

        # shape is (batch, seq, seq)
        attention_weight = Q @ K.transpose(-1, -2) / math.sqrt(self.hidden_dim)
        if attention_mask is not None:
            attention_weight = attention_weight.masked_fill(attention_mask, float('-inf'))
        attention_weight = F.softmax(attention_weight, dim=-1)
        print(attention_weight)

        attention_weight = self.dropout(attention_weight)

        # shape is (batch, seq, dim)
        return attention_weight @ V

X = torch.rand(3, 4, 5)
# shape is batch, seq, seq
mask = torch.triu(torch.ones(4, 4), diagonal=1).bool()  # (4,4)
mask = mask.unsqueeze(0).repeat(3, 1, 1)  # (1,4,4) -> (3,4,4)

net = SelfAttentionV4(hidden_dim=5)
net(X, attention_mask=mask)

