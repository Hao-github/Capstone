import torch
import torch.nn as nn
import torch.nn.functional as F


class PredicateEncoder(nn.Module):
    """
    对 selection predicates 编码的 3 层 MLP，输入每个 predicate 的嵌入三元组（col, op, val）
    """

    def __init__(self, input_dim, hidden_dims=[64, 64, 32]):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.bn3 = nn.BatchNorm1d(hidden_dims[2])

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        return x


class CardinalityClassifier(nn.Module):
    def __init__(
        self,
        table_id_dim,
        subtable_id_dim,
        join_key_dim,
        predicate_input_dim,
        final_hidden_dims=[64, 32],
    ):
        super().__init__()

        # 子模块：编码谓词集合（一个集合中多个谓词）
        self.predicate_encoder = PredicateEncoder(predicate_input_dim)

        # 主 MLP 网络（拼接所有特征后的主分类器）
        input_dim = (
            table_id_dim * 2 + subtable_id_dim * 2 + join_key_dim + 32
        )  # 32为pool后谓词维度
        self.fc1 = nn.Linear(input_dim, final_hidden_dims[0])
        self.fc2 = nn.Linear(final_hidden_dims[0], final_hidden_dims[1])
        self.fc3 = nn.Linear(final_hidden_dims[1], 1)
        self.bn1 = nn.BatchNorm1d(final_hidden_dims[0])
        self.bn2 = nn.BatchNorm1d(final_hidden_dims[1])

    def forward(self, table_feats, subtable_feats, join_feats, predicate_feats):
        """
        - table_feats: (batch, 2 * table_id_dim)
        - subtable_feats: (batch, 2 * subtable_id_dim)
        - join_feats: (batch, join_key_dim)
        - predicate_feats: (batch, num_predicates, predicate_input_dim)
        """
        # 编码谓词
        B, P, D = predicate_feats.size()
        pred_encoded = self.predicate_encoder(
            predicate_feats.view(B * P, D)
        )  # 编码每个谓词
        pred_encoded = pred_encoded.view(B, P, -1)  # reshape 回 batch
        pred_pooled = pred_encoded.mean(dim=1)  # avg pool, 维度为 (B, 32)

        # 合并所有特征
        x = torch.cat([table_feats, subtable_feats, join_feats, pred_pooled], dim=1)

        # 主分类器
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = torch.sigmoid(self.fc3(x))  # 输出概率值

        return x
