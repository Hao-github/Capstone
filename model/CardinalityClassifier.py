import torch
import torch.nn as nn
import torch.nn.functional as F

from model.const import OP_NAME2ID, TABLE_KEY2ID, TABLE_NAME2ID
from model.SQLOptimizer import SQLOptimizer
from database.TablePartitioner import TablePartitioner

# 定义各个维度大小
TABLE_ID_DIM = len(TABLE_NAME2ID)
SUBTABLE_ID_DIM = 50  # 每个表最多有 50 个子表
JOIN_KEY_DIM = len(TABLE_KEY2ID)
OP_DIM = len(OP_NAME2ID)


class PredicateEncoder(nn.Module):
    """
    用于编码 selection predicates 的 3 层 MLP 网络。
    每个谓词的输入维度是 JOIN_KEY + OP + 选择率，共 JOIN_KEY_DIM + OP_DIM + 1 维。
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
    """
    Cardinality 分类器，输入 SQL，预测该 SQL 是否能被高效 merge。
    构造包括：表、连接列、谓词编码。
    """

    def __init__(self, tp: TablePartitioner, final_hidden_dims=[64, 32]):
        super().__init__()
        self.tp = tp  # 数据库中用于估计选择率的分区器
        self.predicate_input_dim = JOIN_KEY_DIM + OP_DIM + 1
        self.predicate_encoder = PredicateEncoder(self.predicate_input_dim)

        # 最终输入维度 = 2 表ID + 2 子表ID + 2 join key + 32 (pool后的谓词向量)
        input_dim = TABLE_ID_DIM * 2 + SUBTABLE_ID_DIM * 2 + JOIN_KEY_DIM * 2 + 32
        self.fc1 = nn.Linear(input_dim, final_hidden_dims[0])
        self.fc2 = nn.Linear(final_hidden_dims[0], final_hidden_dims[1])
        self.fc3 = nn.Linear(final_hidden_dims[1], 1)
        self.bn1 = nn.BatchNorm1d(final_hidden_dims[0])
        self.bn2 = nn.BatchNorm1d(final_hidden_dims[1])

    def forward(self, table_feats, join_feats, predicate_feats):
        """
        前向传播，接收向量输入并输出一个概率。

        参数：
        - table_feats: (B, table_dim) 表和子表的 One-hot
        - join_feats: (B, join_key_dim) 连接列 One-hot
        - predicate_feats: (B, P, predicate_dim) 多个谓词向量组成的集合

        返回：
        - (B, 1) 的概率预测张量
        """
        B, P, D = predicate_feats.size()
        pred_encoded = self.predicate_encoder(predicate_feats.view(B * P, D))
        pred_encoded = pred_encoded.view(B, P, -1)
        pred_pooled = pred_encoded.mean(dim=1)  # 对谓词编码结果做平均池化
        x = torch.cat([table_feats, join_feats, pred_pooled], dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = torch.sigmoid(self.fc3(x))  # 输出为 0~1 概率
        return x

    def encode_sql(self, so: SQLOptimizer | str):
        """
        将一个 SQL 转换为三组输入特征向量
        """
        table_feat, join_feat = [], []

        if isinstance(so, str):
            so = SQLOptimizer(so, self.tp.get_schema_metadata())
        # 提取 JOIN 信息（默认只有一个 join）
        clause = so.join_expr[0]
        ltable, lcolumn, rtable, rcolumn = so.get_condition_from_expr(clause)

        # 表和子表 one-hot 编码
        for table in [ltable, rtable]:
            table_name, subtable_id = self._split_table(table)
            table_feat.extend(self._one_hot(TABLE_NAME2ID[table_name], TABLE_ID_DIM))
            table_feat.extend(self._one_hot(int(subtable_id), SUBTABLE_ID_DIM))

        # join key 编码
        for key in [lcolumn, rcolumn]:
            join_feat.extend(self._one_hot(TABLE_KEY2ID[key], JOIN_KEY_DIM))

        # 谓词编码（col + op + selectivity）
        predicate_set = []
        for expr in so.filter_exprs:
            col = expr.this
            vec = []
            vec += self._one_hot(TABLE_KEY2ID[col.name], JOIN_KEY_DIM)
            vec += self._one_hot(OP_NAME2ID[type(expr).__name__], OP_DIM)
            table = so.alias_dict.get(col.table, col.table)
            assert isinstance(table, str)
            expr_copy = expr.copy()
            expr_copy.this.set("table", None)
            sel = self.tp.qury_selectivity(
                self._split_table(table)[0], table, expr_copy.sql()
            )
            vec.append(sel)
            predicate_set.append(vec)

        return table_feat, join_feat, predicate_set

    def _one_hot(self, index: int, dim: int):
        return [1 if i == index else 0 for i in range(dim)]

    def _split_table(self, name: str):
        return name.split("_")[0], name.split("_")[1]

    def build_batch(self, so_list: list[SQLOptimizer]):
        """
        批量构造输入，用于多个 SQL 的向量化输入。

        参数:
        - model: 已初始化的 CardinalityClassifier 实例
        - sqls: list[str] SQL 查询列表

        返回:
        - table_feats: (B, table_input_dim)
        - join_feats: (B, join_input_dim)
        - predicate_feats: (B, max_pred_count, pred_dim)
        """

        parsed = [self.encode_sql(so) for so in so_list]
        max_pred = max(len(p[2]) for p in parsed)  # 对齐谓词个数
        table_feats, join_feats, pred_feats = [], [], []

        for table, join, preds in parsed:
            table_feats.append(torch.tensor(table, dtype=torch.float))
            join_feats.append(torch.tensor(join, dtype=torch.float))
            while len(preds) < max_pred:
                preds.append([0.0] * self.predicate_input_dim)
            pred_feats.append(torch.tensor(preds, dtype=torch.float))

        return (
            torch.stack(table_feats),
            torch.stack(join_feats),
            torch.stack(pred_feats),
        )
