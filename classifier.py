from config.config import load_config
from database.TablePartitioner import TablePartitioner
from model.SQLOptimizer import SQLOptimizer
from model.CardinalityClassifier import CardinalityClassifier
from model.BloomFilter import BloomFilter
import torch
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from dataset import label_list, sql_list
from dataset_2 import label_list as label_list_2, sql_list as sql_list_2
from dataset_3 import label_list as label_list_3, sql_list as sql_list_3
from dataset_4 import label_list as label_list_4, sql_list as sql_list_4

import torch.optim as optim
from torch import nn

import random
import numpy as np


def make_oversampled_loader(dataset, batch_size, collate_fn, drop_last=False):
    """
    对二分类训练集做上采样（按少数类赋更高采样权重）。
    """
    labels = []
    for i in range(len(dataset)):
        item = dataset[i]
        y = item[-1]                    
        if torch.is_tensor(y):
            y = int(y.detach().cpu().view(-1)[0].item())
        else:
            y = int(y)
        labels.append(y)

    labels = np.asarray(labels, dtype=np.int64)
    # 计算每个样本的采样权重：类别样本数的倒数
    class_counts = np.bincount(labels, minlength=2) 
    # 避免除零
    class_counts = np.where(class_counts == 0, 1, class_counts)
    sample_weights = (1.0 / class_counts[labels]).astype(np.float64)

    sampler = WeightedRandomSampler(sample_weights, # type: ignore
                                    num_samples=len(sample_weights),
                                    replacement=True)

    return DataLoader(dataset,
                      batch_size=batch_size,
                      sampler=sampler,        
                      drop_last=drop_last,
                      collate_fn=collate_fn)




class SQLDataset(Dataset):
    def __init__(self, model: CardinalityClassifier, sql_list, labels):
        self.model = model
        self.sql_list = sql_list
        self.labels = labels

    def __len__(self):
        return len(self.sql_list)

    def __getitem__(self, idx):
        sql = self.sql_list[idx]
        label = self.labels[idx]
        # 提取特征
        table_feat, join_feat, predicate_set = self.model.encode_sql(sql)
        return (
            torch.tensor(table_feat, dtype=torch.float),
            torch.tensor(join_feat, dtype=torch.float),
            torch.tensor(predicate_set, dtype=torch.float),
            torch.tensor([label], dtype=torch.float),
        )


def collate_fn(batch):
    table_feats, join_feats, predicate_sets, labels = zip(*batch)
    max_len = max(p.shape[0] for p in predicate_sets)

    padded_preds = []
    for p in predicate_sets:
        pad_size = max_len - p.shape[0]
        if pad_size > 0:
            pad = torch.zeros((pad_size, p.shape[1]))
            p = torch.cat([p, pad], dim=0)
        padded_preds.append(p)

    return (
        torch.stack(table_feats),
        torch.stack(join_feats),
        torch.stack(padded_preds),
        torch.stack(labels),
    )


example_sql = """
SELECT
    *
FROM
    orders_13 AS o,
    lineitem_13 AS l
WHERE
    l.l_orderkey = o.o_orderkey
AND
    l.l_linenumber < 6
AND 
    o.o_orderstatus = 'O'
"""

# def predict_sql_result(sql: str) -> bool:
#     so = SQLOptimizer(sql, partitioner.get_schema_metadata())
#     table_feats, join_feats, pred_feats = cc.build_batch([so])
#     result = cc.forward(table_feats, join_feats, pred_feats)
#     if result > 0.5:
#         return True
#     ltable, lcolumn, rtable, rcolumn = so.get_condition_from_expr(so.join_expr[0])
#     lschema = ltable.split("_")[0]
#     rschema = rtable.split("_")[0]
#     # use pit
#     lhs_pit = so.get_partition_index_tree(lschema, lcolumn)
#     rhs_intervals = (
#         so.schema_metadata[rschema].query("part_table == @rtable").values.tolist()
#     )
#     if lhs_pit.query_interval_overlap(rhs_intervals):
#         return True
#     # use bloom
#     rtable_data = partitioner.select_data(rschema, rtable, [rcolumn])
#     bf = BloomFilter(len(rtable_data), 0.01)
#     for row in rtable_data:
#         bf.add(row[0])
#     ltable_data = partitioner.select_data(lschema, ltable, [lcolumn])
#     for row in ltable_data:
#         if row[0] in bf:
#             return True
#     return False
if __name__ == "__main__":
    config = load_config()
    partitioner = TablePartitioner(config)
    cc = CardinalityClassifier(partitioner)

    print("finish initializing")

    label_list = label_list_2 + label_list_3 + label_list_4 + label_list
    sql_list = sql_list_2 + sql_list_3 + sql_list_4 + sql_list
    # 标签归一化成 {0,1}
    labels = [1 if label > 0 else 0 for label in label_list]
    print(f"positive ratio: {sum(labels) / len(labels):.4f}")

    # 构建完整数据集
    full_dataset = SQLDataset(cc, sql_list, labels)

    # === 8:2 划分训练集/验证集 ===
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # train_loader = DataLoader(
    #     train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
    # )
    train_loader = make_oversampled_loader(train_dataset, batch_size=32, collate_fn=collate_fn)
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
    )

    print("finish loading data")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(cc.parameters(), lr=1e-3)
    print("finish initializing optimizer")

    for epoch in range(10):
        # === Train ===
        cc.train()
        total_loss, total_correct, total_samples = 0, 0, 0

        for table_feats, join_feats, predicate_feats, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = cc(table_feats, join_feats, predicate_feats)  # logits
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            total_correct += (preds == batch_labels).sum().item()
            total_samples += batch_labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = total_correct / total_samples

        # === Validation ===
        cc.eval()
        val_loss, val_correct, val_samples = 0, 0, 0
        with torch.no_grad():
            for table_feats, join_feats, predicate_feats, batch_labels in val_loader:
                outputs = cc(table_feats, join_feats, predicate_feats)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                # print(preds)
                # print(batch_labels)
                val_correct += (preds == batch_labels).sum().item()
                val_samples += batch_labels.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_samples

        print(
            f"Epoch {epoch}, "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )
