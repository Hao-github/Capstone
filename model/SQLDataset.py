from torch.utils.data import Dataset
import torch
from model.CardinalityClassifier import CardinalityClassifier

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


