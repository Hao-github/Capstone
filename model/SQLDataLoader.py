import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np


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


class SQLDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        use_oversample=False,
    ):
        if not use_oversample:
            super(SQLDataLoader, self).__init__(
                dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
            )
        else:
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

            sampler = WeightedRandomSampler(
                sample_weights,  # type: ignore
                num_samples=len(sample_weights),
                replacement=True,
            )
            super(SQLDataLoader, self).__init__(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                drop_last=False,
                collate_fn=collate_fn,
            )