from config.config import load_config
from database.TablePartitioner import TablePartitioner
from model.SQLOptimizer import SQLOptimizer
from model.CardinalityClassifier import CardinalityClassifier
from model.BloomFilter import BloomFilter
import torch
from torch.utils.data import random_split
from dataset import sql_list, label_list
import torch.optim as optim
from model.SQLDataset import SQLDataset
from model.SQLDataLoader import SQLDataLoader
from torch import nn
from sqlglot import expressions as exp

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


# =========================
# 合并后的推理主流程：cc → PIT → Bloom
# =========================
def predict_sql_result(
    sql: str,
    cc: "CardinalityClassifier",
    partitioner: "TablePartitioner",
    threshold: float = 0.5,
    pit_first: bool = True,
) -> bool:
    """
    1) 用 cc 预测（sigmoid 概率 > threshold 则 True）
    2) 否则（或不确定）根据第一个 join 条件：
        - 若 pit_first=True，先试 PIT 区间重叠，命中 True
        - 再回退 Bloom：右表建 Bloom，左表探测，命中 True
    3) 否则 False
    """
    so = SQLOptimizer(sql, partitioner.get_schema_metadata())

    # === 1) cc 预测 ===
    cc.eval()
    with torch.no_grad():
        table_feats, join_feats, pred_feats = cc.build_batch([so])
        logits = cc(table_feats, join_feats, pred_feats)  # [1] or [1,1]
        logits = logits.view(-1)
        prob = torch.sigmoid(logits)[0].item()
        if prob > threshold:
            return True

    # === 提取第一个 join 条件（用于 PIT / Bloom） ===
    ltable, lcolumn, rtable, rcolumn = so.get_condition_from_expr(so.join_expr[0])
    lschema = ltable.split("_")[0]
    rschema = rtable.split("_")[0]

    # === 2) PIT 优先 ===
    if pit_first:
        try:
            lhs_pit = so.get_partition_index_tree(lschema, lcolumn)
            rhs_intervals = (
                so.schema_metadata[rschema]
                .query("part_table == @rtable")
                .values.tolist()
            )
            if lhs_pit.query_interval_overlap(rhs_intervals):
                return True
        except Exception:
            # PIT 不可用/异常则继续 Bloom
            pass

    # === 3) Bloom 回退 ===
    try:
        rtable_data = partitioner.select_data(rschema, rtable, [rcolumn])
        # 防空表：容量至少为 1
        bf = BloomFilter(max(len(rtable_data), 1), 0.01)
        for row in rtable_data:
            # row 形如 (val,) 或 [val]
            if row:
                bf.add(row[0])

        ltable_data = partitioner.select_data(lschema, ltable, [lcolumn])
        for row in ltable_data:
            if row and (row[0] in bf):
                return True
    except Exception:
        # 数据拉取异常则认定未命中
        pass

    return False


# =========================
# 训练封装（带健壮性）
# =========================
def train_cc_if_data(
    cc: "CardinalityClassifier",
    sql_list,
    label_list,
    batch_size=32,
    epochs=50,
    lr=1e-3,
    use_oversample=True,
):
    """
    有数据则训练；自动处理空数据、验证集为空、形状匹配等。
    """
    if not sql_list or not label_list or len(sql_list) != len(label_list):
        print("[WARN] sql_list or label_list is empty or not match")
        return cc

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

    train_loader = SQLDataLoader(
        train_dataset, batch_size=batch_size, use_oversample=use_oversample
    )
    val_loader = SQLDataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("finish loading data")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(cc.parameters(), lr=lr)

    print("==> Start training")
    for epoch in range(epochs):
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
                val_correct += (preds == batch_labels).sum().item()
                val_samples += batch_labels.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_samples

        print(
            f"Epoch {epoch}, "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

    print("==> Training done")
    return cc


if __name__ == "__main__":
    # 加载配置

    config = load_config()
    partitioner = TablePartitioner(config)

    metadata = partitioner.get_schema_metadata()
    print("Partitioning orders table by range...")
    partitioner.partition_table("orders", 2500000, "o_orderkey")
    print("Partitioning lineitem table by range...")
    partitioner.partition_table("lineitem", 1500000, "l_orderkey")

    print("finish partitioning")

    so1 = SQLOptimizer(example_sql, metadata)
    left_1 = so1.alias_dict[so1.join_expr[0].this.table]
    right_1 = so1.alias_dict[so1.join_expr[0].expression.table]
    table_map1 = [
        {left_1: left, right_1: right}
        for left in metadata[left_1]["part_table"]
        for right in metadata[right_1]["part_table"]
    ]

    sql_list = []
    label_list = []
    for table_map in table_map1:
        parsed_query_copy = so1.parsed_query.copy()
        for table in parsed_query_copy.find_all(exp.Table):
            new_table = exp.Table(
                this=exp.to_identifier(table_map[table.name]),
                db=exp.to_identifier(table.name),
                alias=table.args.get("alias"),
            )
            table.replace(new_table)
        new_sql = parsed_query_copy.sql()
        sql_list.append(new_sql)
        result = partitioner.execute_query(new_sql).fetchone()
        if result is None:
            label_list.append(0)
        else:
            label_list.append(result[0])

    print("finish generating sql list and label list")

    cc = CardinalityClassifier(partitioner)

    print("finish initializing")

    cc = train_cc_if_data(
        cc,
        sql_list,
        label_list,
        batch_size=32,
        epochs=50,
        lr=1e-3,
        use_oversample=True,
    )

    print("finish loading data & training (if any)")

    # ====== 推理示例：cc → PIT → Bloom ======
    test_sql = example_sql.strip()
    ok = predict_sql_result(
        test_sql,
        cc=cc,
        partitioner=partitioner,
        threshold=0.5,
        pit_first=True,
    )
    print("Predicted feasible:", ok)
