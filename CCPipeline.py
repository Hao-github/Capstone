import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split

from model.BloomFilter import BloomFilter
from model.CardinalityClassifier import CardinalityClassifier
from model.SQLDataLoader import SQLDataLoader
from model.SQLDataset import SQLDataset
from model.SQLOptimizer import SQLOptimizer
from database.TablePartitioner import TablePartitioner

logging.basicConfig(
    filename="output.log",  # 指定文件名
    level=logging.INFO,  # 记录 debug 以上的所有日志
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class CCPipeline:
    def __init__(
        self, partitioner: TablePartitioner, sql_list, label_list, config=None
    ):
        self.partitioner = partitioner
        self.sql_list = sql_list
        self.label_list = [1 if label > 0 else 0 for label in label_list]
        self.config = config or {
            "batch_size": 32,
            "epochs": 20,
            "lr": 1e-3,
            "threshold": 0.5,
            "use_oversample": True,
        }
        self.model = CardinalityClassifier(partitioner)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["lr"])

    def _build_dataset(self):
        dataset = SQLDataset(self.model, self.sql_list, self.label_list)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size]
        )

        self.train_loader = SQLDataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            use_oversample=self.config["use_oversample"],
        )
        self.val_loader = SQLDataLoader(
            self.val_dataset, batch_size=self.config["batch_size"], shuffle=False
        )
        logging.info(
            f"Train Size: {len(self.train_dataset)}, Val Size: {len(self.val_dataset)}"
        )

    def train(self):
        self._build_dataset()
        for epoch in range(self.config["epochs"]):
            self.model.train()
            total_loss, total_correct, total_samples = 0, 0, 0
            for table_feats, join_feats, predicate_feats, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(table_feats, join_feats, predicate_feats)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

            train_acc = total_correct / total_samples
            train_loss = total_loss / len(self.train_loader)

            val_acc, val_loss = self._evaluate()
            logging.info(
                f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
            )

    def _evaluate(self):
        self.model.eval()
        total_loss, total_correct, total_samples = 0, 0, 0
        with torch.no_grad():
            for table_feats, join_feats, predicate_feats, labels in self.val_loader:
                outputs = self.model(table_feats, join_feats, predicate_feats)
                loss = self.criterion(outputs, labels)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                total_loss += loss.item()
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
        return total_correct / total_samples, total_loss / len(self.val_loader)

    # def get_all_results(self)

    def predict_by_pit(self, so: SQLOptimizer):
        ltable, lcol, rtable, rcol = so.get_condition_from_expr(so.join_expr[0])
        lhs_pit = so.get_partition_index_tree(ltable, lcol)
        rhs_interval = so.select_intervals(rtable, rcol)
        ret = []
        for interval in rhs_interval:
            ret.extend(lhs_pit.query_interval_overlap(interval))
        return ret

    def predict(
        self, so: SQLOptimizer, sql_list: list[str]
    ) -> tuple[list[bool], list[float]]:
        """
        封装完整推理流程：先 cc 预测，再 PIT，再 Bloom
        """
        results = []
        self.model.eval()
        with torch.no_grad():
            so_list = [
                SQLOptimizer(s, self.partitioner.get_schema_metadata())
                for s in sql_list
            ]
            table_feats, join_feats, pred_feats, pred_mask = self.model.build_batch(
                so_list
            )
            probs = torch.sigmoid(
                self.model(table_feats, join_feats, pred_feats, pred_mask)
            ).view(-1)
            logger.info(f"Predictions: {probs}")
        for idx, prob in enumerate(probs):
            if prob.item() > self.config["threshold"]:
                results.append(True)
                continue  # classifier 已判定为可行
            so = so_list[idx]
            ltable, lcol, rtable, rcol = so.get_condition_from_expr(so.join_expr[0])
            lschema = ltable.split("_")[0]
            rschema = rtable.split("_")[0]
            # === 尝试 PIT 判断 ===
            try:
                print(f"Try to use PIT for {so.query}, idx:{idx}")
                if self._check_pit(so, lschema, lcol, rschema, rtable):
                    results.append(True)
                    continue
            except Exception:
                pass

            # === 最后尝试 Bloom 判断 ===
            try:
                print(f"Try to use Bloom for {so.query}, idx:{idx}")
                # for test
                # results.append(False)
                results.append(
                    self._check_bloom(lschema, ltable, lcol, rschema, rtable, rcol)
                )
            except Exception:
                results.append(False)

        return results, probs.tolist()

    def _check_pit(self, so: SQLOptimizer, ltable, lcol, rtable, rcol):
        try:
            lhs_pit = so.get_partition_index_tree(ltable, lcol)
            rhs_interval = so.select_intervals(rtable, rcol)[0]
            return lhs_pit.query_interval_overlap(rhs_interval)
        except Exception as e:
            print(e)
            return False

    def _check_bloom(self, lschema, ltable, lcol, rschema, rtable, rcol):
        try:
            rdata = self.partitioner.select_data(rschema, rtable, [rcol])
            bf = BloomFilter(max(1, len(rdata)), 0.01)
            for row in rdata:
                if row:
                    bf.add(str(row[0]))
            ldata = self.partitioner.select_data(lschema, ltable, [lcol])
            for row in ldata:
                if str(row[0]) in bf:
                    return True
        except Exception:
            pass
        return False
