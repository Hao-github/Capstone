from typing import Any

import networkx as nx
import pandas as pd
from sqlglot import expressions as exp
from sqlglot import parse_one

from model.PartitionIndexTree import PartitionIndexTree


class SQLOptimizer:
    pit_dict: dict[str, PartitionIndexTree] = {}

    def __init__(self, query: str, schema_metadata: dict[str, pd.DataFrame]):
        self.query = query
        self.parsed_query = parse_one(query)
        self.schema_metadata = schema_metadata
        self.alias_dict: dict[str, str] = {
            t.alias: t.name for t in self.parsed_query.find_all(exp.Table)
        }
        self.filter_exprs, self.join_expr = self.__analysis_query()

    def __analysis_query(self):
        where_clause = self.parsed_query.find(exp.Where)
        filter_list = []
        merge_conditions = []

        def recurse(expr):
            if isinstance(expr, exp.And):
                recurse(expr.left)
                recurse(expr.right)
                return
            if isinstance(expr, exp.EQ):
                if isinstance(expr.left, exp.Column) and isinstance(
                    expr.right, exp.Column
                ):
                    merge_conditions.append(expr)
                    return
            filter_list.append(expr)

        if where_clause:
            recurse(where_clause.this)

        return filter_list, merge_conditions

    def select_intervals(self, table: str, column: str) -> list[tuple]:
        return (
            self.schema_metadata[table]
            .groupby([f"{column}_start", f"{column}_end"], as_index=False)["part_table"]
            .apply(list)
            .to_dict(orient="split")["data"]  # type: ignore
        )

    def select_all_intervals(self, table: str) -> list[str]:
        return self.schema_metadata[table]["part_table"].tolist()  # type: ignore

    def get_partition_index_tree(self, table: str, column: str) -> PartitionIndexTree:
        if pit := self.pit_dict.get(f"{table}_{column}", None):
            return pit
        intervals = self.select_intervals(table, column)
        pit = PartitionIndexTree(intervals)
        self.pit_dict[f"{table}_{column}"] = pit
        return pit

    def get_baseline_query(self, join_order: list[str] | None = None):
        """
        将原始 SQL 改写为子表 UNION ALL 聚合 + Leading Hint 的 baseline 形式，保留表别名。

        参数:
            join_order: 使用别名构成的 JOIN 顺序，例如 ['a', 'b']

        返回:
            改写后的 SQL 字符串，包含 Hint 和子表 UNION ALL 聚合
        """
        raw_query = parse_one(self.query)
        for t in raw_query.find_all(exp.Table):
            name = t.name
            if name not in self.schema_metadata.keys():
                continue
            value = self.schema_metadata[name]["part_table"].tolist()
            union_sql = " UNION ALL ".join(f"SELECT * FROM {name}.{t}" for t in value)
            t.set("this", exp.Subquery(this=parse_one(union_sql)))

        rewritten_sql = raw_query.sql().strip()
        return (
            f"/*+ Leading({' '.join(join_order)}) */\n{rewritten_sql}"
            if join_order
            else rewritten_sql
        )

    def schema_exists(self, schema_name: str) -> bool:
        return schema_name in self.schema_metadata.keys()

    def get_condition_from_expr(self, expr: exp.EQ) -> tuple[str, str, str, str]:
        """
        获取等值条件表达式的左右表名和列名。

        输入：
            expr: 等值条件表达式，如 a.id = b.id
        输出：
            (left_table, left_column, right_table, right_column)
        """
        left, right = expr.left, expr.right
        assert isinstance(left, exp.Column) and isinstance(right, exp.Column)

        left_table = self.alias_dict.get(left.table, left.table)
        right_table = self.alias_dict.get(right.table, right.table)
        left_column = left.name
        right_column = right.name
        return left_table, left_column, right_table, right_column

    @property
    def filters_list(self) -> list[str]:
        """
        粗略提取 WHERE 中的 filter 子句，返回 ['filter1', 'filter2',...]
        """
        return [f.sql() for f in self.filter_exprs]

    @property
    def filters(self) -> str:
        """
        粗略提取 WHERE 中的 filter 子句，返回 'filter1 AND filter2 AND...'
        """

        all_filter: exp.Expression = exp.true()
        for filter_expr in self.filter_exprs:
            all_filter = exp.And(this=all_filter, expression=filter_expr)
        return all_filter.sql()

    def get_cluster_via_pit(self, expr: exp.EQ) -> dict[str, Any]:
        """
        构造 left 表和 right 表下所有子表的配对，返回配对 dict。

        输入：
            expr: 等值条件表达式，如 a.id = b.id
        输出：
            {
                "join": (left_table, right_table),
                "pairs": [
                    ([left_group], [right_group]),
                    ...
                ],
                "on": (left_column, right_column),
            }
            例如：{
                "join": ("R", "S"),
                "pairs": [
                    {["R1"], ["S12", "S13"]},
                    {["R2"], ["S21", "S22"]},
                    ...
                ],
                "on": ("id", "id"),
            }
        """

        G = nx.Graph()
        ltable, lcolumn, rtable, rcolumn = self.get_condition_from_expr(expr)
        # 假如left_table和right_table都存在分区，则根据pit生成edges
        # 否则，生成一个边，连接left_table和right_table或其下的所有分区
        if self.schema_exists(ltable):
            lhs_pit = self.get_partition_index_tree(ltable, lcolumn)
            rhs_keys = (
                self.select_intervals(rtable, rcolumn)
                if self.schema_exists(rtable)
                else [(0, 1000000000, rtable)]
            )
            for key in rhs_keys:
                for overlap in lhs_pit.query_interval_overlap(key):
                    G.add_node(overlap[0], bipartite=0)
                    G.add_node(overlap[1], bipartite=1)
                    G.add_edge(overlap[0], overlap[1])
        else:
            G.add_node(ltable, bipartite=0)
            rhs_keys = (
                self.select_all_intervals(rtable)
                if self.schema_exists(rtable)
                else [rtable]
            )
            for key in rhs_keys:
                G.add_node(key, bipartite=1)
                G.add_edge(ltable, key)

        # 构造bipartite graph
        values = []
        for componenent in nx.connected_components(G):
            # lefts = natsorted([n for n in componenent if G.nodes[n]["bipartite"] == 0])
            # rights = natsorted([n for n in componenent if G.nodes[n]["bipartite"] == 1])
            lefts = [n for n in componenent if G.nodes[n]["bipartite"] == 0]
            rights = [n for n in componenent if G.nodes[n]["bipartite"] == 1]
            values.append((lefts, rights))

        return {
            "join": (ltable, rtable),
            "on": (lcolumn, rcolumn),
            "pairs": values,
        }

    def get_cluster_map(self, to_json: bool = False):
        """
        构造 cluster 图，返回 cluster_map。
        输入：
            so: SQLOptimizer 对象
            to_json: 是否输出 cluster_map 到 json 文件
        输出：
            cluster_map: 字典，key 为 (left_table, right_table) 值，value 为 cluster 信息
        Example:
            {
                ("company_type", "info_type"): {
                    "join": ("company_type", "info_type"),
                    "on": ("id", "id"),
                    "pairs": [
                        (["company_type1"], ["info_type1"]),
                        (["company_type2"], ["info_type2"]),
                        ...
                    ],
                },
                ...
            }
        """
        cluster_map = {}
        for clause in self.join_expr:
            clusters = self.get_cluster_via_pit(clause)
            cluster_map.update({clusters["join"]: clusters})
        if to_json:
            import json

            json.dump(
                list(cluster_map.values()),
                open("output/cluster_map.json", "w"),
                indent=4,
            )

        return cluster_map

    @property
    def cluster_map(self):
        return self.get_cluster_map()
