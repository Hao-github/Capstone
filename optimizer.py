from __future__ import annotations

from collections import defaultdict
from functools import reduce
from typing import Any

import networkx as nx
from sqlglot import expressions as exp

from config.config import load_config
from database.TablePartitioner import TablePartitioner
from model.SQLOptimizer import SQLOptimizer


def enumerate_join_orders(edges: list[tuple[str, str]]) -> list[list[tuple[str, str]]]:
    """
    enumerate all possible join orders for a list of join edges
    to reduce the complexity of the join order search, we only consider
    the cases when we add a new join edge to the existing join order,
    the edges can still be connected graph.
    Input:
        edges: [('t1', 't2'), ('t2', 't3'), ('t3', 't4'), ('t1', 't3)]
        K: limitation or join orders
    Output:
        [
            [('t1', 't2'), ('t2', 't3'), ('t3', 't4')],
            [('t4', 't3'), ('t3', 't2'), ('t2', 't1')],
            [('t1', 't3'), ('t3', 't2'), ('t3', 't4')],
            ...
        ] example sql has 12 possible join orders
    """
    results = []
    edges_map = defaultdict(list)
    for edge in edges:
        left, right = edge
        edges_map[left].append((left, right))
        edges_map[right].append((right, left))
    nodes_num = len(edges_map)

    def dfs(
        path: list[tuple[str, str]],
        visited: set[str],
        waiting_edges: list[tuple[str, str]],
    ):
        while waiting_edges:
            # ensure left is in the visited set
            left, right = waiting_edges.pop()
            if right in visited:
                continue
            visited.add(right)
            path.append((left, right))
            if len(visited) == nodes_num:
                results.append(path[:])
            else:
                dfs(path, visited, waiting_edges + edges_map[right])
            path.pop()
            visited.remove(right)

    if nodes_num > 2:
        for edge in edges:
            left, right = edge
            dfs([edge], {left, right}, edges_map[left] + edges_map[right])
    else:
        results.append(edges)

    return results


def resolve_join_union_order(
    join_order: list[tuple[str, str]],
    cluster_map: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    返回 join-union 顺序结构，仅包含 join 关系和子表聚合结构。
    输出示例：
    [
            {
                "join": ("R", "S"),
                "pairs": [
                    (["R1"], ["S11", "S12"]),
                    (["R2"], ["S21", "S22"])
                    ...
                ],
                "on": ("id", "id"),
            }
        ...
    ]
    """
    plan_steps = []

    for join in join_order:
        t1, t2 = join
        if pair_list := cluster_map.get((t1, t2), {}):
            plan_steps.append(
                {"join": (t1, t2), "pairs": pair_list["pairs"], "on": pair_list["on"]}
            )
        elif pair_list := cluster_map.get((t2, t1), {}):
            reversed_pairs = [
                (p[1], p[0]) for p in pair_list["pairs"]
            ]  # 交换左右两边的聚合键
            plan_steps.append(
                {
                    "join": (t1, t2),
                    "pairs": reversed_pairs,
                    "on": (pair_list["on"][1], pair_list["on"][0]),
                }
            )
        else:
            raise ValueError(f"No join condition found for {t1} and {t2}")
    return plan_steps


class JoinExpr:
    def __init__(self, alias, left_col, right_col, left_table_expr, right_table_expr):
        self.join_expr = self.create_join_expr(
            left_col, right_col, left_table_expr, right_table_expr
        )
        self.alias = alias
        self.related_tables: set[str] = set()

    def create_join_expr(self, left_col, right_col, left_table_expr, right_table_expr):
        if left_col == right_col:
            return (
                exp.select("*")
                .from_(left_table_expr)
                .join(right_table_expr, using=[left_col])
            )
        return (
            exp.select("*")
            .from_(left_table_expr)
            .join(
                right_table_expr,
                on=exp.EQ(
                    this=exp.column(left_col, table=left_table_expr.alias_or_name),
                    expression=exp.column(
                        right_col, table=right_table_expr.alias_or_name
                    ),
                ),
            )
        )

    def __str__(self):
        return self.join_expr.sql()


class Block:
    def __init__(self, tbls: list[str], table: str, alias: str, join_expr: JoinExpr):
        self.cte_expr = reduce(
            lambda x, y: x.union(y, distinct=True),
            [exp.select("*").from_(exp.to_table(t, db=table)) for t in tbls],
        )
        self.alias = alias
        self.join_expr = join_expr
        self.join_expr.related_tables.update(tbls)


def generate_recursive_cluster_sql(join_union_order: list[dict[str, Any]]) -> str:
    """
    递归构造聚合式 SQL
    INPUT:
        join_union_order: 输出 resolve_join_union_order 函数的结果
        join_keys: 用于指定 join 条件，如果不指定，则使用默认 join 条件
    OUTPUT:
        聚合式 SQL
    Example:
        join_union_order = [
            {
                "join": ("R", "S"),
                "pairs": [
                    (["R1"], ["S11", "S12"]),
                    (["R2"], ["S21", "S22"])
                    ...
                ],
                "on": ("id", "id"),
            }
        ]
    """

    # 结构示例：
    # Before:
    # T join S on T.id = S.id
    # table     cte_expr                                       join_expr
    # R1        SELECT * FROM R1                               r1 JOIN s1 ON r1.id = s1.id
    # R2        SELECT * FROM R2                               r2 JOIN s2 ON r2.id = s2.id
    # S11       SELECT * FROM S11 UNION ALL SELECT * FROM S12  r1 JOIN s1 ON r1.id = s1.id
    # S12       SELECT * FROM S11 UNION ALL SELECT * FROM S12  r1 JOIN s1 ON r1.id = s1.id
    # S21       SELECT * FROM S21 UNION ALL SELECT * FROM S22  r2 JOIN s2 ON r2.id = s2.id
    # S22       SELECT * FROM S21 UNION ALL SELECT * FROM S22  r2 JOIN s2 ON r2.id = s2.id
    # T         SELECT * FROM T                                ?
    # After:
    # table     cte_expr                                       join_expr
    # R1        SELECT * FROM R1                               t JOIN (r1 JOIN s1 ON r1.id = s1.id UNION ALL r2 JOIN s2 ON r2.id = s2.id) AS r1_s1_r2_s2 ON r1_s1_r2_s2.id = t.id
    # R2        SELECT * FROM R2                               t JOIN (r1 JOIN s1 ON r1.id = s1.id UNION ALL r2 JOIN s2 ON r2.id = s2.id) AS r1_s1_r2_s2 ON r1_s1_r2_s2.id = t.id
    # S11       SELECT * FROM S11 UNION ALL SELECT * FROM S12  t JOIN (r1 JOIN s1 ON r1.id = s1.id UNION ALL r2 JOIN s2 ON r2.id = s2.id) AS r1_s1_r2_s2 ON r1_s1_r2_s2.id = t.id
    # S12       SELECT * FROM S11 UNION ALL SELECT * FROM S12  t JOIN (r1 JOIN s1 ON r1.id = s1.id UNION ALL r2 JOIN s2 ON r2.id = s2.id) AS r1_s1_r2_s2 ON r1_s1_r2_s2.id = t.id
    # S21       SELECT * FROM S21 UNION ALL SELECT * FROM S22  t JOIN (r1 JOIN s1 ON r1.id = s1.id UNION ALL r2 JOIN s2 ON r2.id = s2.id) AS r1_s1_r2_s2 ON r1_s1_r2_s2.id = t.id
    # S22       SELECT * FROM S21 UNION ALL SELECT * FROM S22  t JOIN (r1 JOIN s1 ON r1.id = s1.id UNION ALL r2 JOIN s2 ON r2.id = s2.id) AS r1_s1_r2_s2 ON r1_s1_r2_s2.id = t.id
    # T         SELECT * FROM T                                t JOIN (r1 JOIN s1 ON r1.id = s1.id UNION ALL r2 JOIN s2 ON r2.id = s2.id) AS r1_s1_r2_s2 ON r1_s1_r2_s2.id = t.id
    table2block: dict[str, Block] = {}  # 存放 table-Block
    alias2join_expr: dict[str, JoinExpr] = {}  # 存放 alias-JoinExpr
    for step, join_struct in enumerate(join_union_order):
        left_col, right_col = join_struct["on"]
        left_table, right_table = join_struct["join"]
        if step == 0:
            for step_idx, pair in enumerate(join_struct["pairs"]):
                lefts, rights = pair
                left_alias = f"{left_table}_{step}_{step_idx}"
                right_alias = f"{right_table}_{step}_{step_idx}"
                join_alias = f"{left_alias}_{right_alias}_{step_idx}"

                left_table_expr = exp.alias_(
                    exp.paren(
                        reduce(
                            lambda x, y: x.union(y, distinct=False),
                            [
                                exp.select("*").from_(exp.to_table(t, db=left_table))
                                for t in lefts
                            ],
                        )
                    ),
                    left_alias,
                )
                right_table_expr = exp.alias_(
                    exp.paren(
                        reduce(
                            lambda x, y: x.union(y, distinct=False),
                            [
                                exp.select("*").from_(exp.to_table(t, db=right_table))
                                for t in rights
                            ],
                        )
                    ),
                    right_alias,
                )
                join_expr = JoinExpr(
                    join_alias, left_col, right_col, left_table_expr, right_table_expr
                )
                left_block = Block(lefts, left_table, left_alias, join_expr)
                right_block = Block(rights, right_table, right_alias, join_expr)
                alias2join_expr[join_alias] = join_expr
                for table in lefts:
                    table2block[table] = left_block
                for table in rights:
                    table2block[table] = right_block
            continue

        values = create_biparite(table2block, join_struct)
        for step_idx, pair in enumerate(values):
            lefts, rights = pair
            left_alias = f"{left_table}_{step}_{step_idx}"
            right_alias = f"{right_table}_{step}_{step_idx}"
            join_alias = f"{left_alias}_{right_alias}_{step_idx}"
            lefts_expr_set = set()
            realated_left_tables = set()
            for lefts_alias in lefts:
                lefts_expr_set.add(alias2join_expr[lefts_alias].join_expr)
                realated_left_tables.update(alias2join_expr[lefts_alias].related_tables)

            if len(lefts_expr_set) > 1:
                left_table_expr = exp.alias_(
                    exp.paren(
                        reduce(lambda x, y: x.union(y, distinct=False), lefts_expr_set)
                    ),
                    left_alias,
                )
            else:
                left_table_expr = exp.alias_(
                    exp.paren(lefts_expr_set.pop()), left_alias
                )

            right_table_expr = exp.alias_(
                exp.paren(
                    reduce(
                        lambda x, y: x.union(y, distinct=False),
                        [
                            exp.select("*").from_(exp.to_table(t, db=right_table))
                            for t in rights
                        ],
                    )
                ),
                right_alias,
            )
            join_expr = JoinExpr(
                join_alias, left_col, right_col, left_table_expr, right_table_expr
            )
            join_expr.related_tables.update(realated_left_tables)
            right_block = Block(rights, right_table, right_alias, join_expr)
            alias2join_expr[join_alias] = join_expr
            for table in rights:
                table2block[table] = right_block
            for table in realated_left_tables:
                table2block[table].join_expr = join_expr
    remaining_blocks: set[Block] = set(table2block.values())
    final_expr_list = list(set(block.join_expr.join_expr for block in remaining_blocks))
    final_expr = reduce(lambda x, y: x.union(y, distinct=False), final_expr_list)

    # for block in remaining_blocks:
    #     final_expr = final_expr.with_(block.alias, block.cte_expr, materialized=False)

    return final_expr.sql()


def create_biparite(table2block: dict[str, Block], join_struct):
    G = nx.Graph()
    for pair in join_struct["pairs"]:
        lefts, rights = pair
        for table in lefts:
            G.add_node(table2block[table].join_expr.alias, bipartite=0)
            for right in rights:
                G.add_node(right, bipartite=1)
                G.add_edge(table2block[table].join_expr.alias, right)
        # 构造bipartite graph
    values = []
    for componenent in nx.connected_components(G):
        values.append(
            (
                [n for n in componenent if G.nodes[n]["bipartite"] == 0],
                [n for n in componenent if G.nodes[n]["bipartite"] == 1],
            )
        )

    return values


def optimizer(so: SQLOptimizer) -> tuple[list[str], str]:
    cluster_map = so.get_cluster_map(to_json=True)
    join_orders = enumerate_join_orders(list(cluster_map.keys()))
    baseline_query = so.get_baseline_query()
    ret = []
    for order in join_orders:
        join_union_order = resolve_join_union_order(order, cluster_map)
        final_sql = generate_recursive_cluster_sql(join_union_order)
        ret.append(final_sql)
    return ret, baseline_query
    # sql_with_hints.append(final_sql)


example_sql = """
SELECT
    *
FROM
    orders AS o,
    lineitem AS l
WHERE
    l.l_orderkey = o.o_orderkey;

"""


if __name__ == "__main__":
    config = load_config()
    tp = TablePartitioner(config)
    so = SQLOptimizer(example_sql, tp.get_schema_metadata())
    sql_with_hints, baseline_query = optimizer(so)
    with open("output/baseline_query.sql", "w") as f:
        f.write(baseline_query)
    # print("baseline_query:", tp.measure_merge_time(baseline_query))
    for idx, sql in enumerate(sql_with_hints):
        with open(f"output/sql_with_hints_{idx}.sql", "w") as f:
            f.write(sql)
        # try:
        #     print(f"sql_{idx}:", tp.measure_merge_time(sql))
        # except Exception:
        #     print(f"sql_{idx} cannot be executed")

    # 按行划分的情况下
    # union 的乱序会影响性能，导致前几次测试耗时升高
    # 要利用pit的原因， where退化成join，也会导致耗时升高
    # 一结合就会产生一个情况
    # 按行划分的话基本上表现就是比baseline更差

    # 按range划分的情况下
    # 依然是60倍的提升
    # R join S
    # R1 > S1
    # S1 > R1
    # R.id use pit  R.date cannot be used
    # R1.id range 0-100000
    # R2.id range 10000-200000
