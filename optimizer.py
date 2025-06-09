import re
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional


# def extract_join_graph_from_sql(sql):
#     pattern = r"(\w+)\.\w+\s+=\s+(\w+)\.\w+"
#     matches = re.findall(pattern, sql)

#     graph = {}
#     for table1, table2 in matches:
#         if table1 not in graph:
#             graph[table1] = []
#         if table2 not in graph:
#             graph[table2] = []

#         if table2 not in graph[table1]:
#             graph[table1].append(table2)
#         if table1 not in graph[table2]:
#             graph[table2].append(table1)

#     return graph

def extract_join_graph_from_sql(sql: str):
    """
    解析 SQL，输出以逻辑表名为 key 的 join graph。
    如：{ 'movie_companies': ['title', 'movie_info_idx'] }
    """
    # 1. 构建别名 -> 逻辑表名映射
    alias_map = {}
    from_match = re.search(r'FROM\s+(.+?)\s+WHERE', sql, re.IGNORECASE | re.DOTALL)
    if from_match:
        from_clause = from_match.group(1)
        entries = [x.strip() for x in from_clause.split(',')]
        for entry in entries:
            match = re.match(r'(\w+)\s+(?:AS\s+)?(\w+)', entry, re.IGNORECASE)
            if match:
                table_name, alias = match.group(1), match.group(2)
                alias_map[alias] = table_name

    # 2. 提取等值连接：别名.key = 别名.key
    pattern = r"(\w+)\.\w+\s*=\s*(\w+)\.\w+"
    matches = re.findall(pattern, sql)

    # 3. 构建逻辑表 join graph
    graph = {}
    for alias1, alias2 in matches:
        table1 = alias_map.get(alias1)
        table2 = alias_map.get(alias2)
        if not table1 or not table2:
            continue
        if table1 not in graph:
            graph[table1] = []
        if table2 not in graph:
            graph[table2] = []
        if table2 not in graph[table1]:
            graph[table1].append(table2)
        if table1 not in graph[table2]:
            graph[table2].append(table1)

    return graph


def enumerate_join_orders(graph: dict) -> list[list[str]]:
    results = []

    def dfs(path, visited):
        if len(path) == len(graph):
            results.append(path[:])
            return
        last = path[-1]
        for neighbor in graph[last]:
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                dfs(path, visited)
                path.pop()
                visited.remove(neighbor)

    for start in graph:
        dfs([start], {start})
    return results

def match_partitions_via_pit(left: str, right: str, sub_tables: dict) -> dict:
    """
    构造 left 表和 right 表下所有子表的配对，返回配对 dict。

    输入：
        left: 左表名称，如 'R'
        right: 右表名称，如 'S'
        sub_tables: dict，如 {'R': ['R1', 'R2'], 'S': ['S12', 'S13']}

    输出：
        dict[(str, str)] -> True
        如 {('R1', 'S12'): True, ('R2', 'S13'): True, ...}
    """
    pairs = {}
    left_subs = sub_tables.get(left, [])
    right_subs = sub_tables.get(right, [])

    for l in left_subs:
        for r in right_subs:
            pairs[(l, r)] = True  # 后续可以改为实际的 PIT 计算
    return pairs

def cluster_partition_pairs(pairs: Dict[Tuple[str, str], bool]) -> List[Tuple[List[str], List[str]]]:
    """
    输入：
        pairs: dict 格式，{(left_partition, right_partition): True, ...}

    输出：
        聚合后的 cluster 列表 [(left_group, right_group), ...]
        例如：[(['R1'], ['S12', 'S13']), (['R2'], ['S22', 'S23'])]
    """
    cluster_map = defaultdict(lambda: (set(), set()))

    for (l, r), ok in pairs.items():
        if not ok:
            continue
        cluster_map[l][0].add(l)
        cluster_map[l][1].add(r)

    clusters = []
    for lefts, rights in cluster_map.values():
        clusters.append((list(lefts), list(rights)))
    return clusters


def build_cluster_graph(graph, sub_tables):
    cluster_map = {}
    for t1 in graph:
        for t2 in graph[t1]:
            if t1 < t2:
                pairs = match_partitions_via_pit(t1, t2, sub_tables)
                clusters = cluster_partition_pairs(pairs)
                cluster_map[(t1, t2)] = clusters
    return cluster_map

def resolve_join_union_order(
    join_order: List[str],
    cluster_map: Dict[Tuple[str, str], List[Tuple[List[str], List[str]]]]
) -> Dict[str, Any]:
    """
    返回 join-union 顺序结构，仅包含 join 关系和子表聚合结构。    
    输出示例：
    [
        {
            "join": ("R", "S"),
            "pairs": [
                {"left": ["R1"], "right": ["S12", "S13"]},
                ...
            ]
        },
        ...
    ]
    """
    plan_steps = []
    
    for i in range(len(join_order) - 1):
        t1, t2 = join_order[i], join_order[i + 1]
        key = (t1, t2) if (t1, t2) in cluster_map else (t2, t1)
        clusters = cluster_map.get(key, [])
        
        pair_list = []
        for lefts, rights in clusters:
            if key == (t1, t2):
                pair_list.append({"left": lefts, "right": rights})
            else:
                pair_list.append({"left": rights, "right": lefts})
        
        plan_steps.append({
            "join": (t1, t2),
            "pairs": pair_list
        })

    return plan_steps


def generate_baseline_sql(base_sql: str, sub_tables: Dict[str, List[str]], join_order: Optional[List[str]] = None) -> str:
    """
    将原始 SQL 改写为子表 UNION ALL 聚合 + Leading Hint 的 baseline 形式，保留表别名。

    参数:
        base_sql: 原始 SQL，例如 SELECT * FROM A AS a, B AS b WHERE ...
        sub_tables: 映射表逻辑名到子表名列表，例如 {'A': ['A1', 'A2']}
        join_order: 使用别名构成的 JOIN 顺序，例如 ['a', 'b']
    
    返回:
        改写后的 SQL 字符串，包含 Hint 和子表 UNION ALL 聚合
    """

    # 1. 构造逻辑表 -> 子表 UNION ALL 块（延后加别名）
    union_blocks = {}
    for table, parts in sub_tables.items():
        union_sql = "\nUNION ALL\n".join([f"SELECT * FROM {p}" for p in parts])
        union_blocks[table.lower()] = f"( {union_sql} )"

    # 2. 提取 FROM 中的原始表和别名
    from_match = re.search(r'FROM\s+(.+?)\s+WHERE', base_sql, re.IGNORECASE | re.DOTALL)
    if not from_match:
        return "-- Error: Could not parse FROM clause."
    
    from_clause = from_match.group(1)
    from_entries = [entry.strip() for entry in from_clause.split(",")]

    alias_map = {}  # alias -> logical table
    rewritten_from = []

    for entry in from_entries:
        match = re.match(r'(\w+)\s+(?:AS\s+)?(\w+)', entry, re.IGNORECASE)
        if not match:
            return f"-- Error parsing FROM entry: {entry}"
        table, alias = match.group(1).lower(), match.group(2)
        alias_map[alias] = table
        if table in union_blocks:
            rewritten_from.append(f"{union_blocks[table]} AS {alias}")
        else:
            rewritten_from.append(entry)

    rewritten_from_clause = ",\n     ".join(rewritten_from)

    # 3. 替换 FROM 子句
    rewritten_sql = re.sub(
        r'FROM\s+(.+?)\s+WHERE',
        f'FROM {rewritten_from_clause} WHERE',
        base_sql,
        flags=re.IGNORECASE | re.DOTALL
    )

    # 4. 插入 Hint（使用别名）
    hint = f"/*+ Leading({ ' '.join(join_order) }) */" if join_order else ""
    final_sql = f"{hint}\n{rewritten_sql.strip()}" if hint else rewritten_sql.strip()

    return final_sql


def generate_recursive_cluster_sql(
    base_sql: str,
    join_union_order: List[Dict[str, Any]],
    join_keys: Dict[Tuple[str, str], Tuple[str, str]] = None,
    filters: Dict[str, str] = None
) -> str:
    """
    递归构造聚合式 SQL
    """

    join_keys = join_keys or {}
    filters = filters or {}

    # 提取全局 WHERE 子句（用于 fallback）
    where_clause = ""
    where_match = re.search(r'WHERE\s+(.+)', base_sql, re.IGNORECASE | re.DOTALL)
    if where_match:
        where_clause = "WHERE " + where_match.group(1).strip()

    current_sql_blocks = []

    for step_idx, step in enumerate(join_union_order):
        join_pair_sql = []
        for pair in step["pairs"]:
            lefts = pair["left"]
            rights = pair["right"]
            on_keys = pair.get("on", ("key", "key"))
            pair_filters = pair.get("filter", {})

            def build_select_block(tbls):
                selects = []
                for t in tbls:
                    cond = filters.get(t, "") or pair_filters.get(t, "")
                    selects.append(f"SELECT * FROM {t}" + (f" WHERE {cond}" if cond else ""))
                return selects[0] if len(selects) == 1 else f"(\n" + "\nUNION ALL\n".join(selects) + "\n)"

            left_block = build_select_block(lefts)
            right_block = build_select_block(rights)
            left_col, right_col = on_keys

            join_sql = f"{left_block} JOIN {right_block} ON {lefts[0]}.{left_col} = {rights[0]}.{right_col}"
            join_pair_sql.append(f"({join_sql})")

        step_union_sql = "\nUNION ALL\n".join(join_pair_sql)

        if step_idx == 0:
            current_sql_blocks.append(f"({step_union_sql})")
        else:
            current_sql_blocks = [
                f"({prev_block} JOIN ({step_union_sql}) ON 1=1)"
                for prev_block in current_sql_blocks
            ]

    final_sql = f"SELECT * FROM {current_sql_blocks[0]}"
    if where_clause:
        final_sql += f"\n{where_clause}"
    return final_sql


def extract_join_keys_from_sql(sql: str) -> Dict[Tuple[str, str], Tuple[str, str]]:
    """
    提取 SQL 中的 join 键，返回 {(table1, table2): (col1, col2)}
    仅处理等值连接 a.col = b.col
    """
    pattern = r"(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)"
    matches = re.findall(pattern, sql)
    result = {}
    for t1, c1, t2, c2 in matches:
        key = tuple(sorted([t1, t2]))
        if key[0] == t1:
            result[(t1, t2)] = (c1, c2)
        else:
            result[(t2, t1)] = (c2, c1)
    return result

def extract_filters_from_sql(sql: str) -> Dict[str, str]:
    """
    粗略提取 WHERE 中的 filter 子句，返回 {alias: condition_string}
    例如 { 'mc': "note LIKE ... AND ..." }
    """
    where_clause = ""
    where_match = re.search(r'WHERE\s+(.+)', sql, re.IGNORECASE | re.DOTALL)
    if where_match:
        where_clause = where_match.group(1).strip()

    conditions = re.split(r'\s+(AND|OR)\s+', where_clause)

    filter_map = defaultdict(list)
    for cond in conditions:
        m = re.match(r"(\w+)\.", cond)
        if m:
            alias = m.group(1)
            filter_map[alias].append(cond.strip())

    return {k: " AND ".join(v) for k, v in filter_map.items()}


def optimizer(sql: str, sub_tables: dict):
    #print(generate_baseline_sql(sql, sub_tables))
    graph = extract_join_graph_from_sql(sql)
    print(graph)
    join_orders = enumerate_join_orders(graph)
    print(join_orders)
    cluster_map = build_cluster_graph(graph, sub_tables)
    print(cluster_map)
    join_keys = extract_join_keys_from_sql(sql)
    filters = extract_filters_from_sql(sql)
    sql_with_hints = []
    print(len(join_orders))
    for order in join_orders:
        join_union_order = resolve_join_union_order(order, cluster_map)
        final_sql = generate_recursive_cluster_sql(sql, join_union_order)
        print(final_sql)
        sql_with_hints.append(final_sql)

example_sql = """
SELECT MIN(mc.note) AS production_note,
       MIN(t.title) AS movie_title,
       MIN(t.production_year) AS movie_year
FROM company_type AS ct,
     info_type AS it,
     movie_companies AS mc,
     movie_info_idx AS mi_idx,
     title AS t
WHERE ct.kind = 'production companies'
  AND it.info = 'top 250 rank'
  AND mc.note NOT LIKE '%(as Metro-Goldwyn-Mayer Pictures)%'
  AND (mc.note LIKE '%(co-production)%'
       OR mc.note LIKE '%(presents)%')
  AND ct.id = mc.company_type_id
  AND t.id = mc.movie_id
  AND t.id = mi_idx.movie_id
  AND mc.movie_id = mi_idx.movie_id
  AND it.id = mi_idx.info_type_id;
"""

# 构造模拟子表映射（sub_tables），每张表模拟几个分区
example_sub_tables = {
    'company_type': ['ct1', 'ct2'],
    'info_type': ['it1'],
    'movie_companies': ['mc1', 'mc2', 'mc3'],
    'movie_info_idx': ['mi_idx1', 'mi_idx2'],
    'title': ['t1', 't2']
}



optimizer(example_sql, example_sub_tables)

