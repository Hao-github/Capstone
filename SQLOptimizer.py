from sqlglot import expressions as exp
from sqlglot import parse_one

from database.TablePartitioner import TablePartitioner
from model.PartitionIndexTree import PartitionIndexTree


class SQLOptimizer:
    pit_dict: dict[str, PartitionIndexTree] = {}

    def __init__(self, query: str):
        self.query = query
        self.parsed_query = parse_one(query)
        self.table_dict = {t.name: t for t in self.parsed_query.find_all(exp.Table)}

    def parse_merge_conditions(self) -> list[dict[str, str]]:
        alias_to_table = {
            t.alias_or_name: t.name for t in self.parsed_query.find_all(exp.Table)
        }

        # 找出所有 WHERE 里的 等值条件 (A.col = B.col)
        merge_conditions = []
        where_clause = self.parsed_query.find(exp.Where)
        if where_clause is None:
            return merge_conditions

        for condition in where_clause.find_all(exp.EQ):
            left, right = condition.left, condition.right

            if isinstance(left, exp.Column) and isinstance(right, exp.Column):
                lt = alias_to_table.get(left.table, left.table)
                rt = alias_to_table.get(right.table, right.table)
                lc, rc = left.name, right.name
                if lt > rt:
                    lt, rt, lc, rc = rt, lt, rc, lc
                merge_conditions.append(
                    {
                        "left_table": lt,
                        "left_column": lc,
                        "right_table": rt,
                        "right_column": rc,
                    }
                )
        return sorted(
            merge_conditions, key=lambda x: (x["left_table"], x["left_column"])
        )

    def get_partition_index_tree(
        self, table_name: str, column_name: str, tp: TablePartitioner
    ) -> PartitionIndexTree:
        if pit := self.pit_dict.get(f"{table_name}_{column_name}", None):
            return pit
        intervals = tp.select_intervals(table_name, column_name)
        pit = PartitionIndexTree(intervals)
        self.pit_dict[f"{table_name}_{column_name}"] = pit
        return pit

    def get_pit_query(self, pit_dict: dict[str, str]):
        for name, table in self.table_dict.items():
            if name not in pit_dict.keys():
                continue
            value = pit_dict[name]
            try:
                if isinstance(value, list):
                    union_sql = " UNION ALL ".join(
                        f"SELECT * FROM pit_{name}.{t}" for t in value
                    )
                    table.set("this", exp.Subquery(this=parse_one(union_sql)))
                else:
                    table.set("this", exp.to_identifier(str(value)))
            except Exception as e:
                print(e)
                table.set("this", exp.to_identifier(str(value)))
        return self.parsed_query.sql()


if __name__ == "__main__":
    optimizer = SQLOptimizer("""
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

        """)

    print(optimizer.parse_merge_conditions())
