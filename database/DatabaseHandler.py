import re
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text


class DatabaseHandler:
    def __init__(self, config: dict[str, Any]):
        config = config["database_config"]
        self.host = config["host"]
        self.port = config["port"]
        self.user = config["user"]
        self.password = config["password"]
        self.database = config["database"]
        self.engine = create_engine(
            f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        )

    def create_table(
        self,
        schema_name: str,
        table_name: str,
        columns: dict[str, Any],
        drop_if_exists: bool = False,
    ):
        if drop_if_exists:
            self.drop_table(schema_name, table_name)
        columns_str = ", ".join([f"{k} {v}" for k, v in columns.items()])
        create_schema_query = f"CREATE SCHEMA IF NOT EXISTS {schema_name}"
        query = f"CREATE TABLE IF NOT EXISTS {schema_name}.{table_name} ({columns_str})"
        try:
            with self.engine.connect() as conn:
                conn.execute(text(create_schema_query))
                conn.execute(text(query))
                conn.commit()
        except Exception as e:
            print(f"Error while creating table {table_name}: {e}")

    def drop_table(self, schema_name: str, table_name: str):
        query = f"DROP TABLE IF EXISTS {schema_name}.{table_name}"
        with self.engine.connect() as conn:
            conn.execute(text(query))
            conn.commit()

    def insert_data(self, schema_name: str, table_name: str, data: dict[str, Any]):
        columns = ", ".join(data.keys())
        values = ", ".join([f":{k}" for k in data.keys()])
        query = f"INSERT INTO {schema_name}.{table_name} ({columns}) VALUES ({values})"
        with self.engine.connect() as conn:
            conn.execute(text(query), **data)
            conn.commit()

    def select_data(
        self,
        schema_name: str,
        table_name: str,
        columns: list[str] | None = None,
        condition: str | None = None,
    ):
        columns_list = "*" if columns is None else ", ".join(columns)
        query = f"SELECT {columns_list} FROM {schema_name}.{table_name}"
        if condition is not None:
            query += f" WHERE {condition}"
        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            return result.fetchall()

    def select_dataframe(
        self,
        schema_name: str,
        table_name: str,
        columns: list[str] | str = "*",
        condition: str = "1=1",
    ) -> pd.DataFrame:
        return pd.read_sql_query(
            f"SELECT {columns} FROM {schema_name}.{table_name} WHERE {condition}",
            self.engine,
        )

    def partition_table_by_count(
        self,
        source_schema: str,
        source_table: str,
        partition_size: int,
        column: str = "id",
    ):
        total_count = self.select_data(
            source_schema, source_table, columns=["COUNT(*)"]
        )[0][0]
        partition_count = (total_count + partition_size - 1) // partition_size

        query = f"""
            SELECT column_name, data_type, character_maximum_length
            FROM information_schema.columns
            WHERE table_name = '{source_table}' AND table_schema = '{source_schema}'
            ORDER BY ordinal_position
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(query)).fetchall()

        columns = {}
        for col_name, data_type, char_len in result:
            if char_len is not None:
                columns[col_name] = f"{data_type}({char_len})"
            else:
                columns[col_name] = data_type

        pit_columns = {
            "part_table": "varchar",
            "start_id": "integer",
            "end_id": "integer",
        }
        prefix = f"partitioned_table_{source_table}_{column}"
        pit_schema = prefix
        pit_metadata_table = f"{prefix}_metadata"
        self.create_table(
            pit_schema, pit_metadata_table, pit_columns, drop_if_exists=True
        )

        for i in range(partition_count):
            pit_partition_table = f"{prefix}_{i}"
            self.create_table(
                pit_schema, pit_partition_table, columns, drop_if_exists=True
            )

            offset = i * partition_size
            insert_query = f"""
                INSERT INTO {pit_schema}.{pit_partition_table}
                SELECT * FROM {source_schema}.{source_table}
                ORDER BY {column}
                OFFSET {offset} LIMIT {partition_size}
            """
            with self.engine.connect() as conn:
                conn.execute(text(insert_query))
                conn.commit()

            start_id = offset + 1
            end_id = min((i + 1) * partition_size, total_count)
            pit_query = f"""
                INSERT INTO {pit_schema}.{pit_metadata_table} (part_table, start_id, end_id)
                VALUES ('{pit_partition_table}', {start_id}, {end_id})
            """
            with self.engine.connect() as conn:
                conn.execute(text(pit_query))
                conn.commit()

        print(
            f"Finished partitioning {source_schema}.{source_table} into {pit_schema}, total {total_count} rows, {partition_count} parts."
        )

    def measure_merge_time(
        self,
        lhs_schema: str,
        lhs_table: str,
        rhs_schema: str,
        rhs_table: str,
        on: str,
    ) -> float:
        query = f"""
            EXPLAIN ANALYZE
            SELECT * FROM {lhs_schema}.{lhs_table} lhs
            INNER JOIN {rhs_schema}.{rhs_table} rhs ON lhs.{on} = rhs.{on}
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(query)).fetchall()

            # 从最后一行找 Execution Time
            last_line = result[-1][0]
            match = re.search(r"Execution Time: ([\d.]+) ms", last_line)
            if match:
                return float(match.group(1)) / 1000  # 转成秒
            else:
                raise ValueError("Execution Time not found in EXPLAIN ANALYZE output.")


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from config.config import load_config

    config = load_config()
    db = DatabaseHandler(config)
    # db.partition_table_by_count("public", "aka_name", 10000, "person_id")
    db.partition_table_by_count("public", "person_info", 31000, "person_id")
