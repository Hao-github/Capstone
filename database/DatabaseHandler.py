from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
import time


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

    def get_schema_metadata(self) -> dict[str, pd.DataFrame]:
        query = "SELECT nspname FROM pg_namespace"
        result = self._execute_query(query).fetchall()
        pit_schemas = set(row[0] for row in result)
        schema_cache: dict[str, pd.DataFrame] = {}
        for schema in pit_schemas:
            try:
                schema_cache[schema] = pd.read_sql_query(
                    f"SELECT * FROM {schema}.{schema}_metadata", self.engine
                )
            except Exception:
                continue
        return schema_cache

    def drop_schema(self, schema_name: str):
        query = f"DROP SCHEMA IF EXISTS {schema_name} CASCADE"
        self._execute_query(query)

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
            self._execute_query(create_schema_query)
            self._execute_query(query)
        except Exception as e:
            print(f"Error while creating table {table_name}: {e}")

    def drop_table(self, schema_name: str, table_name: str):
        query = f"DROP TABLE IF EXISTS {schema_name}.{table_name}"
        self._execute_query(query)

    def insert_data(self, schema_name: str, table_name: str, data: dict[str, Any]):
        columns = ", ".join(data.keys())
        values = ", ".join([f":{k}" for k in data.keys()])
        query = f"INSERT INTO {schema_name}.{table_name} ({columns}) VALUES ({values})"
        self._execute_query(query, **data)

    def select_data(
        self,
        schema_name: str,
        table_name: str,
        columns: list[str] = [],
        condition: str = "1 = 1",
    ):
        columns_list = "*" if not columns else ", ".join(columns)
        query = (
            f"SELECT {columns_list} FROM {schema_name}.{table_name} WHERE {condition}"
        )
        return self._execute_query(query).fetchall()

    def measure_merge_time(self, query: str) -> float:
        start_time = time.perf_counter()
        self._execute_query(query).fetchall()
        end_time = time.perf_counter()
        return end_time - start_time

    def _execute_query(self, query: str, **kwargs):
        with self.engine.connect() as conn:
            ret = conn.execute(text(query), **kwargs)
            conn.commit()
            return ret


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from config.config import load_config

    config = load_config()
    db = DatabaseHandler(config)
    # explain analysis
    # baseline: 60s sql_with_hints: 400s
    # run
    # baseline: 240s sql_with_hints: 4s
