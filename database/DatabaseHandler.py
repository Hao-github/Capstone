from typing import Any
from sqlalchemy import create_engine, text
import pandas as pd


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

    def create_table(self, schema_name: str, table_name: str, columns: dict[str, Any]):
        columns_str = ", ".join([f"{k} {v}" for k, v in columns.items()])
        query = f"CREATE TABLE IF NOT EXISTS {schema_name}.{table_name} ({columns_str})"
        try:
            with self.engine.connect() as conn:
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
        self, schema_name: str, table_name: str, partition_size: int
    ):
        total_count = self.select_data(schema_name, table_name, columns=["COUNT(*)"])[
            0
        ][0]
        partition_count = (total_count + partition_size - 1) // partition_size

        query = f"""
            SELECT column_name, data_type, character_maximum_length
            FROM information_schema.columns
            WHERE table_name = '{table_name}' AND table_schema = '{schema_name}'
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

        pit_table = f"{table_name}_pit"
        self.drop_table(schema_name, pit_table)
        pit_columns = {
            "part_table": "varchar",
            "start_id": "integer",
            "end_id": "integer",
        }
        self.create_table(schema_name, pit_table, pit_columns)

        for i in range(partition_count):
            part_table_name = f"{table_name}_part_{i}"
            self.drop_table(schema_name, part_table_name)
            self.create_table(schema_name, part_table_name, columns)

            offset = i * partition_size
            insert_query = f"""
                INSERT INTO {schema_name}.{part_table_name}
                SELECT * FROM {schema_name}.{table_name}
                ORDER BY id
                OFFSET {offset} LIMIT {partition_size}
            """
            with self.engine.connect() as conn:
                conn.execute(text(insert_query))
                conn.commit()

            start_id = offset + 1
            end_id = min((i + 1) * partition_size, total_count)
            pit_query = f"""
                INSERT INTO {schema_name}.{pit_table} (part_table, start_id, end_id)
                VALUES ('{part_table_name}', {start_id}, {end_id})
            """
            with self.engine.connect() as conn:
                conn.execute(text(pit_query))
                conn.commit()

        print(
            f"Finished partitioning {schema_name}.{table_name} into {partition_count} parts."
        )


if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from config.config import load_config

    config = load_config()
    db = DatabaseHandler(config)
    db.partition_table_by_count("pittest", "aka_name", 10000)
