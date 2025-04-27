from typing import Any
from sqlalchemy import create_engine, text


class DatabaseHandler:
    def __init__(self, config: dict[str, Any]):
        print(config)
        config = config["database_config"]
        self.host = config["host"]
        self.port = config["port"]
        self.user = config["user"]
        self.password = config["password"]
        self.database = config["database"]
        self.engine = create_engine(
            f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        )

    def create_table(self, table_name: str, columns: dict[str, Any]):
        columns_str = ", ".join([f"{k} {v}" for k, v in columns.items()])
        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_str})"
        with self.engine.connect() as conn:
            conn.execute(text(query))

    def drop_table(self, table_name: str):
        query = f"DROP TABLE IF EXISTS {table_name}"
        with self.engine.connect() as conn:
            conn.execute(text(query))

    def insert_data(self, table_name: str, data: dict[str, Any]):
        columns = ", ".join(data.keys())
        values = ", ".join([f":{k}" for k in data.keys()])
        query = f"INSERT INTO {table_name} ({columns}) VALUES ({values})"
        with self.engine.connect() as conn:
            conn.execute(text(query), **data)

    def select_data(
        self,
        table_name: str,
        columns: list[str] | None = None,
        condition: str | None = None,
    ):
        columns_list = "*" if columns is None else ", ".join(columns)
        query = f"SELECT {columns_list} FROM {table_name}"
        if condition is not None:
            query += f" WHERE {condition}"
        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            return result.fetchall()


if __name__ == "__main__":
    from config.config import load_config

    config = load_config("config.yaml")
    dh = DatabaseHandler(config)
