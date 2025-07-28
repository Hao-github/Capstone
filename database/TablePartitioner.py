from database.DatabaseHandler import DatabaseHandler
from sqlalchemy import text
from math import ceil


class TablePartitioner(DatabaseHandler):
    def get_partition_ranges(self, min_value, max_value, partition_size):
        start = (min_value // partition_size) * partition_size
        end = ceil(max_value / partition_size) * partition_size
        return [(s, s + partition_size) for s in range(start, end, partition_size)]

    def get_table_columns(self, source_table):
        query = f"""
            SELECT column_name, data_type, character_maximum_length
            FROM information_schema.columns
            WHERE table_name = '{source_table}' AND table_schema = 'public'
            ORDER BY ordinal_position
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(query)).fetchall()

        columns = {}
        for col_name, data_type, char_len in result:
            columns[col_name] = f"{data_type}({char_len})" if char_len else data_type
        return columns

    def partition_table(self, source_table: str, partition_size: int, column: str):
        min_value, max_value = self.select_data(
            "public", source_table, columns=[f"MIN({column})", f"MAX({column})"]
        )[0]

        ranges = self.get_partition_ranges(min_value, max_value, partition_size)
        columns = self.get_table_columns(source_table)

        prefix = f"{source_table}"
        pit_schema = prefix
        pit_metadata_table = f"{prefix}_metadata"
        self.drop_schema(pit_schema)
        pit_columns = {
            "part_table": "varchar",
            f"{column}_start": "integer",
            f"{column}_end": "integer",
        }
        self.create_table(pit_schema, pit_metadata_table, pit_columns)

        for i, (start_id, end_id) in enumerate(ranges):
            part_table = f"{prefix}_{i}"
            self.create_table(pit_schema, part_table, columns)

            insert_query = f"""
                INSERT INTO {pit_schema}.{part_table}
                SELECT * FROM public.{source_table}
                WHERE {column} >= {start_id} AND {column} < {end_id}
            """
            self._execute_query(insert_query)

            pit_query = f"""
                INSERT INTO {pit_schema}.{pit_metadata_table}
                (part_table, {column}_start, {column}_end)
                VALUES ('{part_table}', {start_id}, {end_id})
            """
            self._execute_query(pit_query)

        print(
            f"Finished partitioning public.{source_table} into {pit_schema}, total {len(ranges)} partitions."
        )

    def partition_table_by_column_nums(
        self, source_table: str, partition_size: int, column: str
    ):
        column_count = self.select_data("public", source_table, columns=["COUNT(*)"])[
            0
        ][0]
        ranges = self.get_partition_ranges(0, column_count, partition_size)
        columns = self.get_table_columns(source_table)

        prefix = f"{source_table}"
        pit_schema = prefix
        pit_metadata_table = f"{prefix}_metadata"
        self.drop_schema(pit_schema)
        pit_columns = {
            "part_table": "varchar",
            f"{column}_start": "integer",
            f"{column}_end": "integer",
        }
        self.create_table(pit_schema, pit_metadata_table, pit_columns)

        for i, (start_id, end_id) in enumerate(ranges):
            part_table = f"{prefix}_{i}"
            self.create_table(pit_schema, part_table, columns)

            insert_query = f"""
                INSERT INTO {pit_schema}.{part_table}
                SELECT * FROM public.{source_table}
                ORDER BY {column}
                LIMIT {end_id - start_id} OFFSET {start_id}
            """
            self._execute_query(insert_query)
            min_id, max_id = self.select_data(
                pit_schema, part_table, columns=[f"MIN({column})", f"MAX({column})"]
            )[0]

            pit_query = f"""
                INSERT INTO {pit_schema}.{pit_metadata_table}
                (part_table, {column}_start, {column}_end)
                VALUES ('{part_table}', {min_id}, {max_id})
            """
            self._execute_query(pit_query)

        print(
            f"Finished partitioning public.{source_table} into {pit_schema}, total {len(ranges)} partitions."
        )

    def partition_table_2d(
        self,
        source_table: str,
        partition_size_1: int,
        partition_size_2: int,
        column_1: str,
        column_2: str,
    ):
        (min_1, max_1), (min_2, max_2) = [
            self.select_data(
                "public", source_table, columns=[f"MIN({col})", f"MAX({col})"]
            )[0]
            for col in (column_1, column_2)
        ]

        ranges_1 = self.get_partition_ranges(min_1, max_1, partition_size_1)
        ranges_2 = self.get_partition_ranges(min_2, max_2, partition_size_2)
        columns = self.get_table_columns(source_table)

        prefix = f"{source_table}"
        pit_schema = prefix
        pit_metadata_table = f"{prefix}_metadata"
        self.drop_schema(pit_schema)
        pit_columns = {
            "part_table": "varchar",
            f"{column_1}_start": "integer",
            f"{column_1}_end": "integer",
            f"{column_2}_start": "integer",
            f"{column_2}_end": "integer",
        }
        self.create_table(pit_schema, pit_metadata_table, pit_columns)

        for i, (start_1, end_1) in enumerate(ranges_1):
            for j, (start_2, end_2) in enumerate(ranges_2):
                part_table = f"{prefix}_{i}_{j}"
                self.create_table(pit_schema, part_table, columns)

                insert_query = f"""
                    INSERT INTO {pit_schema}.{part_table}
                    SELECT * FROM public.{source_table}
                    WHERE {column_1} >= {start_1} AND {column_1} < {end_1}
                    AND {column_2} >= {start_2} AND {column_2} < {end_2}
                """
                self._execute_query(insert_query)

                pit_query = f"""
                    INSERT INTO {pit_schema}.{pit_metadata_table}
                    (part_table, {column_1}_start, {column_1}_end, {column_2}_start, {column_2}_end)
                    VALUES ('{part_table}', {start_1}, {end_1}, {start_2}, {end_2})
                """
                self._execute_query(pit_query)

        total_parts = len(ranges_1) * len(ranges_2)
        print(
            f"Finished 2D partitioning public.{source_table} into {pit_schema}, total {total_parts} partitions."
        )