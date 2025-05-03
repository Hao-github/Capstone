from model.PartitionIndexTree import PartitionIndexTree
from database.DatabaseHandler import DatabaseHandler
from config.config import load_config


# Example usage
if __name__ == "__main__":
    config = load_config()
    dh = DatabaseHandler(config)

    # dh.partition_table_by_count("pittest", "aka_name", 10000)
    keys = dh.select_dataframe(schema_name="pittest", table_name="aka_name_pit")[
        ["start_id", "end_id"]
    ].to_dict(orient="split")["data"]

    pit = PartitionIndexTree()

    for key in keys:
        pit.insert(key)

    print(pit.query_interval_overlap(pit.root, 34560, 60000))
