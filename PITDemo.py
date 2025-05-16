from config.config import load_config
from database.DatabaseHandler import DatabaseHandler
from model.PartitionIndexTree import PartitionIndexTree

# Example usage
if __name__ == "__main__":
    config = load_config()
    dh = DatabaseHandler(config)

    # create partitioned tables
    dh.partition_table_by_count("public", "aka_name", 10000, "person_id")
    dh.partition_table_by_count("public", "person_info", 31000, "person_id")

    # query all intervals need to be checked for overlap
    lhs_prefix = "partitioned_table_aka_name_person_id"
    rhs_prefix = "partitioned_table_person_info_person_id"
    lhs_keys = dh.select_dataframe(
        schema_name=lhs_prefix, table_name=f"{lhs_prefix}_metadata"
    ).to_dict(orient="split")["data"]
    rhs_keys = dh.select_dataframe(
        schema_name=rhs_prefix, table_name=f"{rhs_prefix}_metadata"
    ).to_dict(orient="split")["data"]

    # build index tree for lhs keys, and query overlap for each key in rhs keys
    lhs_keys_pit = PartitionIndexTree(lhs_keys)
    overlap_keys = []
    for key in rhs_keys:
        for overlap_key in lhs_keys_pit.query_interval_overlap(rhs_keys):
            overlap_keys.append((overlap_key, key))

    # after PIT query, measure the time it cost
    total_time = 0
    for overlap_key_pair in overlap_keys:
        total_time += dh.measure_merge_time(
            lhs_prefix,
            overlap_key_pair[0][0],
            rhs_prefix,
            overlap_key_pair[1][0],
            on="person_id",
        )
    print(
        f"After PIT: Total time for merging {len(overlap_keys)} tables: {total_time:.2f} seconds"
    )

    # without PIT query, measure the time it cost
    total_time_2 = dh.measure_merge_time(
        lhs_schema="public",
        lhs_table="aka_name",
        rhs_schema="public",
        rhs_table="person_info",
        on="person_id",
    )
    print(f"Without PIT: Total time for merge: {total_time_2:.2f} seconds")
