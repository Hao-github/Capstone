from database.TablePartitioner import TablePartitioner
from config.config import load_config

if __name__ == '__main__':
    config = load_config()
    tp = TablePartitioner(config)
    print("Partitioning orders table by range...")
    # tp.partition_table_2d("orders", 2500000, 300000,"o_orderkey", "o_custkey")
    # tp.partition_table_2d("lineitem", 2500000, 500000, "l_orderkey", "l_partkey")
    tp.partition_table("customer", 200000, "c_custkey")
    # tp.partition_table("part", 100000, "p_partkey")
    # tp.partition_table("partsupp", 100000, "ps_partkey")