from database.TablePartitioner import TablePartitioner
from config.config import load_config

if __name__ == '__main__':
    config = load_config()
    tp = TablePartitioner(config)
    print("Partitioning orders table by range...")
    tp.partition_table("orders", 1200000, "o_orderkey")
    tp.partition_table("lineitem", 1200000, "l_orderkey")