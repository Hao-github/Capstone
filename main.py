import logging

from sqlglot import expressions as exp

from CCPipeline import CCPipeline
from config.config import load_config
from database.TablePartitioner import TablePartitioner
from dataset import label_list, sql_list
from model.SQLOptimizer import SQLOptimizer

logging.basicConfig(
    filename="output.log",  # 指定文件名
    level=logging.INFO,  # 记录 debug 以上的所有日志
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    config = load_config()
    partitioner = TablePartitioner(config)

    metadata = partitioner.get_schema_metadata()

    # # 加载配置
    from sql_for_test_1 import example_sql, l_partition_size, r_partition_size

    so = SQLOptimizer(example_sql, metadata)
    l_column = so.join_expr[0].this.this.this
    r_column = so.join_expr[0].expression.this.this
    l_table = so.alias_dict[so.join_expr[0].this.table]
    r_table = so.alias_dict[so.join_expr[0].expression.table]
    logging.info(f"Partitioning {l_table} table by range...")
    partitioner.partition_table(l_table, l_partition_size, l_column)
    logging.info(f"Partitioning {r_table} table by range...")
    partitioner.partition_table(r_table, r_partition_size, r_column)
    logging.info("finish partitioning")

    """
    1. 把最后的query补出来
    2。 3个数据集分别测出查询优化时间和执行时间：
        pit可用数据集+pit，pit可用数据集+classifer，pit不可用数据集+classifer
    直接union 对比 classifer优化后query 对比 pit优化后query
    多表联合暂时只考虑其中两个表partition联合
    
    3. LaTeX编辑报告 overleaf
    
    zhangwentang
    liguoliang
    Databrik
    MIT AI for Data
    """

    left_1 = so.alias_dict[so.join_expr[0].this.table]
    right_1 = so.alias_dict[so.join_expr[0].expression.table]
    table_map1 = [
        {left_1: left, right_1: right}
        for left in metadata[left_1]["part_table"]
        for right in metadata[right_1]["part_table"]
    ]

    sql_list = []
    label_list = []
    for table_map in table_map1:
        parsed_query_copy = so.parsed_query.copy()
        for table in parsed_query_copy.find_all(exp.Table):
            new_table = exp.Table(
                this=exp.to_identifier(table_map[table.name]),
                db=exp.to_identifier(table.name),
                alias=table.args.get("alias"),
            )
            table.replace(new_table)
        new_sql = parsed_query_copy.sql()
        sql_list.append(new_sql)
        result = partitioner.execute_query(new_sql).fetchone()
        if result is None:
            label_list.append(0)
        else:
            label_list.append(result[0])

    logging.info("finish generating sql list and label list")

    ccp = CCPipeline(partitioner, sql_list, label_list)
    ccp.train()
    logging.info("finish loading data & training (if any)")
    so = SQLOptimizer(example_sql, metadata)
    res = ccp.predict(so, sql_list)
    label_check = [1 if label > 0 else 0 for label in label_list]
    for idx, value in enumerate(res[0]):
        if label_check[idx] != value:
            print(f"{idx},Error: {label_check[idx]} != {value}")
