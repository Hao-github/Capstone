from config.config import load_config
from database.TablePartitioner import TablePartitioner
from model.SQLOptimizer import SQLOptimizer
from model.CardinalityClassifier import CardinalityClassifier
from model.BloomFilter import BloomFilter

example_sql = """
SELECT
    *
FROM
    orders_13 AS o,
    lineitem_13 AS l
WHERE
    l.l_orderkey = o.o_orderkey
AND
    l.l_linenumber < 6
AND 
    o.o_orderstatus = 'O'
"""
config = load_config()
partitioner = TablePartitioner(config)
cc = CardinalityClassifier(partitioner)


def predict_sql_result(sql: str) -> bool:
    so = SQLOptimizer(sql, partitioner.get_schema_metadata())
    table_feats, join_feats, pred_feats = cc.build_batch([so])
    result = cc.forward(table_feats, join_feats, pred_feats)
    if result > 0.5:
        return True
    ltable, lcolumn, rtable, rcolumn = so.get_condition_from_expr(so.join_expr[0])
    lschema = ltable.split("_")[0]
    rschema = rtable.split("_")[0]
    # use pit
    lhs_pit = so.get_partition_index_tree(lschema, lcolumn)
    rhs_intervals = (
        so.schema_metadata[rschema].query("part_table == @rtable").values.tolist()
    )
    if lhs_pit.query_interval_overlap(rhs_intervals):
        return True
    # use bloom
    rtable_data = partitioner.select_data(rschema, rtable, [rcolumn])
    bf = BloomFilter(len(rtable_data), 0.01)
    for row in rtable_data:
        bf.add(row[0])
    ltable_data = partitioner.select_data(lschema, ltable, [lcolumn])
    for row in ltable_data:
        if row[0] in bf:
            return True
    return False
