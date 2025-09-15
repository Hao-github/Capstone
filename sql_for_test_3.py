example_sql = """
    SELECT
        *
    FROM
        orders AS o,
        lineitem AS l
    WHERE
        l.l_orderkey = o.o_orderkey
    AND
        l.l_linenumber < 6
    AND
        o.o_orderstatus = 'O'
    """

test_sql = """
    SELECT
        *
    FROM
        orders_12 AS o,
        lineitem_14 AS l
    WHERE
        l.l_orderkey = o.o_orderkey
    AND
        l.l_linenumber < 6
    AND
        o.o_orderstatus = 'O'
    """.strip()
    
l_partition_size = 2500000
r_partition_size = 1500000
