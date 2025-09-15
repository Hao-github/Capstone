example_sql = """
    SELECT
        COUNT(*)
    FROM
        partsupp AS ps,
        lineitem AS l
    WHERE
        ps.ps_partkey = l.l_partkey
    AND
        ps.ps_availqty > 2000
    """

test_sql = """
    SELECT
        COUNT(*)
    FROM
        partsupp_6 AS ps,
        lineitem_5 AS l
    WHERE
        ps.ps_partkey = l.l_partkey
    AND
        ps.ps_availqty > 2000
    """.strip()

l_partition_size = 70000
r_partition_size = 90000
