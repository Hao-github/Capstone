example_sql = """
SELECT
    COUNT(*)
FROM
    part AS p,
    partsupp AS ps
WHERE
    p.p_partkey = ps.ps_partkey
AND
    p.p_size > 20
"""

test_sql = """
SELECT
    COUNT(*)
FROM
    part_12 AS p,
    partsupp_14 AS ps
WHERE
    p.p_partkey = ps.ps_partkey
AND
    p.p_size > 20
""".strip()

l_partition_size = 100000
r_partition_size = 70000
