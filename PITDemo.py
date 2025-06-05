from config.config import load_config
from SQLOptimizer import SQLOptimizer
from database.TablePartitioner import TablePartitioner


import time
import pandas as pd

# Example usage
config = load_config()
tp = TablePartitioner(config)
test_query = """
SELECT MIN(cn.name) AS movie_company,
       MIN(mi_idx.info) AS rating,
       MIN(t.title) AS western_violent_movie
FROM company_name AS cn,
     company_type AS ct,
     info_type AS it1,
     info_type AS it2,
     keyword AS k,
     kind_type AS kt,
     movie_companies AS mc,
     movie_info AS mi,
     movie_info_idx AS mi_idx,
     movie_keyword AS mk,
     title AS t
WHERE cn.country_code != '[us]'
  AND it1.info = 'countries'
  AND it2.info = 'rating'
  AND k.keyword IN ('murder',
                    'murder-in-title',
                    'blood',
                    'violence')
  AND kt.kind IN ('movie',
                  'episode')
  AND mc.note NOT LIKE '%(USA)%'
  AND mc.note LIKE '%(200%)%'
  AND mi.info IN ('Sweden',
                  'Norway',
                  'Germany',
                  'Denmark',
                  'Swedish',
                  'Danish',
                  'Norwegian',
                  'German',
                  'USA',
                  'American')
  AND mi_idx.info < '8.5'
  AND t.production_year > 2005
  AND kt.id = t.kind_id
  AND t.id = mi.movie_id
  AND t.id = mk.movie_id
  AND t.id = mi_idx.movie_id
  AND t.id = mc.movie_id
  AND mk.movie_id = mi.movie_id
  AND mk.movie_id = mi_idx.movie_id
  AND mk.movie_id = mc.movie_id
  AND mi.movie_id = mi_idx.movie_id
  AND mi.movie_id = mc.movie_id
  AND mc.movie_id = mi_idx.movie_id
  AND k.id = mk.keyword_id
  AND it1.id = mi.info_type_id
  AND it2.id = mi_idx.info_type_id
  AND ct.id = mc.company_type_id
  AND cn.id = mc.company_id;


"""
so = SQLOptimizer(test_query)

# Generate partition index trees and parse hash join conditions
merge_conditions = so.parse_merge_conditions()
result = pd.DataFrame()
for merge_condition in merge_conditions:
    left_table = merge_condition["left_table"]
    left_column = merge_condition["left_column"]
    right_table = merge_condition["right_table"]
    right_column = merge_condition["right_column"]
    if not (tp.schema_exists(left_table) and tp.schema_exists(right_table)):
        continue
    # detact the overlap of partition index trees
    lhs_pit = so.get_partition_index_tree(left_table, left_column, tp)
    rhs_keys = tp.select_intervals(right_table, right_column)
    validate_pairs = [
        pair for key in rhs_keys for pair in lhs_pit.query_interval_overlap(key)
    ]
    df = pd.DataFrame(validate_pairs, columns=[right_table, left_table])
    if result.empty:
        result = df
        continue
    if left_table in result.columns:
        if right_table in result.columns:
            result = result.merge(df, on=[left_table, right_table], how="inner")
        else:
            result = result.merge(df, on=left_table, how="inner")

result_columns = result.columns.tolist()
# cluster parse

result_copy = result.copy()
for col in result_columns:
    remain_columns = [c for c in result_columns if c != col]
    result = result.groupby(remain_columns, as_index=False)[col].apply(
        lambda x: str(list(x))
    )
for col in result_columns:
    result[col] = result[col].apply(lambda x: eval(x))


# use pit for query optimization
start = time.time()

sum_time = 0
cnt = 0
for row in result.to_dict(orient="records"):  # type: ignore
    sql = so.get_pit_query(row)  # type: ignore
    time_taken = tp.measure_merge_time(sql)
    sum_time += time_taken
    cnt += 1
    print(f"Query {cnt}")
    print(f"Time taken for query: {time_taken} seconds")
end = time.time()
print(f"Execution time: {end - start} seconds")
print("sum_time", sum_time)
# sum_time = 59.5s

# benchmark query
result_copy_column = result_copy.columns
another_dict = {}
for column in result_copy_column:
    another_dict[column] = result_copy[column].drop_duplicates().tolist()
benchmark_query = so.get_pit_query(another_dict)
print("benchmark_query", tp.measure_merge_time(benchmark_query))

# time = 5.1s
