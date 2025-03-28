# Report 

## abstract

Successfully implement the PIT strucutre, for now the program can create a partition index tree by a given list (as partition index list), and we can use another interval for query the PIT to get the overlapping intervals.  

## Introduction

Partition index tree (PIT) is a data structure that can efficiently answer range queries on a list of intervals. It is a red black tree based data structure, where each node contain the following information:

- Interval: the interval represented by the node
- Max: the maximum value of the interval and its subtrees

The Max value is used to reduce the complexity of the range queries, by only considering the intervals that overlap with the query interval.

## Demo

The code demo is in the file `PITDemo.py`. The code can be run by executing the following command in the terminal:
```
python PITDemo.py
```

The output of the program will be:
`[(30, 40), (10, 20), (30, 40), (40, 50)]`

which is the overlapping intervals of the query interval (15, 45) with the given list of intervals.

# Need to do

1. Did not consider the edge cases, such as empty list, list with multiple intervals, and the inteval like `[10, 45)`,etc.
2. Need to consider the case when the interval including the infinity, such as `(10, inf)` or `(inf, 50)`.
3. Need to create a query function to transform a list of tuples `(attr, value)` to the list of intervals, so that we can use the PIT to answer range queries.
   
# Confusion
1. How to apply this PIT to those data structures in postgreSQL?
2. For the link you provided, Some dataset is outdated, so I cannot create the dataset for the PIT, And I have no idea about how to apply the Python code to the PostgreSQL, but just use it to operate the data in the sql, using tools like sqlachemy.