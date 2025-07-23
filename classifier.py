from model import CardinalityClassifier, BloomFilter, PartitionIndexTree

class Model:
    def __init__(
        self,
        table_id_dim,
        subtable_id_dim,
        join_key_dim,
        predicate_input_dim,
        n_item,
        false_positive_rate=0.01,
    ):
        self.cardinality_classifier = CardinalityClassifier(
            table_id_dim, subtable_id_dim, join_key_dim, predicate_input_dim
        )
        self.bloom_filter = BloomFilter(n_item, false_positive_rate)
        self.pit = PartitionIndexTree()
