class PartitionNode:
    # Initially set to None; this will be initialized when needed
    EMPTY_NODE: "PartitionNode" = None  # type: ignore

    @classmethod
    def initialize_empty_node(cls):
        if cls.EMPTY_NODE is None:
            cls.EMPTY_NODE = PartitionNode(key=(None, None, None), color="BLACK")

    def __init__(self, key: tuple, color="RED"):
        # Ensure EMPTY_NODE is initialized before usage

        # key must be a tuple [start, end)
        self.key: tuple = key
        self.color: str = color  # RED or BLACK
        self.left: PartitionNode = PartitionNode.EMPTY_NODE
        self.right: PartitionNode = PartitionNode.EMPTY_NODE
        self.parent: PartitionNode = PartitionNode.EMPTY_NODE
        self.max = key[2]  # max represents the end value of the interval
        self.min = key[1]  # min represents the start value of the interval
    @property
    def partition_name(self):
        return self.key[0]

    @property
    def interval_start(self):
        return self.key[1]

    @property
    def interval_end(self):
        return self.key[2]



PartitionNode.initialize_empty_node()
