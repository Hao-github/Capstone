class PartitionNode:
    def __init__(self, key: tuple, color="RED"):
        # key must be a tuple [start, end)
        self.key: tuple = key
        self.color: str = color  # RED or BLACK
        self.left: PartitionNode = None
        self.right: PartitionNode = None
        self.parent: PartitionNode = None
        self.max = (
            key[1] if key[1] is not None else None
        )  # max represents the end value of the interval


class PartitionIndexTree:
    def __init__(self):
        self.NIL = PartitionNode(key=(None, None), color="BLACK")  # Sentinel NIL node
        self.root = self.NIL

    def insert(self, key: tuple):
        new_node = PartitionNode(key)
        new_node.left = self.NIL
        new_node.right = self.NIL

        parent = None
        current = self.root

        # Traverse the tree to find the correct position for the new node
        while current != self.NIL:
            parent = current
            if new_node.key[1] < current.max:
                current = current.left
            else:
                current.max = max(current.max, new_node.key[1])
                current = current.right

        # Insert the new node
        new_node.parent = parent
        if parent is None:
            self.root = new_node
        elif new_node.key[1] <= parent.key[0]:
            parent.left = new_node
        else:
            parent.right = new_node

        # Set the new node's color to RED and fix the Red-Black tree properties
        new_node.color = "RED"
        self.__fix_insert(new_node)

    def __fix_insert(self, node):
        # Fix the Red-Black tree properties after insertion
        while node != self.root and node.parent.color == "RED":
            if node.parent == node.parent.parent.left:
                uncle = node.parent.parent.right
                if uncle.color == "RED":
                    # Case 1: Uncle is RED
                    node.parent.color = "BLACK"
                    uncle.color = "BLACK"
                    node.parent.parent.color = "RED"
                    node = node.parent.parent
                else:
                    if node == node.parent.right:
                        # Case 2: Current node is a right child
                        node = node.parent
                        self.__left_rotate(node)
                    # Case 3: Current node is a left child
                    node.parent.color = "BLACK"
                    node.parent.parent.color = "RED"
                    self.__right_rotate(node.parent.parent)
            else:
                uncle = node.parent.parent.left
                if uncle.color == "RED":
                    node.parent.color = "BLACK"
                    uncle.color = "BLACK"
                    node.parent.parent.color = "RED"
                    node = node.parent.parent
                else:
                    if node == node.parent.left:
                        # Case 2: Current node is a left child
                        node = node.parent
                        self.__right_rotate(node)
                    # Case 3: Current node is a right child
                    node.parent.color = "BLACK"
                    node.parent.parent.color = "RED"
                    self.__left_rotate(node.parent.parent)

        # Ensure the root is BLACK
        self.root.color = "BLACK"
        # Update the max value of the node
        self.__update_max(node)

    def __left_rotate(self, x: PartitionNode):
        y = x.right
        x.right = y.left
        if y.left != self.NIL:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y
        # Update max values for nodes x and y
        self.__update_max(x)
        self.__update_max(y)

    def __right_rotate(self, x: PartitionNode):
        y = x.left
        x.left = y.right
        if y.right != self.NIL:
            y.right.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y
        # Update max values for nodes x and y
        self.__update_max(x)
        self.__update_max(y)

    def __update_max(self, node):
        """
        Update the `max` value of the node to ensure the max values in the interval tree are correct.
        """
        if node == self.NIL:
            return
        # The max value of node should be the maximum of its right child's max value and its own end value
        node.max = (
            max(node.right.max, node.key[1])
            if node.right.max is not None
            else node.key[1]
        )

    def query_interval_overlap(
        self, current: PartitionNode, start: int, end: int
    ) -> list:
        """
        Query all intervals that overlap with the interval [start, end).
        """
        result = []
        if current == self.NIL:
            # Empty subtree
            return result
        if start > end or start > current.max:
            # The interval [start, end) is larger than all intervals in the subtrees
            return result
        if end >= current.key[0] and start < current.key[1]:
            # The interval [start, end) overlaps with the current interval
            result.append(current.key)
        if start < current.key[0]:
            # The interval [start, end) may overlap with the left subtree
            result.extend(self.query_interval_overlap(current.left, start, end))
        if end >= current.key[1]:
            # The interval [start, end) may overlap with the right subtree
            result.extend(self.query_interval_overlap(current.right, start, end))

        return result


# Example usage
if __name__ == "__main__":
    pit = PartitionIndexTree()
    keys = [
        (0, 10),
        (10, 20),
        (20, 30),
        (30, 40),
        (40, 50),
        (50, 60),
        (60, 70),
        (70, 80),
        (80, 90),
        (90, 100),
    ]
    for key in keys:
        pit.insert(key)

    print(pit.query_interval_overlap(pit.root, 15, 45))
