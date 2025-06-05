from model.PartitionNode import PartitionNode
from typing import Any
from itertools import product


class PartitionIndexTree:
    def __init__(self, intervals: list[tuple]):
        self.nil = PartitionNode.EMPTY_NODE
        self.root = self.nil
        for interval in intervals:
            self.insert(interval)

    def insert(self, key: tuple):
        new_node = PartitionNode(key)

        parent = self.nil
        current = self.root

        # Traverse the tree to find the correct position for the new node
        while current != self.nil:
            parent = current
            if new_node.interval_end <= current.interval_start:
                current.min = min(current.min, new_node.interval_start)
                current = current.left
            elif new_node.interval_start >= current.interval_end:
                current.max = max(current.max, new_node.interval_end)
                current = current.right
            else:
                # The new node overlaps with the current node
                # should not appear
                print("Error: Overlapping intervals not allowed")

        # Insert the new node
        new_node.parent = parent
        if parent == self.nil:  # The tree is empty
            self.root = new_node
        elif new_node.interval_end <= parent.interval_start:
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
        if y.left != self.nil:
            y.left.parent = x
        y.parent = x.parent
        if x.parent == self.nil:
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
        if y.right != self.nil:
            y.right.parent = x
        y.parent = x.parent
        if x.parent == self.nil:
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

    def __update_max(self, node: PartitionNode):
        """
        Update the `max` value of the node and `min` value
        to ensure the max values and the min values in the interval tree are correct.
        """
        if node == self.nil:
            return
        # The max value of node should be the maximum of its right child's max value and its own end value
        node.max = (
            max(node.right.max, node.interval_end)
            if node.right != self.nil
            else node.interval_end
        )
        # The min value of node should be the minimum of its left child's min value and its own start value
        node.min = (
            min(node.left.min, node.interval_start)
            if node.left != self.nil
            else node.interval_start
        )

    def query_interval_overlap(self, interval: tuple[int, int, Any]) -> list:
        """
        Query all intervals that overlap with the interval [start, end).
        """
        start, end, value = interval
        result = self.__query_interval_overlap(self.root, start, end)
        return list(product(value, result))

    def __query_interval_overlap(
        self, current: PartitionNode, start: int, end: int
    ) -> list:
        """
        Query all intervals that overlap with the interval [start, end).
        """
        result = []
        if current == self.nil:
            # Empty subtree
            return result
        if start > end or start > current.max or end < current.min:
            # The interval [start, end) is larger than all intervals in the subtrees
            return result
        if end > current.interval_start and start < current.interval_end:
            # The interval [start, end) overlaps with the current interval
            result.extend(current.value)
        if start < current.interval_start:
            # The interval [start, end) may overlap with the left subtree
            result.extend(self.__query_interval_overlap(current.left, start, end))
        if end >= current.interval_end:
            # The interval [start, end) may overlap with the right subtree
            result.extend(self.__query_interval_overlap(current.right, start, end))

        return result
