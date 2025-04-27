from model.PartitionIndexTree import PartitionIndexTree

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

    print(pit.query_interval_overlap(pit.root, 10, 50))
