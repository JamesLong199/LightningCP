import unittest

def get_next_indices(interval_seq, i, max_parallel=float('inf')):
    # Case 1: i is in interval_seq - standard behavior with max_parallel limit
    if i in interval_seq:
        current_idx = interval_seq.index(i)
        # Check if we're at the last item
        if current_idx == len(interval_seq) - 1:
            return []
        
        next_item = interval_seq[current_idx + 1]
        # Return list truncated to max_parallel
        return list(range(i + 1, min(i + 1 + max_parallel, next_item)))
    
    # Case 2: i is not in interval_seq - find the appropriate interval
    else:
        # Find the enclosing interval
        for idx in range(len(interval_seq) - 1):
            current = interval_seq[idx]
            next_item = interval_seq[idx + 1]
            
            # Check if i falls between two interval points
            if current < i < next_item:
                # Continue from i to the next_item, limited by max_parallel
                return list(range(i, min(i + max_parallel, next_item)))
        
        # If i is after the last interval point, return empty list
        if interval_seq and i > interval_seq[-1]:
            return []
            
        # If i is before the first interval point
        if interval_seq and i < interval_seq[0]:
            return list(range(i, min(i + max_parallel, interval_seq[0])))
            
        return []

class TestGetNextIndices(unittest.TestCase):
    def test_normal_case(self):
        interval_seq = [0, 5, 10, 15]
        self.assertEqual(get_next_indices(interval_seq, 0), [1, 2, 3, 4])
        self.assertEqual(get_next_indices(interval_seq, 5), [6, 7, 8, 9])
        self.assertEqual(get_next_indices(interval_seq, 10), [11, 12, 13, 14])
        self.assertEqual(get_next_indices(interval_seq, 15), [])  # Last element
        
    def test_max_parallel_limit(self):
        interval_seq = [0, 10, 20]
        # Test max_parallel limits the output
        self.assertEqual(get_next_indices(interval_seq, 0, max_parallel=3), [1, 2, 3])
        self.assertEqual(get_next_indices(interval_seq, 0, max_parallel=5), [1, 2, 3, 4, 5])
        self.assertEqual(get_next_indices(interval_seq, 10, max_parallel=2), [11, 12])
        
    def test_continuing_from_non_interval_point(self):
        interval_seq = [0, 10, 20]
        # Test continuing from a point after previous truncated result
        self.assertEqual(get_next_indices(interval_seq, 4), [4, 5, 6, 7, 8, 9])
        self.assertEqual(get_next_indices(interval_seq, 15), [15, 16, 17, 18, 19])
        
    def test_edge_cases(self):
        interval_seq = [5, 10, 15]
        # Test points before first interval
        self.assertEqual(get_next_indices(interval_seq, 2), [2, 3, 4])
        # Test points after last interval
        self.assertEqual(get_next_indices(interval_seq, 16), [])
        # Test with empty interval sequence
        self.assertEqual(get_next_indices([], 5), [])
        
    def test_max_parallel_edge_cases(self):
        interval_seq = [0, 10]
        # Test max_parallel=0
        self.assertEqual(get_next_indices(interval_seq, 0, max_parallel=0), [])
        # Test max_parallel exceeds interval
        self.assertEqual(get_next_indices(interval_seq, 0, max_parallel=15), [1, 2, 3, 4, 5, 6, 7, 8, 9])
        # Test continuing with max_parallel
        self.assertEqual(get_next_indices(interval_seq, 3, max_parallel=2), [3, 4])
        self.assertEqual(get_next_indices(interval_seq, 5, max_parallel=3), [5, 6, 7])

if __name__ == "__main__":
    unittest.main()