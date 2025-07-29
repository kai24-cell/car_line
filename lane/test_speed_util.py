from last_homework import Car_line

def test_filter_list_basic():
    test_lines = [
        [[0, 0, 10, 10]],  # slope = 1
        [[0, 5, 10, 5]],   # slope = 0 (should be filtered out)
        [[5, 5, 5, 10]],   # vertical line (inf slope)
    ]
    result = Car_line.filter_list(test_lines, small_line_thresh=0.5)
    assert len(result) == 2
    assert (0, 0, 10, 10) in result
    assert (5, 5, 5, 10) in result
