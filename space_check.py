import numpy as np
import scipy.linalg
import math

def get_space_usage_only(A):
    ''' Computes the space needed to compute 'Ax' where 'x' is received one element at a time.

    NOTE: Assumes 'A' indeed can be computed in low space.
        i.e., does not check that intervals line up correctly.
        Also does not account for memory needed to store 'A' itself.

    Parameters:
    A: A square lower-triangular matrix.

    Return:
    Integer representing the number of floats needed to store intermediate computation.
    '''
    return max( len(_get_interval_list_from_row(A[i, :i+1])) for i in range(A.shape[0]) )

def _get_interval_list_from_row(r):
    ''' Returns the intervals on which a row is constant.

    NOTE: Assumes that r[i] <= r[i+1]. (Always true for MRMs)

    Parameters:
    r: A numpy array representing a row. 

    Returns:
    A list of tuples (a, b) where the union of all tuples is a decomposition of [0, r.shape[0] -1]
    '''

    # np.unique does not handle non-contiguous repeating values in the array as liked
    # Therefore, enforce that the values in 'r' are non-decreasing (valid for our matrices)
    assert np.all(r[1:] >= r[:-1]), "Row is not non-decreasing w.r.t. column index"
    _, idx_list = np.unique(r, return_index=True) # No need to sort 'idx_list' under above assumption

    intervals  = [[idx_list[j], idx_list[j+1] - 1] for j in range(len(idx_list) - 1)]
    intervals.append((idx_list[-1], r.shape[0] - 1))
    return intervals

def _check_if_consecutive_rows_align(intervals, old_intervals, row_idx):
    ''' Checks if a row can be computed efficiently given the previous row.

    NOTE: The underlying matrix here and elsewhere is assumed to be lower-triangular,
    meaning that 'intervals' cover [0, row_idx] and 'old_intervals' cover [0, row_idx - 1.]

    Parameters:
    intervals: List of tuples representing the intervals on which the current row
        is constant on.
    old_intervals: List of tuples representing the intervals on which the past row
        is constant on.
    row_idx: The index of the row that 'intervals' represent.

    Return:
    'True' if the computation of the past row can be efficiently re-used for the current row,
        otherwise 'False'.
    '''

    # Check that the rows involved are lower triangular 
    assert row_idx == 0 or old_intervals[-1][-1] == row_idx - 1
    assert intervals[-1][-1] == row_idx
    
    # Check intervals
    old_interval_idx = 0
    interval_idx = 0
    while interval_idx < len(intervals):
        pos, end = intervals[interval_idx]
        interval_idx += 1

        # We've reached the diagonal element
        if pos == row_idx:
            break

        while pos <= end:
            old_start, old_end = old_intervals[old_interval_idx]
            old_interval_idx += 1

            # Check that intervals align
            if pos != old_start or old_end > end:
                return False

            # Move along
            pos = old_end + 1
    
    # If everything checked out, we should have gone through all intervals
    assert row_idx == 0 or old_interval_idx == len(old_intervals)
    assert interval_idx == len(intervals)

    return True

def _is_dyadic_interval(a, b):
    ''' Check if the interval [a, b] is dyadic.

    NOTE: Intervals start from '0'.

    Parameters:
    a: Integer corresponding to the start of the interval.
    b: Integer corresponding to the start of the interval.

    Returns:
    'True' if the interval [a, b] is dyadic, otherwise 'False'.
    '''

    # Check if the size of the interval is a power of 2, and if
    # the left border is a multiple of the size.
    size = b - a + 1
    return (np.binary_repr(size).count('1') == 1) and (a % size == 0)

def verify_efficient_structure(A, verbose=False):
    ''' Checks if a lower-triangular matrix can compute one row given the past row efficiently.

    Close to being "binnable", but does not allow for any entry to appear multiple times for disjoint intervals.

    Parameters:
    A: A square lower-triangular matrix.
    verbose: 'True' to activate more printing.

    Returns:
    The function returns three outputs:
        1. 'True' if the intervals across all rows in the matrix align to allow for space-efficiency.
        2. Integer representing the size of the binning. Same computation as 'get_space_usage_only'.
        3. True if all underlying intervals are dyadic.
    '''
    n = A.shape[0]

    # Things we are checking.
    space_usage = 0
    is_dyadic = True
    intervals_align = True

    old_intervals = None
    row_idx = 0
    while row_idx < n and intervals_align:
        # Get all unique values in A[k, :k+1]
        intervals = _get_interval_list_from_row(A[row_idx, :row_idx+1])
        #print(intervals)
        
        # Check the maximum number of intervals we are storing so far
        space_usage = max(space_usage, len(intervals))

        # Check if all intervals involved are dyadic
        is_dyadic = is_dyadic and all( _is_dyadic_interval(a, b) for a, b in intervals )

        # Check if the intervals align
        intervals_align = _check_if_consecutive_rows_align(intervals, old_intervals, row_idx)

        old_intervals = intervals
        row_idx += 1

    if not intervals_align:
        is_dyadic = False # Meaningless to look at if the intervals do not align
        if verbose:
            print("Intervals do not align!")
        space_usage = n # Assume worst for the space usage

    return intervals_align, space_usage, is_dyadic
