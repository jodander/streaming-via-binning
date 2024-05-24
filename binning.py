import numpy as np
import scipy.linalg
import math

import space_check

def counting_matrix(n):
    ''' Returns the lower-triangular triangular matrix of all 1s.
    
    Parameters:
    n: The resulting matrix is (n, n)

    Returns:
    The counting matrix of dimension 'n' by 'n'.
    '''
    return scipy.linalg.toeplitz([1] * n, [1] + [0] * (n - 1))

def diff_matrix(n):
    ''' Returns the difference matrix.

    NOTE: Equivalent to the inverse of the counting matrix.
    
    Parameters:
    n: The resulting matrix is (n, n)

    Returns:
    The difference matrix of dimension 'n' by 'n'.
    '''
    mat = np.eye(n)
    np.fill_diagonal(mat[1:, :], -1)
    return mat

def bennett_matrix(n):
    ''' Returns the matrix which is the square-root of the counting matrix.
    
    Parameters:
    n: The resulting matrix is (n, n)

    Returns:
    The Bennet matrix of dimension 'n' by 'n'.
    '''
    mat = np.zeros(shape=(n,n), dtype=float)
    entry_val = 1
    for k in range(n):
        np.fill_diagonal(mat[k:,:], entry_val)
        entry_val *= (1.0 - 0.5/(k+1))
    return mat

# NOTE: Much, much slower. But serves as a sanity check.
def _bennett_matrix_via_sqrtm(T):
    return scipy.linalg.sqrtm( counting_matrix(T) )

# Corresponds to f(k) in HUU'23
def bennett_constant_k_old(k):
    ''' Returns the value of B[i, j] = b(i-j) '''
    if k < 0:
        return 0
    output = 1
    for i in range(k):
        output *= (1.0 - 0.5 / (i + 1))
    return output
def bennett_constant_k(k):
    return math.comb(2*k, k) / math.pow(4, k)

def get_variance(L, R):
    """ Returns a vector of the variance from running the matrix mechansims based on A = LR
    with the Gaussian mechanism and rho=1/2.

    Assuming that A=LR is a square matrix of dimension 'n', this algorithm reports the variance
    as an array of length 'n', where the i^th entry is the variance from answering the query 
    corresponding to the the i^th row in A.

    Parameters:
    L : A matrix of dimension [d, b]
    R : A matrix

    Returns:
    An array of floats where each entry is the variance of the corresponding output.
    """
    sensitivity = compute_sensitivity(R)
    return np.square(np.linalg.norm(L, axis=1, ord=2)) * (sensitivity ** 2)

def get_mse(L, R):
    """ Returns the expected mean-squared error over all outputs from running the matrix mechansims based on A = LR
    with the Gaussian mechanism and rho=1/2. Equivalent to 'mean variance'.

    Assuming that A=LR is a square matrix of dimension 'n', this algorithm reports the variance
    as an array of length 'n', where the i^th entry is the variance from answering the query 
    corresponding to the the i^th row in A.

    Parameters:
    L : A decoder matrix of dimension [n, d]
    R : An encoder matrix of dimension [d, n]

    Returns:
    The MSE over all outputs.
    """
    return get_variance(L, R).mean()

def get_max_se(L, R):
    """ Returns the maximum expected square error over all outputs from running the matrix mechansims based on A = LR
    with the Gaussian mechanism and rho=1/2. Equivalent to 'maximum variance'.

    Assuming that A=LR is a square matrix of dimension 'n', this algorithm reports the variance
    as an array of length 'n', where the i^th entry is the variance from answering the query 
    corresponding to the the i^th row in A.

    Parameters:
    L : A decoder matrix of dimension [n, d]
    R : An encoder matrix of dimension [d, n]

    Returns:
    The MSE over all outputs.
    """
    return get_variance(L, R).max()

def compute_sensitivity(R):
    ''' Computes the sensitivity of a matrix mechanism where 'R' is the encoder matrix
    and we are concerned with rho-zCDP.

    Parameters:
    R: An encoder matrix.
    
    Returns:
    A float representing the L2-sensitiviy of the matrix mechanism, equal to the greatest
        L2-norm over all columns of 'R'.
    '''
    return np.linalg.norm(R, axis=0, ord=2).max()

def merge_intervals(intervals, r, c):
    """ Merges intervals in a partition on the line under the condition of Algorithm 1 in the paper.

    Parameters:
    intervals : List of intervals to merge. Intervals[0] is the diagonal singleton.
    r : The corresponding row of the matrix which is undergoing the merge.
    c : The parameter c from Algorithm 1.

    Returns:
    The new merged interval list.
    """

    # Always add the diagonal element
    new_intervals = [intervals[0]]
    interval_idx = 1

    # Greedily merge from then on
    while interval_idx < len(intervals) - 1:
        left_idx, right_idx = intervals[interval_idx]

        # The current fraction
        curr_val = r[left_idx] / r[right_idx + 1]

        # We consider merging it.
        interval_merge_idx = None
        next_interval_idx = interval_idx + 1
        next_left_idx, next_right_idx = intervals[next_interval_idx]
        new_val = r[next_left_idx] / r[right_idx + 1]

        # We can consider merging the interval
        while next_interval_idx < len(intervals) and curr_val > c and new_val >= c**2:

            interval_merge_idx = next_interval_idx
            next_left_idx, next_right_idx = intervals[next_interval_idx]
            curr_val = new_val

            next_interval_idx += 1

            # abort once we hit the end
            if next_interval_idx == len(intervals):
                break

            next_left_idx, next_right_idx = intervals[next_interval_idx]
            new_val = r[next_left_idx] / r[right_idx + 1]

        if interval_merge_idx is None:
            # Sanity check : make sure any interval we have satisfies the condition we enforce
            left_idx, right_idx = intervals[interval_idx]
            assert left_idx == right_idx or r[left_idx] / r[right_idx + 1] >= c ** 2, ""

            new_intervals.append(intervals[interval_idx])
            interval_idx += 1
            continue
        else:
            new_intervals.append((intervals[interval_merge_idx][0], intervals[interval_idx][1]))
            interval_idx = interval_merge_idx + 1

    # Make sure not to skip the last interval in case it did not get merged.
    if len(new_intervals) == 0 or (new_intervals[-1][1] != intervals[-1][1]):
        new_intervals.append(intervals[-1])
    
    return new_intervals

# Approximation rule, always 'mean' by default and is what is used for the paper.
def _approximation_rule(r, idx_start, idx_end, mode='mean'):
    val = None
    if mode == 'left':
        val = r[idx_start] # set to left end-point
    elif mode == 'right':
        val = r[idx_end] # set to right end-point
    elif mode == 'mean':
        val = r[idx_start : idx_end + 1].mean() # set to mean
    elif mode == 'squared-mean':
        val = np.sqrt(np.square(r[idx_start : idx_end + 1]).mean()) # questionable
    return val

def intervals_to_row(intervals, r, mode):
    ''' Updates the matrix row 'r' in-place based on 'mode'. '''
    for start, end in intervals:
        r[start : end + 1] = _approximation_rule(r, start, end, mode)
    return r

# Approximates a lower-triangular matrix based on Algorithm 1 in the paper.
def approx_matrix(A, c, perform_extra_checks=False):
    """ Performs a binning of matrix 'A' using Algorithm 1 with input parameter 'c'. """

    space_requirement = 0
    assert A.shape[0] == A.shape[1]
    intervals = []
    for i in range(A.shape[0]):
        # Compute intervals after a merge
        intervals.insert(0, (i, i))
        intervals = merge_intervals(intervals, A[i, :i+1], c)
        space_requirement = max(space_requirement, len(intervals))

        # Use the intervals to express the row
        A[i, :i+1] = intervals_to_row(intervals, A[i, :i+1], mode='mean')

    if perform_extra_checks:
        is_aligned, space, _ = space_check.verify_efficient_structure(A)
        assert is_aligned, "The rows in the matrix are not aligned to allow for streaming"
        assert space_requirement == space, "Conflicting results for space needed"

    return A, space_requirement

def approx_bennett_mm(n, c, perform_extra_checks=False):
    """ Performs a binning of the Bennett matrix of size 'n' using Algorithm 1 with input parameter 'c'. """

    L, space_requirement = approx_matrix(bennett_matrix(n), c)
    # Compute the corresponding R
    R = scipy.linalg.solve_triangular(L, counting_matrix(n), lower=True)

    return L, R, space_requirement
    