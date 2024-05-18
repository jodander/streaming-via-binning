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

def amir_matrix(n, decay, space=None, always_mult=False):

    if space == None:
        space = math.ceil(math.log2(n))

    mat = np.zeros(shape=(n,n), dtype=float)
    entry_val = 1
    for k in range(space):
        np.fill_diagonal(mat[k:,:], entry_val)
        if always_mult or k > space // 2:
            entry_val *= (1.0 - 0.5/(k+1)) * decay
        else:
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

# Crude starting point.
def approx_matrix(A, threshold_f, mode='mean', verbose=False, perform_extra_checks=False):
    ''' Approximate a lower-triangular matrix using our dyadic-interval-based approach.

    NOTE: Modifies A in-place.

    Parameters:
    A: A lower-triangular matrix.
    threshold_f: A function that takes an integer 'i' as input and outputs a threshold 't' where
        if the right-endpoint of a dyadic interval of size '2^i' satisfies, then all values in
        the interval are set to the same with rule derived from 'mode',
    mode: The rule for how all values covered by an interval are to be approximated.
        Options:
        1) 'left' : set to the left-endpoint.
        2) 'right' : set to the right-endpoint.
        3) 'mean' : set to the mean of all entries in the interval.
        Default: 'mean'
    verbose: Activate more printing. Default: 'False'
    perform_extra_checks: If 'True', it verifies that the returned matrix satisfies all criteria
        we expect from the computation. Default: 'False'
    
    Returns:
        The final matrix, and an integer representing the space usage.
    '''

    # Enforce it being a square matrix for now.
    assert A.shape[0] == A.shape[1]
    for i in range(A.shape[0]):
        A[i, :i+1] = approx_row(A[i, :i+1], threshold_f, mode)

    if perform_extra_checks:
        is_aligned, space_usage, intervals_are_dyadic = space_check.verify_efficient_structure(A)
        assert is_aligned, "The rows in the matrix are not aligned to allow for streaming"
        assert intervals_are_dyadic, "While aligned, the intervals are not dyadic."
    else:
        space_usage = space_check.get_space_usage_only(A)

    if verbose:
        print(f"Uses space={space_usage}")
    
    return A, space_usage

def approx_bennett_mm(n, threshold_f, mode='mean', verbose=False, perform_extra_checks=False):
    ''' Approximate matrix mechanism induced by the 'n' by 'n' Bennet matrix.

    Parameters:
    n: Integer representing the dimension of the matrix.
    threshold_f: A function that takes an integer 'i' as input and outputs a threshold 't' where
        if the right-endpoint of a dyadic interval of size '2^i' satisfies, then all values in
        the interval are set to the same with rule derived from 'mode',
    mode: The rule for how all values covered by an interval are to be approximated.
        Options:
        1) 'left' : set to the left-endpoint.
        2) 'right' : set to the right-endpoint.
        3) 'mean' : set to the mean of all entries in the interval.
        Default: 'mean'
    verbose: Activate more printing. Default: 'False'
    perform_extra_checks: If 'True', it verifies that the returned matrix satisfies all criteria
        we expect from the computation. Default: 'False'
    
    Returns:
        Returns matrices L and R s.t. L@R the counting matrix, and an integer representing the space usage.
    '''

    # A = B**2
    # Approximate B as L using threshold_f, then compute the resulting R

    # Approximate 'B' as our 'L'
    L, space_req = approx_matrix(bennett_matrix(n), threshold_f, mode, verbose, perform_extra_checks)

    # Compute the corresponding R
    R = scipy.linalg.inv(L) @ counting_matrix(n)

    return L, R, space_req

def upper_bound_sensitivity(L, sum_N_terms=None):


    n = L.shape[0]
    if sum_N_terms == None:
        sum_N_terms = n

    B = bennett_matrix(n)
    P = L - B
    Q = -np.linalg.inv(B) @ P

    M = sum( np.linalg.matrix_power(Q, k) for k in range(sum_N_terms) )
    print(np.linalg.matrix_power(Q, n-10))

    return np.linalg.norm(M, ord=2)

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
    """ Returns the MSE over all outputs from running the matrix mechansims based on A = LR
    with the Gaussian mechanism and rho=1/2.

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


def approx_row(r, threshold_f, mode='mean'):
    ''' Approximates a row of a matrix.

    NOTE: Changes 'r' in-place.
        Also, assumes 'r' is non-decreasing along its axis.

    Parameters:
    r: The row to be approximated.
    threshold_f: A function that takes an integer 'i' as input and outputs a threshold 't' where
        if the right-endpoint of a dyadic interval of size '2^i' satisfies, then all values in
        the interval are set to the same with rule derived from 'mode',
    mode: The rule for how all values covered by an interval are to be approximated.
        Options:
        1) 'left' : set to the left-endpoint.
        2) 'right' : set to the right-endpoint.
        3) 'mean' : set to the mean of all entries in the interval.
        Default: 'mean'
    
    Returns:
    The approximated row.
    '''

    n = r.shape[0]
    if n == 1:
        return r
    lvl = int(math.floor(math.log2(n)))
    interval_size = int(2 ** lvl)

    i = 0
    while i < n and lvl > 0:
        i_check = i + interval_size - 1
        #print(f"i_check={i_check}, lvl={lvl}, interval_size={interval_size}")

        # Check if the value is small enough to be 'chunked'
        if i_check < n and r[i_check] < threshold_f(lvl):
            # TODO: Consider alternative settings
            if mode == 'left':
                r[i : i_check + 1] = r[i] # set to left end-point
            elif mode == 'right':
                r[i : i_check + 1] = r[i_check] # set to right end-point
            elif mode == 'mean':
                r[i : i_check + 1] = r[i : i_check + 1].mean() # set to mean
            elif mode == 'squared-mean':
                r[i : i_check + 1] = np.sqrt(np.square(r[i : i_check + 1]).mean()) # questionable

            i = i_check + 1
        else: # If not possible, look at smaller intervals.
            lvl -= 1
            interval_size = interval_size // 2
    return r


# E.g. [(10, 10), (9, 8), (7, 4), ..]

# TODO: Consider having to do multiple merges.
# Intervals are in order [(t, t), (t-1, s) ..., (q, 0)]
def merge_intervals(intervals, r, c, s):

    # First, compute the new interval list after a merge
    new_intervals = []
    idx = 0
    # Store exactly the first 's' values
    while idx < s and idx < len(intervals):
        new_intervals.append(intervals[idx])
        idx += 1

    # Greedily merge from then on
    while idx < len(intervals) - 1:
        curr_start, curr_end = intervals[idx]

        # Check the smallest value over the largest in interval
        curr_val = r[curr_end] / r[curr_start]

        # We can now consider merging it.
        merge_idx = None
        next_idx = idx + 1
        next_start, next_end = intervals[next_idx]
        new_val = r[next_end] / r[curr_start]

        # We can consider merging the interval
        while next_idx < len(intervals) and curr_val > c and new_val > c**2:

            merge_idx = next_idx
            curr_val = r[merge_idx] / r[curr_start]

            next_idx += 1

            # abort once we hit the end
            if next_idx == len(intervals):
                break
            next_start, next_end = intervals[next_idx]
            new_val = r[next_end] / r[curr_start]

        if merge_idx is None:
            # Sanity check
            ####
            start, end = intervals[idx]
            assert r[end] / r[start] > c ** 2
            ####

            new_intervals.append(intervals[idx])
            idx += 1
            continue
        else:
            new_intervals.append((intervals[idx][0], intervals[merge_idx][1]))
            idx = merge_idx + 1

    # Make sure not to skip the last interval.
    if len(new_intervals) == 0 or (new_intervals[-1][1] != intervals[-1][1]):
        new_intervals.append(intervals[-1])
    
    return new_intervals

# TODO: Consider having to do multiple merges.
# Intervals are in order [(t, t), (t-1, s) ..., (q, 0)]
#def merge_intervals(intervals, r, c, s):
#
#    # First, compute the new interval list after a merge
#    new_intervals = []
#    idx = 0
#    # Store exactly the first 's' values
#    while idx < s and idx < len(intervals):
#        new_intervals.append(intervals[idx])
#        idx += 1
#
#    # Greedily merge from then on
#    while idx < len(intervals) - 1:
#        curr_start, curr_end = intervals[idx]
#
#        # Check the smallest value over the largest in interval
#        val = r[curr_end] / r[curr_start]
#
#        # If the current interval has a good ratio, continue to the next interval
#        if val < c:
#            # Assert that the interval was not too greedily merged in the past
#            assert val > c ** 2
#            idx += 1
#            new_intervals.append((curr_start, curr_end))
#            continue
#
#        # We can now consider merging it.
#        next_start, next_end = intervals[idx + 1]
#
#        new_val = r[next_end] / r[curr_start]
#        if new_val > c ** 2:
#            # Merge it.
#            new_intervals.append((curr_start, next_end))
#            idx += 2
#            continue
#        
#        # If we got here, the current interval is a candidate for merging, but the merge is too aggressive
#        # Skip merging and continue
#        idx += 1
#        new_intervals.append((curr_start, curr_end))
#
#    # Make sure not to skip the last interval.
#    if len(new_intervals) == 0 or (new_intervals[-1][1] != intervals[-1][1]):
#        new_intervals.append(intervals[-1])
#    
#    return new_intervals

# NOTE:
def _approximation_rule(r, idx_start, idx_end, mode):

    val = None
    if mode == 'left':
        val = r[idx_end] # set to left end-point
    elif mode == 'right':
        val = r[idx_start] # set to right end-point
    elif mode == 'mean':
        val = r[idx_end : idx_start + 1].mean() # set to mean
    elif mode == 'squared-mean':
        val = np.sqrt(np.square(r[idx_end : idx_start + 1]).mean()) # questionable
    return val

def intervals_to_row(intervals, r, mode):
    ''' RETURNS IT IN-PLACE!!! '''
    for start, end in intervals:
        r[end : start + 1] = _approximation_rule(r, start, end, mode)
    
    return r

def approx_matrix_v2(A, c, s):

    space_requirement = 0
    assert A.shape[0] == A.shape[1]
    intervals = []
    for i in range(A.shape[0]):
        # Compute intervals after a merge
        intervals.insert(0, (i, i))
        intervals = merge_intervals(intervals, A[i, :i+1], c, s)
        space_requirement = max(space_requirement, len(intervals))

        # Use the intervals to express the row
        A[i, :i+1] = intervals_to_row(intervals, A[i, :i+1], mode='mean')

    return A, space_requirement

def approx_bennett_mm_v2(n, c, s):
    L, space_requirement = approx_matrix_v2(bennett_matrix(n), c, s)
    # Compute the corresponding R
    R = scipy.linalg.inv(L) @ counting_matrix(n)

    return L, R, space_requirement

def new_approx(n, c):

    B = bennett_matrix(n)

    