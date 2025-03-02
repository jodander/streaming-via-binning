import math

import numpy as np
import scipy.linalg

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

def bennett_coefficients(n):
    ''' Computes the entries c of the Bennett matrix where c[k] = (2k choose k) / 4^k

    Parameters:
    n: Number coefficients to compute.

    Returns:
    An n-dimensional array storing the subdiagonals of the Bennett matrix.
    '''
    c = np.zeros(n)
    c[0] = 1
    for k in range(1, n):
        c[k] = (1.0 - 0.5/k) * c[k-1]
    return c

def bennett_matrix(n):
    ''' Returns the matrix which is the square-root of the counting matrix.
    
    Parameters:
    n: The resulting matrix is (n, n)

    Returns:
    The Bennett matrix of dimension 'n' by 'n'.
    '''
    return scipy.linalg.toeplitz( bennett_coefficients(n), [1] + [0] * (n-1) )

# NOTE: Much, much slower. But serves as a sanity check.
def _bennett_matrix_via_sqrtm(T):
    return scipy.linalg.sqrtm( counting_matrix(T) )

def bennett_coefficient(k):
    ''' Returns the value of the 'k'th subdiagonal of the Bennett matrix'''
    return math.comb(2*k, k) / math.pow(4, k)

def get_variance(L, R):
    """ Returns a vector of the variance from running the matrix mechansims based on A = LR
    with the Gaussian mechanism and rho=1/2.

    Assuming that A=LR is a square matrix of dimension 'n', this algorithm reports the variance
    as an array of length 'n', where the i^th entry is the variance from answering the query 
    corresponding to the the i^th row in A.

    Parameters:
    L : A decoder matrix of dimension [n, d]
    R : An encoder matrix of dimension [d, n]

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
    The MaxSE over all outputs.
    """
    return get_variance(L, R).max()

def compute_sensitivity(R):
    ''' Computes the L2-sensitivity of a matrix mechanism where 'R' is the encoder matrix
    and we are dealing with an L1 neighboring relation.

    Parameters:
    R: An encoder matrix.
    
    Returns:
    A float representing the L2-sensitiviy of the matrix mechanism, equal to the greatest
        L2-norm over all columns of 'R'.
    '''
    return np.linalg.norm(R, axis=0, ord=2).max()

def merge_intervals(intervals, r, c, tau):
    """ Merges intervals in a partition under specific conditions.

    Corresponds to merging intervals for one row in Algorithm 1.

    Parameters:
    intervals : List of intervals to merge; intervals[0] is the diagonal singleton.
    r : Row of the matrix undergoing merge.
    c : Condition parameter from Algorithm 1.
    tau : Threshold parameter from Algorithm 1.

    Returns:
    List of merged intervals.
    """
    
    # Start with the diagonal element
    merged_intervals = [intervals[0]]
    idx = 1  # Index for interval iteration

    while idx < len(intervals) - 1:
        left, right = intervals[idx]

        # Check if merging is feasible based on conditions
        if r[right + 1] == 0 or r[right] < tau:
            merge_idx = len(intervals) - 1  # Set to last if tau condition met
            next_idx = len(intervals)
        else:
            # Calculate initial merge potential
            curr_val = r[left] / r[right + 1]
            merge_idx = None
            next_idx = idx + 1
            next_left, next_right = intervals[next_idx]
            next_val = r[next_left] / r[right + 1]

        # Continue merging as long as conditions hold
        while next_idx < len(intervals) and curr_val > c and next_val >= c**2:
            if r[next_left] < tau:
                merge_idx = len(intervals) - 1  # Merge to end if threshold met
                break

            # Update merge index and values
            merge_idx = next_idx
            curr_val = next_val

            next_idx += 1
            if next_idx == len(intervals):  # Exit if at end
                break

            next_left, next_right = intervals[next_idx]
            next_val = r[next_left] / r[right + 1]

        # Add interval to merged list if it didn't meet merge conditions
        if merge_idx is None:
            assert left == right or r[left] / r[right + 1] >= c ** 2  # Condition check
            merged_intervals.append(intervals[idx])
            idx += 1
        else:
            # Merge selected interval
            merged_intervals.append((intervals[merge_idx][0], intervals[idx][1]))
            idx = merge_idx + 1

    # Ensure last interval is added if unmerged
    if merged_intervals[-1][0] != intervals[-1][0]:
        merged_intervals.append(intervals[-1])

    return merged_intervals

# Approximation rule, 'endpoints_mean' is used for the plots in the paper.
def _approximation_rule(r, idx_start, idx_end, rule):
    val = None
    if rule == 'left':
        val = r[idx_start] # set to left end-point
    elif rule == 'right':
        val = r[idx_end] # set to right end-point
    elif rule == 'mean':
        val = r[idx_start : idx_end + 1].mean() # set to mean
    elif rule == 'endpoints_mean':
        val = 0.5 * (r[idx_start] + r[idx_end]) # set to mean of end points
    return val

def intervals_to_row(intervals, r, rule):
    ''' Updates the matrix row 'r' in-place based on 'mode'. '''
    for start, end in intervals:
        r[start : end + 1] = _approximation_rule(r, start, end, rule)
    return r

# Approximates a lower-triangular matrix based on Algorithm 1 in the paper.
def approx_matrix(A, c, tau, perform_extra_checks=False, rule='endpoints_mean'):
    """ Performs a binning of matrix 'A' using Algorithm 1 with input parameter 'c' and 'tau'. 
    
    WARNING: Updates 'A' in-place.
    
    """

    assert A.shape[0] == A.shape[1], "Not a square matrix!"
    assert c > 0 and c < 1, f"Invalid parameter range: c={c}"
    assert tau > 0 and tau < 1, f"Invalid parameter range: tau={tau}"
    assert A.max() <= 1 and A.min() >= 0, "Entries not in range [0, 1]"

    space_requirement = 0
    intervals = []
    for i in range(A.shape[0]):
        # Compute intervals after a merge
        intervals.insert(0, (i, i))
        assert A[i, :i+1].min() >= 0, f"Negative entry encountered in {i}-th row of A"
        intervals = merge_intervals(intervals, A[i, :i+1], c, tau)
        space_requirement = max(space_requirement, len(intervals))

        # Use the intervals to express the row
        A[i, :i+1] = intervals_to_row(intervals, A[i, :i+1], rule=rule)

    if perform_extra_checks:
        is_aligned, space, _ = space_check.verify_efficient_structure(A)
        assert is_aligned, "The rows in the matrix are not aligned to allow for streaming"
        assert space_requirement == space, f"Conflicting results for space needed {space_requirement}!={space}"

    return A, space_requirement

def approx_bennett_mm(n, c, tau, perform_extra_checks=False, rule='endpoints_mean'):
    """ Performs a binning of the Bennett matrix of size 'n' using Algorithm 1 with input parameter 'c, tau'. """

    L, space_requirement = approx_matrix(bennett_matrix(n), c, tau, perform_extra_checks=perform_extra_checks, rule=rule)
    # Compute the corresponding R
    R = scipy.linalg.solve_triangular(L, counting_matrix(n), lower=True)

    return L, R, space_requirement
    
def check_if_monotone_ratio(A):
    """Checks if a lower-triangular matrix 'A' has a monotone ratio.
    
    See Property 3) in the definition of MRMs.
    
    """
    assert A.shape[0] == A.shape[1]
    n = A.shape[0]

    for j1 in range(n):
        for j2 in range(j1+1, n):
            h = A[j2:, j1] / A[j2:, j2]

            # Find out if 'h' is non-decreasing
            if any( np.diff(h) < 0 ):
                return False
    return True

###
# Related to momentum + weight decay
###
def counting_matrix_with_decay_and_momentum(n, alpha, beta):
    """ Return A_{'alpha', 'beta'}) of size 'n'. 
    
    See https://arxiv.org/pdf/2405.13763 Section 3.1 for more information.
    
    """

    assert beta >= 0
    assert beta < alpha
    assert alpha <= 1

    a = np.array( [ math.pow(alpha, i+1) - math.pow(beta, i+1) for i in range(n) ] ) / (alpha - beta)

    return scipy.linalg.toeplitz(a, [1] + [0] * (n - 1))

def get_square_root_matrix(n, alpha, beta):
    """ Return sqrt(A_{'alpha', 'beta'}) of size 'n'.
    
    See https://arxiv.org/pdf/2405.13763 Section 3.1 for more information.
    
    """

    assert beta >= 0
    assert beta < alpha
    assert alpha <= 1

    b = bennett_coefficients(n)

    c = np.ones(n)
    for j in range(1, n):
        c[j] = sum( math.pow(alpha, j - i) * math.pow(beta, i) * b[j-i] * b[i] for i in range(j+1) )

    # NOTE: For sufficiently small 'alpha' and large 'n', entries may be rounded to '0'
    assert c.min() >= 0, "Entry < 0 encountered in c"

    return scipy.linalg.toeplitz(c, [1] + [0] * (n - 1))

def approx_counting_with_decay_momentum(n, alpha, beta,  c, tau, perform_extra_checks=False, rule='endpoints_mean'):
    """ Performs a binning of sqrt(A_{'alpha', 'beta'}) of size 'n' using Algorithm 1 with input parameter 'c'. """

    L, space_requirement = approx_matrix(get_square_root_matrix(n, alpha, beta), c, tau, perform_extra_checks=perform_extra_checks, rule=rule)
    # Compute the corresponding R
    R = scipy.linalg.solve_triangular(L, counting_matrix_with_decay_and_momentum(alpha=alpha, beta=beta, n=n), lower=True)

    return L, R, space_requirement
