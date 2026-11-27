"""USR similarity: the score between sets of USR moments.

The score of two moment vectors ``a`` and ``b`` of length ``n`` is
``1 / (1 + mean(|a - b|))``: 1 for identical moments, falling towards 0 as they
differ. :func:`calc_usr_sim` scores one vector against every row of a matrix, as
:func:`usr.search.screen` does for each query conformer.
"""

import numpy as np

from usr.moments import FloatArray


def calc_usr_sim(m1: FloatArray, m2: FloatArray) -> FloatArray:
    """Return the USR similarity of one set of USR moments to each row of a matrix.

    Parameters
    ----------
    m1 : FloatArray
        One set of moments, in the shape ``(n,)``.
    m2 : FloatArray
        Sets of moments to score, in the shape ``(rows, n)``. To score a pair, pass
        the second set as one row: ``m2[None]``.

    Returns
    -------
    FloatArray
        The score of ``m1`` against each row of ``m2``, in the interval ``(0, 1]``, in
        the shape ``(rows,)``.

    Raises
    ------
    ValueError
        If ``m1`` is not one-dimensional, ``m2`` is not two-dimensional, or a row of
        ``m2`` differs in length from ``m1``.

    Examples
    --------
    Moments that differ by 0.5 each score 1 / 1.5, and by 1 each 1 / 2:

    >>> database = np.array([np.zeros(12), np.full(12, 0.5), np.ones(12)])
    >>> print(np.round(calc_usr_sim(np.zeros(12), database), 4))
    [1.     0.6667 0.5   ]
    """
    if m1.ndim != 1 or m2.ndim != 2 or m2.shape[1] != m1.size:
        raise ValueError(
            f"cannot calculate USR similarities: m1 must be one set of moments and m2 "
            f"a matrix of sets of the same length, not the shapes {m1.shape} and "
            f"{m2.shape}"
        )

    return 1.0 / (1.0 + np.mean(np.abs(m1 - m2), axis=1))
