"""USR similarity: the score between two sets of USR moments.

The score of two moment vectors ``a`` and ``b`` of length ``n`` is
``1 / (1 + sum(|a - b|) / n)``: 1 for identical moments, falling towards 0 as they
differ. :func:`calc_usr_sim` scores one pair, and :func:`calc_usr_sim_nd` one vector
against every row of a matrix, as a screen of a database does.
"""

import numpy as np

from usr.moments import FloatArray


def calc_usr_sim(m1: FloatArray, m2: FloatArray) -> float:
    """Return the USR similarity of two sets of USR moments.

    Parameters
    ----------
    m1, m2 : FloatArray
        One set of moments each, in the same one-dimensional shape.

    Returns
    -------
    float
        The score, in the interval ``(0, 1]``.

    Raises
    ------
    ValueError
        If ``m1`` or ``m2`` is not one-dimensional, or the two differ in length.

    See Also
    --------
    calc_usr_sim_nd : Score one set of moments against many.

    Examples
    --------
    Moments that differ by 0.2 each score 1 / 1.2:

    >>> round(calc_usr_sim(np.zeros(12), np.full(12, 0.2)), 4)
    0.8333
    """
    if not (m1.ndim == m2.ndim == 1) or m1.shape != m2.shape:
        raise ValueError(
            f"cannot calculate a USR similarity: m1 and m2 must be one set of moments "
            f"each, of the same length, not the shapes {m1.shape} and {m2.shape}"
        )

    return float(1.0 / (1.0 + np.abs(m1 - m2).sum() / m1.size))


def calc_usr_sim_nd(m1: FloatArray, m2: FloatArray) -> FloatArray:
    """Return the USR similarity of one set of USR moments to each row of a matrix.

    Parameters
    ----------
    m1 : FloatArray
        One set of moments, in the shape ``(n,)``.
    m2 : FloatArray
        Many sets of moments, in the shape ``(rows, n)``.

    Returns
    -------
    FloatArray
        The score of ``m1`` against each row of ``m2``, in the shape ``(rows,)``.

    Raises
    ------
    ValueError
        If ``m1`` is not one-dimensional, ``m2`` is not two-dimensional, or a row of
        ``m2`` differs in length from ``m1``.

    See Also
    --------
    calc_usr_sim : Score one pair.

    Examples
    --------
    >>> database = np.array([np.zeros(12), np.full(12, 0.5), np.ones(12)])
    >>> print(np.round(calc_usr_sim_nd(np.zeros(12), database), 4))
    [1.     0.6667 0.5   ]
    """
    if m1.ndim != 1 or m2.ndim != 2 or m2.shape[1] != m1.size:
        raise ValueError(
            f"cannot calculate USR similarities: m1 must be one set of moments and m2 "
            f"a matrix of sets of the same length, not the shapes {m1.shape} and "
            f"{m2.shape}"
        )

    return 1.0 / (1.0 + np.abs(m1 - m2).sum(axis=1) / m1.size)
