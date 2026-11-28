"""USR and USRCAT similarity: the score between sets of moments.

The USR score of two moment vectors ``a`` and ``b`` of length ``n`` is
``1 / (1 + mean(|a - b|))``: 1 for identical moments, falling towards 0 as they
differ. :func:`calc_usr_sim` scores one vector against every row of a matrix.

The USRCAT score splits the vectors into blocks of :data:`~usr.moments.NUM_MOMENTS`
and weights the mean absolute difference of each block:
``1 / (1 + sum_k w_k * mean_k(|a - b|))``. :func:`calc_usrcat_sim` computes it, and
:func:`validate_weights` checks the weights. With one block and the default weight of 1,
the two scores agree. :func:`usr.search.screen` calls :func:`calc_usrcat_sim` for
each query conformer.
"""

from collections.abc import Sequence

import numpy as np

from usr.moments import NUM_MOMENTS, FloatArray


def _check_moment_shapes(m1: FloatArray, m2: FloatArray, descriptor: str) -> None:
    """Raise ValueError unless ``m1`` is one set of moments and ``m2`` rows like it."""
    if m1.ndim != 1 or m2.ndim != 2 or m2.shape[1] != m1.size:
        raise ValueError(
            f"cannot calculate {descriptor} similarities: m1 must be one set of moments "
            f"and m2 a matrix of sets of the same length, not the shapes {m1.shape} and "
            f"{m2.shape}"
        )


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
    _check_moment_shapes(m1, m2, "USR")

    return 1.0 / (1.0 + np.mean(np.abs(m1 - m2), axis=1))


def validate_weights(
    weights: Sequence[float] | FloatArray | None, num_moments: int
) -> FloatArray:
    """Return the weight of each block of moments, validated.

    Parameters
    ----------
    weights : Sequence[float] | FloatArray | None
        One finite, non-negative weight per block, not all zero. None gives each block
        the weight 1.
    num_moments : int
        The moments per set, a positive multiple of :data:`~usr.moments.NUM_MOMENTS`.

    Returns
    -------
    FloatArray
        The weights, in the shape ``(num_moments // NUM_MOMENTS,)``.

    Raises
    ------
    ValueError
        If ``num_moments`` is not a positive multiple of 12, or ``weights`` is not
        one-dimensional, does not hold one weight per block, holds one that is
        negative or not finite, or is all zero.

    Examples
    --------
    >>> print(validate_weights(None, 60))
    [1. 1. 1. 1. 1.]
    """
    if num_moments < NUM_MOMENTS or num_moments % NUM_MOMENTS:
        raise ValueError(
            f"moments come in blocks of {NUM_MOMENTS}, so {num_moments} moments "
            f"cannot be weighted"
        )
    blocks = num_moments // NUM_MOMENTS
    if weights is None:
        return np.ones(blocks)

    array = np.asarray(weights, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(
            f"weights must be one-dimensional, one weight per block, not of the shape "
            f"{array.shape}"
        )
    if array.size != blocks:
        raise ValueError(
            f"weights must hold one weight per block of {NUM_MOMENTS} moments, "
            f"{blocks} in all, not {array.size}"
        )
    if not (np.isfinite(array) & (array >= 0.0)).all():
        raise ValueError(
            f"weights must be finite and non-negative, not {array.tolist()}"
        )
    if not (array > 0.0).any():
        raise ValueError(f"weights must not all be zero, not {array.tolist()}")

    return array


def calc_usrcat_sim(
    m1: FloatArray,
    m2: FloatArray,
    *,
    weights: Sequence[float] | FloatArray | None = None,
) -> FloatArray:
    """Return the weighted USRCAT similarity of one set of moments to each row of many.

    The score is ``1 / (1 + sum_k w_k * d_k)``, where ``d_k`` is the mean absolute
    difference of block ``k`` of :data:`~usr.moments.NUM_MOMENTS` moments and ``w_k``
    its weight, as in Schreyer and Blundell (2012), Eq. 1. The weights are not
    normalised.

    Parameters
    ----------
    m1 : FloatArray
        One set of moments, in the shape ``(n,)``, where ``n`` is a positive multiple
        of 12: 60 for USRCAT, 12 for USR.
    m2 : FloatArray
        Sets of moments to score, in the shape ``(rows, n)``.
    weights : Sequence[float] | FloatArray | None, optional
        One finite, non-negative weight per block of 12, not all zero. Default is None,
        a weight of 1 for every block.

    Returns
    -------
    FloatArray
        The score of ``m1`` against each row of ``m2``, in the interval ``(0, 1]``, in
        the shape ``(rows,)``.

    Raises
    ------
    ValueError
        If ``m1`` is not one-dimensional, ``m2`` is not two-dimensional, a row of
        ``m2`` differs in length from ``m1``, the length is not a positive multiple of
        12, or the weights are invalid, as :func:`validate_weights` checks them.

    Examples
    --------
    Two blocks that differ by 1 and 3 each score 1 / (1 + 1 + 3) with equal weights,
    and 1 / (1 + 1) when the second block weighs nothing:

    >>> a, b = np.zeros(24), np.repeat([1.0, 3.0], 12)[None]
    >>> print(calc_usrcat_sim(a, b), calc_usrcat_sim(a, b, weights=[1, 0]))
    [0.2] [0.5]
    """
    _check_moment_shapes(m1, m2, "USRCAT")
    weight = validate_weights(weights, m1.size)

    # (rows, blocks): the mean absolute difference of each block of 12.
    distance = np.abs(m1 - m2).reshape(len(m2), weight.size, NUM_MOMENTS).mean(axis=2)

    return 1.0 / (1.0 + distance @ weight)
