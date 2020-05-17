"""
This module contains function to calculate USR similarity scores between USR moment
arrays and larger sets thereof.
"""
import numpy as np


def calc_usr_sim(m1: np.ndarray, m2: np.ndarray) -> float:
    """
    Returns the USR similarity between two sets of USR moments.

    Raises
    ------
    ValueError
        If either `m1` or `m2` has more than one dimension.

    See Also
    --------
    usr.similarity.calc_usr_sim_nd
        Function that can be used to calculate the similarity between one and many USR
        moments.

    Examples
    --------
    >>> m1 = np.array([
    ...     4.34859836, 2.08080746, 0.49435263, 4.36142747, 2.14622434, 0.21392014,
    ...     8.63819824, 4.46538956, -0.58845999, 7.18077256, 4.34155721, 0.59564972])
    >>> m2 = np.array([
    ...     4.6100277, 2.14107738, 0.41200207, 4.5819709, 2.30719899, 0.4590534,
    ...     9.3256312, 4.77782589, -0.6800332 , 7.31515403, 4.51165546, 0.61859795])
    >>> sim = calc_usr_sim(m1, m2)
    >>> np.isclose(sim, 0.83, rtol=0.01)
    True
    """
    if not (m1.ndim == m2.ndim == 1):
        raise ValueError(
            f"cannot calculate USR similarity score: `m1` and `m2` must be a single "
            f"array of USR moments, not {m1.shape} / {m2.shape}."
        )

    return 1.0 / (1.0 + np.abs(m1 - m2).sum() / 12)


def calc_usr_sim_nd(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """
    Returns the USR similarity scores between a single reference set of USR moments `m1`
    and an array containing many, `m2`.

    Parameters
    ----------
    m1: np.ndarray
        Array in the shape (12,) containing a single set of USR moments.
    m2: np.ndarray
        Array in the shape (-1, 12) containing many USR moments.

    Returns
    -------
    scores : np.ndarray
        Array containing the USR similarity between `m1` and each set of USR moments in
        `m2`.

    Examples
    --------
    Similarity between two single USR moment arrays

    >>> m1 = np.array([
    ...     4.34859836, 2.08080746, 0.49435263, 4.36142747, 2.14622434, 0.21392014,
    ...     8.63819824, 4.46538956, -0.58845999, 7.18077256, 4.34155721, 0.59564972])
    >>> m3 = np.array([
    ...     [4.36429755, 2.15300964, 0.43715434,  4.37717747, 2.17126829, 0.36962492,
    ...      8.94737014, 4.35744822, -0.60149784, 7.37277859, 4.28824251, 0.59016373],
    ...     [4.49636078, 1.94280714, 0.59848894, 4.55326873, 2.24365412, 0.22350438,
    ...      8.98633362, 4.51972326, -0.71319056, 7.03050308, 4.17132961, 0.59401722],
    ...     [4.53439992, 2.05733505, 0.20057339, 4.50595881, 2.22565709, 0.22377553,
    ...      9.16843659, 4.61779919, -0.70101657, 7.23349541, 4.35435924, 0.54254737]])
    >>> for score in calc_usr_sim_nd(m1, m3):
    ...     print(np.round(score, 3))
    0.921
    0.886
    0.879
    """
    if m2.ndim != 2:
        raise ValueError(
            f"cannot calculate USR similarity scores: `m2` must be an array in the "
            f"shape (-1, 12), not {m2.shape}."
        )

    rows, cols = m2.shape

    return 1.0 / (1.0 + np.abs(m1-m2).sum(axis=1) / cols)
