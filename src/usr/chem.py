"""Chemistry for USRCAT: the pharmacophoric atom types of a molecule.

USRCAT (Schreyer and Blundell, 2012) describes, besides all atoms, the atoms of four
types, which :data:`USRCAT_SMARTS` defines: hydrophobic, aromatic, hydrogen-bond
acceptor and hydrogen-bond donor. :func:`compile_smarts` turns SMARTS into RDKit
queries, checking them, and :func:`get_pharmacophore_subsets` matches them against a
molecule, giving an :data:`AtomSubsets`, the indices of the atoms of each type.

The queries always match the heavy-atom graph, so that an explicit hydrogen atom
never changes the type of the atom it is bound to. :mod:`usr.moments` calls this
module to calculate USRCAT moments.
"""

import functools
from collections.abc import Mapping, Sequence
from types import MappingProxyType

from rdkit import Chem


type AtomSubsets = tuple[tuple[int, ...], ...]
"""Sorted, 0-based atom indices, one tuple per atom type."""

USRCAT_SMARTS: Mapping[str, str] = MappingProxyType(
    {
        "hydrophobic": "[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]",
        "aromatic": "[a]",
        "acceptor": (
            "[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),"
            "$([N&v3;H1,H2]-[!$(*=[O,N,P,S])]),$([N;v3;H0]),$([n,o,s;+0]),F]"
        ),
        "donor": "[N!H0v3,N!H0+v4,OH+0,SH+0,nH+0]",
    }
)
"""The SMARTS of the USRCAT atom types, in block order, as RDKit defines them.

Hydrophobic, aromatic, hydrogen-bond acceptor and hydrogen-bond donor atoms, the blocks
after the all-atom block. Read only.
"""

_USRCAT_QUERIES = tuple(Chem.MolFromSmarts(smarts) for smarts in USRCAT_SMARTS.values())
"""The queries of :data:`USRCAT_SMARTS`, compiled once."""

_ALL_MATCHES = 2**31 - 1
"""The match limit of a query: none, as RDKit's default of 1000 cuts large molecules."""

_ATOM_INDEX = "_usr_atom_index"
"""The atom property that carries an atom's index through ``Chem.RemoveHs``."""


# Cached, so that a table of molecules typed with the same custom SMARTS compiles each
# SMARTS once. The queries are shared and read only.
@functools.cache
def _compile_pattern(smarts: str) -> Chem.Mol | None:
    """Return the RDKit query of one SMARTS, or None if it is empty or invalid."""
    return Chem.MolFromSmarts(smarts) if smarts.strip() else None


def compile_smarts(smarts: Sequence[str] | None = None) -> tuple[Chem.Mol, ...]:
    """Return the RDKit queries of SMARTS strings, checked.

    Each SMARTS is compiled once per process; the queries are shared, so do not modify
    them.

    Parameters
    ----------
    smarts : Sequence[str] | None, optional
        One SMARTS per atom type. Default is None, the values of
        :data:`USRCAT_SMARTS`, compiled once at import.

    Returns
    -------
    tuple[Chem.Mol, ...]
        One query per SMARTS, in order.

    Raises
    ------
    ValueError
        If ``smarts`` is a single string, or holds a string that is empty or that RDKit
        cannot parse.

    Examples
    --------
    >>> len(compile_smarts()), len(compile_smarts(["[#8]"]))
    (4, 1)
    """
    if smarts is None:
        return _USRCAT_QUERIES
    if isinstance(smarts, str):
        raise ValueError(
            f"smarts must be a sequence of SMARTS strings, not the string {smarts!r}"
        )
    queries: list[Chem.Mol] = []
    for number, text in enumerate(smarts):
        query = _compile_pattern(text)
        if query is None:
            raise ValueError(f"smarts[{number}] is not a valid SMARTS: {text!r}")
        queries.append(query)

    return tuple(queries)


def _match_smarts(mol: Chem.Mol, queries: Sequence[Chem.Mol]) -> AtomSubsets:
    """Return the sorted indices of the atoms of ``mol`` in any match of each query."""
    return tuple(
        tuple(
            sorted(
                {
                    index
                    for match in mol.GetSubstructMatches(query, maxMatches=_ALL_MATCHES)
                    for index in match
                }
            )
        )
        for query in queries
    )


def _has_explicit_hs(mol: Chem.Mol) -> bool:
    """Return whether any atom of ``mol`` is a hydrogen atom of the graph."""
    return any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms())


def _remove_hs_keeping_indices(mol: Chem.Mol) -> tuple[Chem.Mol, list[int]]:
    """Return ``Chem.RemoveHs(mol)`` and the index in ``mol`` of each of its atoms.

    The atoms of a copy carry their index through ``Chem.RemoveHs``; ``mol`` itself is
    not modified.
    """
    tagged = Chem.Mol(mol)
    for atom in tagged.GetAtoms():
        atom.SetIntProp(_ATOM_INDEX, atom.GetIdx())
    heavy = Chem.RemoveHs(tagged)

    return heavy, [atom.GetIntProp(_ATOM_INDEX) for atom in heavy.GetAtoms()]


def get_pharmacophore_subsets(
    mol: Chem.Mol, *, explicit_h: bool = False, smarts: Sequence[str] | None = None
) -> AtomSubsets:
    """Return the atoms of ``mol`` of each USRCAT atom type.

    The queries match the molecule without its explicit hydrogen atoms, so that an
    explicit hydrogen atom never changes the type of the atom it is bound to. A
    molecule without hydrogen atoms in its graph is matched as it is, without a copy.

    Parameters
    ----------
    mol : Chem.Mol
        The molecule. It is not modified.
    explicit_h : bool, optional
        Whether the indices count the explicit hydrogen atoms of ``mol``, as its
        coordinates do. Default is False: the indices are those of
        ``Chem.RemoveHs(mol)``, whose coordinates
        :func:`usr.moments.get_usrcat_moments` uses.
    smarts : Sequence[str] | None, optional
        One SMARTS per atom type. Default is None, the values of
        :data:`USRCAT_SMARTS`.

    Returns
    -------
    AtomSubsets
        For each SMARTS in turn, the sorted 0-based indices of the atoms in any match.

    Raises
    ------
    ValueError
        If ``smarts`` is a single string, or holds a string that is empty or that RDKit
        cannot parse.

    Examples
    --------
    Acetic acid: the methyl carbon is hydrophobic, the carbonyl oxygen accepts and the
    hydroxyl oxygen donates. With its hydrogen atoms, the indices are the same, since
    ``Chem.AddHs`` puts them last.

    >>> acid = Chem.MolFromSmiles("CC(=O)O")
    >>> get_pharmacophore_subsets(acid)
    ((0,), (), (2,), (3,))
    >>> get_pharmacophore_subsets(Chem.AddHs(acid), explicit_h=True)
    ((0,), (), (2,), (3,))
    """
    queries = compile_smarts(smarts)
    if not _has_explicit_hs(mol):
        return _match_smarts(mol, queries)
    if not explicit_h:
        return _match_smarts(Chem.RemoveHs(mol), queries)

    heavy, original = _remove_hs_keeping_indices(mol)

    return tuple(
        tuple(original[index] for index in subset)
        for subset in _match_smarts(heavy, queries)
    )
