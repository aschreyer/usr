"""Tests for the USRCAT atom types: SMARTS, their queries and atom typing."""

import pytest
from rdkit import Chem

from usr.chem import (
    USRCAT_SMARTS,
    compile_smarts,
    get_pharmacophore_subsets,
)


def test_the_usrcat_smarts_name_four_types_in_block_order() -> None:
    assert list(USRCAT_SMARTS) == ["hydrophobic", "aromatic", "acceptor", "donor"]
    assert USRCAT_SMARTS["aromatic"] == "[a]"


def test_the_usrcat_smarts_cannot_be_changed() -> None:
    with pytest.raises(TypeError):
        USRCAT_SMARTS["aromatic"] = "[c]"  # type: ignore[index]


@pytest.mark.parametrize(
    ("smiles", "acceptors", "donors"),
    [
        ("CC(=O)O", (2,), (3,)),
        ("NC(=O)N", (2,), (0, 3)),
        ("CC(N)=O", (3,), (2,)),
    ],
)
def test_explicit_hydrogens_do_not_make_acids_and_amides_acceptors(
    smiles: str, acceptors: tuple[int, ...], donors: tuple[int, ...]
) -> None:
    """Typing runs on the heavy-atom graph, so both settings give the same types.

    Matched on the graph with explicit Hs, the acceptor SMARTS also types the hydroxyl
    oxygen of an acid and the nitrogen of an amide or urea as acceptors: the test
    shows that first.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    pattern = Chem.MolFromSmarts(USRCAT_SMARTS["acceptor"])
    naive = {match[0] for match in mol.GetSubstructMatches(pattern)}

    assert naive > set(acceptors)
    for explicit_h in (False, True):
        subsets = get_pharmacophore_subsets(mol, explicit_h=explicit_h)
        assert subsets[2:] == (acceptors, donors)


def test_explicit_h_subsets_index_the_molecule_with_its_hydrogens() -> None:
    """Derived by hand for acetic acid renumbered to put its acidic H first.

    With Hs, the atoms are H 0, O 1, C 2, O 3, C 4 and three Hs; without, O 0, C 1,
    O 2 and C 3. The methyl carbon is hydrophobic, the carbonyl oxygen accepts and the
    hydroxyl oxygen donates. The caller's molecule keeps no trace of the mapping.
    """
    # AddHs numbers C 0, C 1, O 2 (carbonyl), O 3 (hydroxyl), H 4 to 6 and H 7 on O 3.
    added = Chem.AddHs(Chem.MolFromSmiles("CC(=O)O"))
    mol = Chem.RenumberAtoms(added, [7, 3, 1, 2, 0, 4, 5, 6])

    with_h = get_pharmacophore_subsets(mol, explicit_h=True)
    without_h = get_pharmacophore_subsets(mol)

    assert [atom.GetSymbol() for atom in mol.GetAtoms()][:5] == [
        "H",
        "O",
        "C",
        "O",
        "C",
    ]
    assert with_h == ((4,), (), (3,), (1,))
    assert without_h == ((3,), (), (2,), (0,))
    assert not any(atom.HasProp("_usr_atom_index") for atom in mol.GetAtoms())


def test_custom_smarts_type_the_atoms() -> None:
    mol = Chem.MolFromSmiles("OCc1ccccc1O")

    assert get_pharmacophore_subsets(mol, smarts=["[#8]", "[c;R]~[c;R]"]) == (
        (0, 8),
        (2, 3, 4, 5, 6, 7),
    )


@pytest.mark.parametrize(
    ("smarts", "message"),
    [
        (["[#8]", ""], "smarts[1] is not a valid SMARTS: ''"),
        (["[C"], "smarts[0] is not a valid SMARTS: '[C'"),
        ("[a]", "smarts must be a sequence of SMARTS strings, not the string '[a]'"),
    ],
)
def test_an_invalid_smarts_is_refused(smarts: list[str] | str, message: str) -> None:
    with pytest.raises(ValueError, match="smarts") as caught:
        get_pharmacophore_subsets(Chem.MolFromSmiles("CCO"), smarts=smarts)

    assert str(caught.value) == message


def test_the_default_queries_are_compiled_once() -> None:
    assert compile_smarts() is compile_smarts(None)
    assert len(compile_smarts()) == len(USRCAT_SMARTS)


def test_custom_smarts_are_compiled_once() -> None:
    first = compile_smarts(["[#8]", "[#7]"])

    assert compile_smarts(["[#7]"])[0] is first[1]


PHENOL_TYPES = ((0, 1, 2, 3, 4), tuple(range(6)), (6,), (6,))
"""The atom types of phenol, ``c1ccccc1O``, derived by hand.

The ring carbon bound to O is not hydrophobic, and the oxygen accepts and donates.
"""


def count_remove_hs(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count the calls of ``Chem.RemoveHs`` for the rest of the test, one 1 each."""
    calls: list[int] = []
    remove_hs = Chem.RemoveHs

    def counted(mol: Chem.Mol) -> Chem.Mol:
        calls.append(1)
        return remove_hs(mol)

    monkeypatch.setattr(Chem, "RemoveHs", counted)
    return calls


def test_a_heavy_atom_molecule_is_typed_without_removing_hs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mol = Chem.MolFromSmiles("c1ccccc1O")
    calls = count_remove_hs(monkeypatch)

    assert get_pharmacophore_subsets(mol) == PHENOL_TYPES
    assert get_pharmacophore_subsets(mol, explicit_h=True) == PHENOL_TYPES
    assert calls == []


@pytest.mark.parametrize("explicit_h", [False, True])
def test_a_molecule_with_hs_is_typed_after_one_removal(
    explicit_h: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``Chem.AddHs`` puts the H atoms last: both settings index the atoms alike."""
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))
    calls = count_remove_hs(monkeypatch)

    assert get_pharmacophore_subsets(mol, explicit_h=explicit_h) == PHENOL_TYPES
    assert calls == [1]
    assert mol.GetNumAtoms() == 13
