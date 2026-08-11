import cudaq
import tequila as tq

from gqe_qsci.molecule import PySCFMolecule
from gqe_qsci.gqe.utils import convert_pauli_to_cudaq_spin


def _identity_term(n_qubits: int) -> cudaq.SpinOperatorTerm:
    term = cudaq.spin.i(0)
    for q in range(1, n_qubits):
        term = term * cudaq.spin.i(q)
    return term


def hamiltonian_to_cudaq_spin(molecule: PySCFMolecule) -> cudaq.SpinOperator:
    geometry_lines = [
        f"{atom_type} {coords[0]} {coords[1]} {coords[2]}"
        for atom_type, coords in molecule.geometry
    ]
    geometry_str = "\n".join(geometry_lines)
    tq_molecule = tq.Molecule(
        geometry=geometry_str,
        basis_set=molecule.basis,
        active_orbitals=molecule.active_indices,
        transformation="jordan-wigner",
    )
    H = tq_molecule.make_hamiltonian()
    n_qubits = molecule.norb * 2

    spin_op = None
    for p in H.paulistrings:
        coeff = complex(p.coeff)
        items = dict(p.items())
        if len(items) == 0:
            term = _identity_term(n_qubits)
        else:
            term = convert_pauli_to_cudaq_spin(items)
        contrib = coeff * cudaq.SpinOperator(term)
        spin_op = contrib if spin_op is None else (spin_op + contrib)

    if spin_op is None:
        raise ValueError("Tequila returned an empty Hamiltonian")
    return spin_op
