"""
"""

import numpy as np

from qiskit.quantum_info import Pauli
from qiskit.chemistry import FermionicOperator, QiskitChemistryError
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.aqua.operators import Z2Symmetries, WeightedPauliOperator

from openfermion import (
        MolecularData,
        symmetry_conserving_bravyi_kitaev,
        get_fermion_operator
    )
from openfermion.linalg import get_ground_state, get_sparse_operator
from openfermion.transforms import freeze_orbitals

from openfermionpyscf import run_pyscf


def convert_wpo_and_openfermion(operator):
    """
    Converts between openfermion qubit hamiltonians and qiskit weighted Pauli operators
    Uses dict decompositions in both cases and is full general. 
    Parameters
    ----------
    operator : openfermion hamiltonian OR qiskit wpo
        Input operator 

    Returns
    -------
    operator : openfermion hamiltonian OR qiskit wpo

    """
    def _count_qubits(openfermion_operator):
        """ Counts the number of qubits in the openfermion.operator""" 
        nb_qubits = 0
        for sett, coef in openfermion_operator.terms.items():
            if len(sett)>0:
                nb_qubits = max(nb_qubits, max([s[0] for s in sett]))
        return nb_qubits+1
                
    # (commented out version is for updated git version of openfermion)
    # import openfermion
    # if type(operator) is openfermion.ops.operators.qubit_operator.QubitOperator:
    if str(operator.__class__) == "<class 'openfermion.ops._qubit_operator.QubitOperator'>":
        nb_qubits = _count_qubits(operator)

        iden = Pauli.from_label('I'*nb_qubits)
        qiskit_operator = WeightedPauliOperator([(0., iden)])
        for sett, coef in operator.terms.items():
            new_sett = 'I'*nb_qubits
            for s in sett:
                new_sett = new_sett[:(s[0])] + s[1] + new_sett[(s[0]+1):]
            pauli = Pauli.from_label(new_sett)
            op = WeightedPauliOperator([(coef, pauli)])
            # print(coef)
            # print(new_sett)
            qiskit_operator = qiskit_operator + op
        return qiskit_operator    
    else:
        raise NotImplementedError("Currently only converts 1 way, openfermion-> qiskit wpo")


def get_H2_qubit_op(dist):
    """ 
    Use the qiskit chemistry package to get the qubit Hamiltonian for H2

    Parameters
    ----------
    dist : float
        The nuclear separations

    Returns
    -------
    qubitOp : qiskit.aqua.operators.WeightedPauliOperator
        Qiskit representation of the qubit Hamiltonian, energy shifts are
        incorporated into the identity string of the qubit Hamiltonian
    """
    # I have experienced some crashes
    _retries = 50
    for i in range(_retries):
        try:
            driver = PySCFDriver(atom="H .0 .0 .0; H .0 .0 " + str(dist), 
                                 unit=UnitsType.ANGSTROM, 
                                 charge=0, 
                                 spin=0, 
                                 basis='sto3g',
                                )
            molecule = driver.run()
            repulsion_energy = molecule.nuclear_repulsion_energy
            num_particles = molecule.num_alpha + molecule.num_beta
            num_spin_orbitals = molecule.num_orbitals * 2
            ferOp = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)
            qubitOp = ferOp.mapping(map_type='parity', threshold=1E-8)
            qubitOp = Z2Symmetries.two_qubit_reduction(qubitOp,num_particles)
            break
        except QiskitChemistryError:
            if i==(_retries-1):
                raise
            pass

    # add nuclear repulsion energy onto identity string
    qubitOp = qubitOp + WeightedPauliOperator([[repulsion_energy,Pauli(label='I'*qubitOp.num_qubits)]])

    return qubitOp


def get_LiH_qubit_op(dist):
    """ 
    Use the qiskit chemistry package to get the qubit Hamiltonian for LiH

    Parameters
    ----------
    dist : float
        The nuclear separations

    Returns
    -------
    qubitOp : qiskit.aqua.operators.WeightedPauliOperator
        Qiskit representation of the qubit Hamiltonian, energy shifts are
        incorporated into the identity string of the qubit Hamiltonian
    """
    # I have experienced some crashes
    _retries = 50
    for i in range(_retries):
        try:
            driver = PySCFDriver(atom="Li .0 .0 .0; H .0 .0 " + str(dist), 
                                 unit=UnitsType.ANGSTROM, 
                                 charge=0, 
                                 spin=0, 
                                 basis='sto3g',
                                )
            molecule = driver.run()
            freeze_list = [0]
            remove_list = [-3, -2]
            repulsion_energy = molecule.nuclear_repulsion_energy
            num_particles = molecule.num_alpha + molecule.num_beta
            num_spin_orbitals = molecule.num_orbitals * 2
            remove_list = [x % molecule.num_orbitals for x in remove_list]
            freeze_list = [x % molecule.num_orbitals for x in freeze_list]
            remove_list = [x - len(freeze_list) for x in remove_list]
            remove_list += [x + molecule.num_orbitals - len(freeze_list)  for x in remove_list]
            freeze_list += [x + molecule.num_orbitals for x in freeze_list]
            ferOp = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)
            ferOp, energy_shift = ferOp.fermion_mode_freezing(freeze_list)
            num_spin_orbitals -= len(freeze_list)
            num_particles -= len(freeze_list)
            ferOp = ferOp.fermion_mode_elimination(remove_list)
            num_spin_orbitals -= len(remove_list)
            qubitOp = ferOp.mapping(map_type='parity', threshold=1E-8)
            #qubitOp = qubitOp.two_qubit_reduced_operator(num_particles)
            qubitOp = Z2Symmetries.two_qubit_reduction(qubitOp,num_particles)
            shift = repulsion_energy + energy_shift
            break
        except QiskitChemistryError:
            if i==(_retries-1):
                raise
            pass

    # add energy shifts onto identity string
    qubitOp = qubitOp + WeightedPauliOperator([[shift,Pauli(label='I'*qubitOp.num_qubits)]])

    return qubitOp


def get_H_chain_qubit_op(dist_vec):
    """ 
    Use the qiskit chemistry package to get the qubit Hamiltonian for LiH

    Parameters
    ----------
    dist_vec : float
        Vec of relative nuclear separations. 

    Returns
    -------
    qubitOp : qiskit.aqua.operators.WeightedPauliOperator
        Qiskit representation of the qubit Hamiltonian
    shift : float
        The ground state of the qubit Hamiltonian needs to be corrected by this amount of
        energy to give the real physical energy. This includes the replusive energy between
        the nuclei and the energy shift of the frozen orbitals.
    """
    # I have experienced some crashes

    dist_vec = np.atleast_1d(dist_vec)
    atoms = '; '.join(['H 0 0 {}'.format(dd) for dd in np.cumsum([0] + list(dist_vec))])
    
    atoms = atoms.split('; ')
    open_fermion_geom = []
    for aa in atoms:
        sym = aa.split(' ')[0]
        coords = tuple([float(ii) for ii in aa.split(' ')[1:]])
        open_fermion_geom.append((sym, coords))
        
   
    basis = 'sto-3g'
    multiplicity = 1 + len(atoms)%2
    charge = 0
    molecule = MolecularData(geometry=open_fermion_geom, 
                             basis=basis, 
                             multiplicity=multiplicity,
                             charge=charge)
    num_particles = molecule.get_n_alpha_electrons() + molecule.get_n_beta_electrons()
    molecule = run_pyscf(molecule)
        
    # Convert result to qubit measurement stings
    ham = molecule.get_molecular_hamiltonian()
    fermion_hamiltonian = get_fermion_operator(ham)
    #qubit_hamiltonian = bravyi_kitaev(fermion_hamiltonian)
    qubit_hamiltonian = symmetry_conserving_bravyi_kitaev(
        fermion_hamiltonian,
        active_orbitals=2*molecule.n_orbitals,
        active_fermions=molecule.get_n_alpha_electrons()+molecule.get_n_beta_electrons()
    )
        
    
    return convert_wpo_and_openfermion(qubit_hamiltonian)


def get_HHLi_qubit_op(d1, d2, get_gs=False, get_exact_E=False, freezeOcc=[0,1], freezeEmpt=[6,7,8,9]):
    """
    Generates the qubit weighted pauli operators for a chain of H + H + Li with
    distance d1 between leftmost H and central H, and distance d2 between
    central H and Li.
    Allows for freezing of occupied and empty orbitals to reduce # qubits.
    If requested returns ground state energy and vector AFTER freezing orbitals.
    Exact energy (without freezing orbitals) can also be requested for consistency
    check.
    
    Parameters
    ----------
    d1 : float 
        Distance between leftmost H and central H
    d2 : float
        Distance between central H and Li
    get_gs : boolean
        If true returns ground state energy and vector AFTER orbital freezing
    get_exact_E : boolean
        If true returns exact energy (no freezing orbitals)
    freezeOcc : list of integers
        Indices of orbitals to be frozen as occupied - [0,1] for chemical
        reaction simulation
    freezeEmpt : list of integers
        Indices of orbitals to be frozen as empty - [6,7,8,9] for chemical
        reaction simulation

    Returns
    -------
    out : WeightedPauliOperator, ndarray if get_gs, float if get_gs, float if get_exact_E
        Weighted pauli operator for qubit Hamiltonian, ground state vector if requested,
        ground state energy if requested, exact ground state energy if requested
    """
    atoms='H 0 0 {}; H 0 0 {}; Li 0 0 {}'.format(-d1, 0, d2)
    # Converts string to openfermion geometery
    n_frozen=len(freezeOcc)+len(freezeEmpt)
    atom_vec = atoms.split('; ')
    open_fermion_geom = []
    for aa in atom_vec:
        sym = aa.split(' ')[0]
        coords = tuple([float(ii) for ii in aa.split(' ')[1:]])
        open_fermion_geom.append((sym, coords))
    basis = 'sto-6g'
    multiplicity = 1 + len(atom_vec)%2
    charge = 0
    # Construct the molecule and calc overlaps
    molecule = MolecularData(
        geometry=open_fermion_geom, 
        basis=basis, 
        multiplicity=multiplicity,
        charge=charge,
    )
    num_particles = molecule.get_n_alpha_electrons() + molecule.get_n_beta_electrons()
    molecule = run_pyscf(
        molecule,
    )
    _of_molecule = molecule

    # Convert result to qubit measurement stings
    ham = molecule.get_molecular_hamiltonian()
    fermion_hamiltonian = get_fermion_operator(ham)
    if get_exact_E:
        sparse_operator = get_sparse_operator(fermion_hamiltonian)
        egse, egs=get_ground_state(sparse_operator)
    fermion_hamiltonian = freeze_orbitals(fermion_hamiltonian, freezeOcc, freezeEmpt)
    sparse_operator = get_sparse_operator(fermion_hamiltonian)
    gse, gs=get_ground_state(sparse_operator)
    qubit_hamiltonian = symmetry_conserving_bravyi_kitaev(
        fermion_hamiltonian,
        active_orbitals=2*molecule.n_orbitals-n_frozen,
        active_fermions=molecule.get_n_alpha_electrons()+molecule.get_n_beta_electrons()
    )
    weighted_pauli_op = convert_wpo_and_openfermion(qubit_hamiltonian)
    if get_gs:
        dense_H = sum([p[0]*p[1].to_matrix() for p in weighted_pauli_op.paulis])
        sparse_operator = get_sparse_operator(qubit_hamiltonian)
        gse, gsv=get_ground_state(sparse_operator)
        out = weighted_pauli_op, gse, gsv
    else:
        out = weighted_pauli_op
    if get_exact_E:
        out=*out, egse
    return out


def check_HHLi(occ=[], uocc=[], d1=1, d2=2.39):
    """
    Lightweight version of get_HHLi_qubit_op that just returns the
    ground state energy of the Hamiltonian with the specified 
    orbitals frozen, useful for checking which to freeze.
    
    Parameters
    ----------
    occ : list of integers
        Indices of orbitals to be frozen as occupied
    uocc : list of integers
        Indices of orbitals to be frozen as empty
    d1 : float 
        Distance between leftmost H and central H
    d2 : float
        Distance between central H and Li

    Returns
    -------
    GSE : float
        Ground state energy of Hamiltonian with specified distances
        and freezing
    """
    atoms='H 0 0 {}; H 0 0 {}; Li 0 0 {}'.format(-d1, 0, d2)
    # Converts string to openfermion geometery
    atom_vec = atoms.split('; ')
    open_fermion_geom = []
    for aa in atom_vec:
        sym = aa.split(' ')[0]
        coords = tuple([float(ii) for ii in aa.split(' ')[1:]])
        open_fermion_geom.append((sym, coords))
    basis = 'sto-6g'
    multiplicity = 1 + len(atom_vec)%2
    charge = 0
    molecule = MolecularData(
        geometry=open_fermion_geom, 
        basis=basis, 
        multiplicity=multiplicity,
        charge=charge,
    )
    num_particles = molecule.get_n_alpha_electrons() + molecule.get_n_beta_electrons()
    molecule = run_pyscf(
        molecule,
    )
    _of_molecule = molecule

    # Convert result to qubit measurement stings
    ham = molecule.get_molecular_hamiltonian()
    fermion_hamiltonian = get_fermion_operator(ham)
    fermion_h_frozen=freeze_orbitals(fermion_hamiltonian,occ,uocc)
    sparse_operator = get_sparse_operator(fermion_h_frozen)
    GSE=get_ground_state(sparse_operator)[0]
    return GSE
