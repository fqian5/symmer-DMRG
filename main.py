import numpy as np
from ncon import ncon
from typing import Union, List, Dict, Tuple
from symmer.operators import PauliwordOp, QuantumState
from copy import copy
from functools import cached_property
from quimb.experimental.operatorbuilder import SparseOperatorBuilder
from quimb.tensor import MatrixProductOperator
from quimb.tensor.tensor_dmrg import DMRG2
class MPO(MatrixProductOperator):
    """
    Class to build MPO operator from Pauli strings and coeffs.
    """

    def __init__(self):
        """
        Initialize an MPO and inherit from QUIMB's MatrixProductOperator.
        """
        super().__init__()
    @classmethod
    def from_dictionary(cls,operator: Dict[str, complex], Dmax: int = None) -> MatrixProductOperator:
        """
        Initialize MPO using Hamiltonian dictionary.

        Args:
            operator (Hamiltonian_dict): Hamiltonian dictionary in the form: H_dict = {'IIIIIIII': -331.6627801640259,'ZIIIIIII': 59.14285951244939,'IIIIZIII': 3.7005815842211165}
            Dmax (int): Maximum bond dimension. By default it is set to 'None'. 

        Returns:
            MPO: Matrix Product Operator object in QUIMB.
        """
        mpo = get_MPO(operator, Dmax)
        return mpo
    @classmethod
    def from_WordOp(cls,
            WordOp: PauliwordOp) -> MatrixProductOperator:
        """
        Initialize MPO using PauliwordOp.

        Args:
            WordOp (PauliwordOp): PauliwordOp to initialize MPO Approximator.

        Returns:
            Matrix Product Operator (MPO) object in QUIMB.
        """
        mpo = get_MPO(WordOp.to_dictionary)
        return mpo
def get_MPO(operator: Dict[str, complex], max_bond_dimension: int = None) -> MatrixProductOperator:
    """ 
    Return the Matrix Product Operator (MPO)in QUIMB MPO Object of a Hamiltonian dictionary.

    Args: 
        operator (Hamiltonian_dict): Hamiltonian dictionary in the form: H_dict = {'IIIIIIII': -331.6627801640259,'ZIIIIIII': 59.14285951244939,'IIIIZIII': 3.7005815842211165}
        max_bond_dimension (int): Maximum bond dimension.

    Returns:
        MPOBuilder: MPOBuilder representing operator.
    """
    builder = SparseOperatorBuilder()
    for bitstring, coeff in operator.items():
        ops = bitstring.lower()
        #ops = bitstring
        term = [
            (op,i) 
            for i, op in enumerate(ops)
            if op != 'i' 
        ]
        #print(term)
        builder.add_term(coeff, *term)

    H_mpo = builder.build_mpo()
    return H_mpo

def dmrg_solver(MPOOp: MatrixProductOperator, dmrg=None, gs_guess=None,bond_dims=[10, 20, 100, 100, 200],cutoffs=1e-10) -> Tuple[QuantumState,np.complex128]:
    """
    Use quimb to find the groundstate and energy of an MPOOp.

    Args:
        MPOOp: MPOOp representing operator.
        dmrg: Quimb DMRG solver class. By default it is set to 'None'.
        gs_guess: Guess for the ground state, used as intialisation for the DMRG optimiser. Represented as a dense array. By default it is set to 'None'.
    Returns:
        dmrg_state (QuantumState): Approximated groundstate.
        energy (np.complex128): Energy of the groundstate.
    """

    ### ground state initialisation to be done
    if dmrg is None:
        dmrg = DMRG2(MPOOp, bond_dims=bond_dims, cutoffs=cutoffs, p0=gs_guess)
    dmrg.solve(verbosity=0, tol=1e-6)

    dmrg_state = dmrg.state.to_dense()
    dmrg_state = QuantumState.from_array(dmrg_state).cleanup(zero_threshold=1e-5)
    dmrg_energy = dmrg.energy.real
    return dmrg_state, dmrg_energy