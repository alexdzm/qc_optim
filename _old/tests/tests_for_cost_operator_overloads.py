
from ansatz import *
from cost import *
from qiskit import Aer
from qiskit.aqua import QuantumInstance

anz = RegularXYZAnsatz(2,2)
inst = QuantumInstance(Aer.get_backend('qasm_simulator'))

cst1 = GHZPauliCost(anz,inst)
cst2 = GHZWitness1Cost(anz,inst)

# Round 1: no errors
# ------------------

# test 1: can add  & subtract two cost objects
cst3 = cst1 + cst2
# check that the ansatz still equals
assert cst3.ansatz == cst1.ansatz
assert cst3.ansatz == cst2.ansatz
# check can keep adding new ones on
cst4 = cst3 + cst1
assert cst4.ansatz == cst1.ansatz
assert cst4.ansatz == cst2.ansatz
assert cst4.ansatz == cst3.ansatz
#
cst3 = cst1 - cst2
# check that the ansatz still equals
assert cst3.ansatz == cst1.ansatz
assert cst3.ansatz == cst2.ansatz
# check can keep adding new ones on
cst4 = cst3 - cst1
assert cst4.ansatz == cst1.ansatz
assert cst4.ansatz == cst2.ansatz
assert cst4.ansatz == cst3.ansatz

# test 2: can add & subtract a scalar
for scalar in [1,1.1,(1.1+0.1j)]:
    cst5 = scalar + cst1
    cst6 = cst1 + scalar
    assert cst5.ansatz == cst1.ansatz
    assert cst6.ansatz == cst1.ansatz
    cst5 = scalar - cst1
    cst6 = cst1 - scalar
    assert cst5.ansatz == cst1.ansatz
    assert cst6.ansatz == cst1.ansatz
    
# test 2: can multiply by a scalar
for scalar in [1,1.1,(1.1+0.1j)]:
    cst5 = scalar * cst1
    cst6 = cst1 * scalar
    assert cst5.ansatz == cst1.ansatz
    assert cst6.ansatz == cst1.ansatz

# Round 2: sensible results
# -------------------------