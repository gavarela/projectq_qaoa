## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  Variational QUantum Eigensolver
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import projectq as pq
from projectq import MainEngine
from projectq.ops import QubitOperator, Measure, All, Rx, Ry

import numpy as np, cmath


class VQE(object):

    def __init__(self):
        ''' To-do... '''
        pass
        
    def run(self, ansatz, hamiltonian, draws = 10000, eng = None):
        ''' Main function. '''
        pass
        
    @staticmethod
    def parity(bitstring, relevant_bits):
        ''' Checks parity of the binary representation of \bitstring in the \relevant_bits. Returns +1 even parity, -1 otherwise. 
            
            Parameters:
            
            \bitstring : (int) number whose binary representation's parity we want to check.
            
            \relevant_bits : (list) list of integers representing the bits in \bitstring whose parity we want to check.
            
            '''
        
        # Build int whose binary representation is all 0s with 1s in the \relevant_bits
        ones_rel_bits = 0
        for bit in relevant_bits:
            ones_rel_bits |= 1 << bit
        
        # Count the parity of \bitstring in \relevant bits
        even_parity = bin(bitstring & ones_rel_bits).count('1') % 2 == 0
        
        return 1 if even_parity else -1
        
    def expectation(self, prep_state, operator, draws = 10000, engine = None):
        ''' Measures, by sampling \draws times, the expectation value of the operator \operator in the state prepared by applying \prep_state on the initial state of all 0s (|000...>).
        
            Note: currently assumes \prep_state has only one term of unit norm. There's a comment below where the assumption is used.
            
            Parameters:
            
            \prep_state : (function) takes in a QC engine, creates a qubit register and applies a series of operations on it. Returns the qubit register.
            
            \operator : (QubitOperator) operator whose expectation in state prepared by \prep_state we want to measure.
            
            \draws : (int) number of times we will sample to get expectation.
            
            \eng : (MainEngine) a quantum computing engine.
            
            '''
        
        # Check arguments
#        if not callable(prep_state):
#            raise TypeError('Argument `prep_state` provided to expectation must be a function of one argument - a QC engine.')
            
        if not isinstance(operator, QubitOperator):
            raise TypeError('Argument `prog` provided to expectation must be a QubitOperator.')
        
        if not isinstance(draws, int):
            raise TypeError('Argument `draws` provided to expectation must be an integer.')
        
        if draws <= 0:
            raise ValueError('Argument `draws` provided to expectation must be positive.')
        
        engine = engine or MainEngine()
        if not isinstance(engine, MainEngine):
            raise TypeError('Argument `engine` provided to expectation must be a MainEngine.')
        
        # Get expectation, term by term
        ''' Each term in the operator will be made of Is, Xs, Ys and Zs. Measuring X is the same as rotating -pi/2 around Y and measuring Z. Measuring Y is the same as rotating +pi/2 around X and measuring Z. And measuring Z is just measuring Z. 
            We build the expectation by applying the relevant rotations to the prepared state, then measure in the Z basis. Repeating this process many times builds a sample we can calculate expectations from. '''
        expectation = 0
        for term, coef in operator.terms.items():
            
            rotations = []
            qubits_of_interest = []
            
            # If term is identity, add one to expectation
            if term == (): 
                expectation += coef
                continue
            
            # Else, get corresponding rotations
            for qubit, op in term:
                
                qubits_of_interest.append(qubit)
                
                if op == 'X':
                    rotations.append((Ry(-np.pi/2), qubit))
                elif op == 'Y':
                    rotations.append((Rx(np.pi/2), qubit))
            
            # Get expectation from sampling
            results = {}
            for _ in range(draws):
                
                # Prepare state
                qureg = prep_state(engine)

                # Apply rotations
                for rot, q in reversed(rotations):
                    rot | qureg[q]

                # Measure
                All(Measure) | qureg
                engine.flush()

                # Results
                result = int(''.join(str(int(q)) for q in reversed(qureg)), base = 2)
                results[result] = results.get(result, 0) + 1

            # Process results
            ''' Our result is a string of 0s and 1s, for every qubit used. The result is the product of the eigenvalue of each qubit of interest (+1 for measured 0, -1 for measured 1). Thus, we count the 1s measured in our qubits of interest and check whether there's an even or odd number of them. This is equivalent to checking the parity of the result in the qubits of interest. 
                Thus, if parity is even, result is +1, else -1. We add the corresponding result to the expectation.
                '''
            for result, count in results.items():

                # Get parity
                parity = self.parity(result, qubits_of_interest)

                # Add to expectation
                expectation += parity * coef * count / draws
        
        return expectation

## Test
## ~~~~

if __name__ == "__main__":
    
    from projectq.ops import H, X
    import time
    
    # First define a state_prep function. This should take our initialized state from
    # |0 0 > ----> |+ 1 >
    def state_prep(eng):
        qureg = eng.allocate_qureg(2)
        H | qureg[0]
        X | qureg[1]
        return qureg
    
    # Now let's create some example Hamiltonian. We'll use the Hamiltonian
    # X*Z, where Z acts on the first qubit, Z acts on the second qubit
    hamiltonian_example = 3 * QubitOperator('X0 Z1') + QubitOperator('Y1') + 2 * QubitOperator('Z1')
    
    # Let's just create an engine instance
    eng_example = MainEngine()
    
    # All of the above should result in an expectation value of -5.0. Let's check it out:
    start = time.time()
    exp = VQE().expectation(state_prep, hamiltonian_example, 1000, eng_example)
    print('\nExpectation (calculated in', time.time() - start, 'seconds):\n', exp, '\n')
            