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
        evenn_parity = bin(bitstring & ones_rel_bits).count('1') % 2 == 0
        
        return 1 if even_parity else -1
        
    @staticmethod
    def get_terms(qubit_op):
        ''' When a QubitOperator object has multiple terms or has a coefficient, we can't just apply it to a qubit register as:
            QubitOperator | qureg
            Thus, here we separate the provided QubitOperator, \quibit_op, into terms and those terms into (coefficient, operator), where operator has one term and unit norm. 
            Returns list of such tuples.
            
            Parameters:
            
            \qubit_op : (QubitOperator) operator we want to separate into terms of coefficient and operator. 
            
            '''
        
        terms = []
        for term, coef in qubit_op.terms.items():
            
            # Add each operator in order to a string
            t = ''
            for subterm in term:
                t += subterm[1]+str(subterm[0]) + ' '
            
            # Remove trailing space
            t = t[:-1]
            
            # Append
            terms.append((coef, QubitOperator(t)))
        
        return terms
    
    def expectation(self, prep_state, operator, draws = 10000, engine = None):
        ''' Measures, by sampling \draws times, the expectation value of the operator \operator in the state prepared by applying \prep_state on the initial state of all 0s (|000...>).
        
            Note: currently assumes \prep_state has only one term of unit norm. There's a comment below where the assumption is used.
            
            Parameters:
            
            \prep_state : (QubitOperator) preparation of the state we want to measure in.
            
            \operator : (QubitOperator) operator whose expectation in state prepared by \prep_state we want to measure.
            
            \draws : (int) number of times we will sample to get expectation.
            
            \eng : (MainEngine) a quantum computing engine.
            
            '''
        
        # Check types
        if not isinstance(prep_state, QubitOperator):
            raise TypeError('Argument `prog` provided to expectation must be a QubitOperator.')
            
        if not isinstance(operator, QubitOperator):
            raise TypeError('Argument `prog` provided to expectation must be a QubitOperator.')
        
        if not isinstance(draws, int):
            raise TypeError('Argument `draws` provided to expectation must be an integer.')
        
        engine = engine or MainEngine()
        if not isinstance(engine, MainEngine):
            raise TypeError('Argument `engine` provided to expectation must be a MainEngine.')
        
        # Split QubitOperators into terms
        prep_terms = self.get_terms(prep_state)
        
            ''' Note coefs may be complex below. '''
        prep_normalisation = [(complex(coef).conjugate() * coef).real \
                              for coef, term in prep_terms]
        prep_normalisation = sum(prep_normalisation)**0.5
        prep_normalisation = 1/prep_normalisation
        
        op_terms = self.get_terms(operator)
        
        # Get highest-numbered qubit for making qureg
        operations = [term for coef, term in prep_terms]
        operations += [term for coef, term in op_terms]
        max_qubit = max([qubit for term in operations for qubit, _ in term.terms.keys()])
        
        # Get expectation
        ''' Our \operator whose expectation we want to measure will be made of Is, Xs, Ys and Zs. Measuring X is the same as rotating -pi/2 around Y and measuring Z. Measuring Y is the same as rotating +pi/2 around X and measuring Z. And measuring Z is just measuring Z. 
            We build the expectation by applying the relevant rotations to the prepared state, then measure in the Z basis. Repeating this process many times builds a sample we can calculate expectations from.
            '''
        expectation = 0
        for coef, term in op_terms:
            
            rotations = []
            qubits_of_interest = []
            ops = list(term.terms.keys())[0]
            
            # If term is identity, add one to expectation
            if ops == (): 
                expectation += coef
                continue
            
            # Else, get corresponding rotations
            for qubit, op in ops:
                
                qubits_of_interest.append(qubit)
                
                if op == 'X':
                    rotations.append((Ry(-np.pi/2), qubit))
                elif op == 'Y':
                    rotations.append((Rx(np.pi/2), qubit))
            
            # Sample
            ''' Assumes that \prep_state has only one term of coefficient 1. How would we procede otherwise? Will the `otherwise` case ever come up? '''
            for prep_coef, prep_term in prep_terms:
                
                sq_prep_coef = (complex(prep_coef).conjugate() * prep_coef).real
                
                results = {}
                for _ in range(draws):

                    eng = engine()
                    qureg = eng.allocate_qureg(max_qubit + 1)

                    # Prepare system for measurement
                    prep_term | qureg
                    for rot, q in rotations:
                        rot | qureg[q]

                    # Measure
                    All(Measure) | qureg
                    eng.flush()

                    # Results
                    result = [int(q) for q in qureg]
                    results[result] = results.get(result, 0) + 1

                # Process results
                ''' Our result is a string of 0s and 1s, for every qubit used. The result is the product of the eigenvalue of each qubit of interest (+1 for measured 0, -1 for measured 1). Thus, we count the 1s measured in our qubits of interest and check whether there's an even or odd number of them. This is equivalent to checking the parity of the result in the qubits of interest. 
                    Thus, if parity is even, result is +1, else -1. We add the corresponding result to the expectation.
                    '''
                for result, count in results.items():

                    # Get parity
                    int_result = int(''.join(str(q) for q in reversed(result)), base = 2)
                    parity = self.parity(int_result, qubits_of_interest)
                    
                    frq = count / draws
                    
                    # Add to expectation
                    expectation += parity * coef * (sq_prep_coef / prep_normalisation) * freq
        
        return expectation


## Test
## ~~~~

if __name__ == "__main__":
    
    # 0.3 * X | q1 should prep state whose expectation is 0.3 so I'll test that
    vqe = VQE()
    exp = vqe.expectation_naive(0.3 * QubitOperator('X0'))
    print('Expectation:', exp)
    
    assert exp == 0.3          # Doesn't work but close enough
            