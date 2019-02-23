## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  Variational QUantum Eigensolver
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


from projectq import MainEngine
from projectq.ops import QubitOperator, Measure, All

import numpy as np


class VQE(object):

    def __init__(self):
        ''' To-do... '''
        
        
    def run(self, ansatz, hamiltonian, draws = 10000, eng = None):
        ''' Main function. '''
        pass
        
    @staticmethod
    def expectation_naive(prog, draws = 10000, eng = None):
        ''' Takes in a program, \prog, that prepares the state we want to take the expectation of and measures it. Does this \draws times and returns the mean of the outcomes.
            
            Should we take in a program to prepare state and also an operator we want to take the expectation of under that state? The only difference below (I think) would be that we would have to do 
                hamiltonian | q
            after we do 
                term | q
            in the code below.
            Or is this the way to go?
            
            Also, I'm taking the expectation of the number that comes out when we write the qubits in binary. Should we be doing something else?
            
            I'm calling this naive because I was thinking we could later do another expectation function the way pyquil does which is more involved but I suspect is faster - they do rotations and then measure stuff and use that parity shit.
            '''
        
        # Check types
        if not isinstance(prog, QubitOperator):
            raise TypeError('Argument `prog` provided to expectation_naive must be a QubitOperator.')
        
        if not isinstance(draws, int):
            raise TypeError('Argument `draws` provided to expectation_naive must be an integer.')
        
        eng = eng or MainEngine()
        if not isinstance(eng, MainEngine):
            raise TypeError('Argument `eng` provided to expectation_naive must be a MainEngine.')
        
        # Turn each term into its own QubitOperator
        ''' Bc if QubitOperator has many terms or has a coefficient, we can't just apply it to the qubit register so I separate the terms and coefs. '''
        terms = []
        for term, coef in prog.terms.items():
            
            t = ''
            for subterm in term:
                t += subterm[1]+str(subterm[0]) + ' '
            t = t[:-1]
            
            terms.append((coef, QubitOperator(t)))
            
        # Get highest-numbered qubit for making qureg
        operations = []
        for term in prog.terms.keys():
            operations += list(term)
        
        qubits = set([operation[0] for operation in operations])
        max_qubit = max(qubits)
        
        # Take expectations
        measurements = []
        for _ in range(draws):
            
            measurement = 0
            for coef, term in terms:
                
                q = eng.allocate_qureg(max_qubit+1)
                
                # Prepare state and measure
                term | q
                All(Measure) | q
                
                # Flush and add to counter
                eng.flush()
                str_result = ''.join([str(int(qubit)) for qubit in reversed(q)])
                measurement += coef * int(str_result)
                
            measurements.append(measurement)
            
        return sum(measurements)/len(measurements)


## Test
## ~~~~

if __name__ == "__main__":
    
    # 0.3 * X | q1 should prep state whose expectation is 0.3 so I'll test that
    vqe = VQE()
    exp = vqe.expectation_naive(0.3 * QubitOperator('X0'))
    print('Expectation:', exp)
    
    assert exp == 0.3          # Doesn't work but close enough
            