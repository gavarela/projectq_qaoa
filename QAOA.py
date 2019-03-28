## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  Quantum Approximate Optimisation Algorithm
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

''' Inspired by Grove's pyQuil implementation. '''

import projectq as pq
from projectq import MainEngine
from projectq.ops import QubitOperator, Measure, All, H, TimeEvolution

import numpy as np, cmath
from scipy.optimize import minimize

from VQE import VQE


class QAOA(object):
    
    def __init__(self, cost, mixer = None, init_state = None,
                 n_steps = 1,
                 engine = MainEngine, n_qubits = None,
                 init_betas = [], init_gammas = [],
                 minimiser = minimize, 
                 min_args = [], min_kwargs = {'method': 'Nelder-Mead'},
                 vqe_run_kwargs = {}):
        ''' Sets up parameters, given default values, for use in main QAOA algorithm.
        
        Parameters:
        
        \cost: (QubitOperator) the cost function (in operator form) we want to maximise.
        
        \mixer: (QubitOperator) the operator whose highest energy state we want to start in for the QAOA algorithm.
        
        \init_state: (function: engine -> qureg) function that prepares the highest energy state of the \mixer operator. Should return a qubit register created using the provided engine with the state prepared.
        
        \n_steps: (int > 0) number of steps to take in mixer -> cost evolution in QAOA algorithm.
        
        \engine: (?) an engine to use.
        
        \n_qubits: (int > 0) number of qubits needed.
        
        \init_betas: (list: len \n_steps) list of the initial guesses for betas (exponential coefficients for the mixer function).
        
        \init_gammas: (list: len \n_steps) list of the initial guesses for gammas (exponential coefficients for the cost function).
        
        \minimiser: (function(objective, x0, *args, **kwargs)) minimiser to be used in VQE - see VQE comments for more detail. 
        Additional requirement: result returned must contain final (optimised) parameter values as an attribute `x`.
        
        \min_args: (list) arguments to be passed to minimiser as *args - see VQE comments for more detail.
        
        \min_kwargs: (dict) arguments to be passed to minimiser as **kwargs - see VQE comments for more detail.
        
        \vqe_run_kwargs: (dict) arguments to be passed to VQE run() method as **kwargs. Can include `draws`, `verbose` and `callback` methods. See VQE comments for more detail.
        
        
        '''
        
        # Handle arguments' defaults
        if n_qubits is None:
            cost_terms = list(cost.terms.items())
            qubits = set()
            for term in cost_terms:
                for qubit, op in term[0]: qubits.add(qubit)
            n_qubits = max(qubits) + 1
        
        self.n_qubits = n_qubits
        
        def default_init_state(eng):
            qureg = eng.allocate_qureg(self.n_qubits)
            All(H) | qureg
            return qureg
        
        self.init_state = init_state or default_init_state
        self.mixer = mixer or QubitOperator(' '.join(['X' + str(i) for i in range(self.n_qubits)]))
        
        self.n_steps = n_steps
        self.betas = np.array(init_betas) or np.random.uniform(0, np.pi, self.n_steps)
        self.gammas = np.array(init_gammas) or np.random.uniform(0, 2*np.pi, self.n_steps)
        
        # Store values of arguments
        self.cost = cost
        
        self.engine = engine 
        
        self.minimiser = minimiser
        self.min_args = min_args
        self.min_kwargs = min_kwargs
        
        self.vqe_run_kwargs = vqe_run_kwargs
        
        # Check types
        types = [QubitOperator, QubitOperator, int, MainEngine, int]#, np.ndarray, np.ndarray]
        args = [self.cost, self.mixer, self.n_steps, 
                self.engine, self.n_qubits]#, 
        #        self.betas, self.gammas]
        argstr = ['cost', 'mixer', 'n_steps', 'engine', 'n_qubits']#, 
                  #'init_betas', 'init_gammas']
        
        for i, arg in enumerate(args):
            if not isinstance(arg, types[i]):
                raise TypeError('Argument `%s` provided to expectation must be a %s' % (argstr[i], types[i].__name__))
        
        if self.n_steps <= 0: raise ValueError('Argument `n_steps` must be positive.')
        if self.n_qubits <= 0: raise ValueError('Argument `n_qubits` must be positive.')
        
        if len(self.betas) != self.n_steps: raise ValueError('Argument `init_betas` must be a list of length equal to argument `n_steps`.')
        if len(self.gammas) != self.n_steps: raise ValueError('Argument `init_gammas` must be a list of length equal to argument `n_steps`.')
            
        if not callable(self.init_state):
            raise TypeError('Argument `init_state` must be a function of a single parameter: a QC engine.')
        
        for key in self.vqe_run_kwargs:
            if key not in ('draws', 'verbose', 'callback'):
                raise ValueError('Argument `vqe_run_kwargs` should only contain keys from ("draws", "verbose", "callback").')
    
    def _prep_state(self, params, eng):
        ''' \params must be of length 2*n_steps, where first half is betas and second half is gammas. 
            
            Prepares state by first applying 
                |psi> = self.init_state |0...0> 
            then doing 
                Prod U(beta, B)U(gamma, C) |psi>.
            '''
        
        betas  = params[:self.n_steps]
        gammas = params[self.n_steps:]
        
        qureg = self.init_state(eng)
        
        for i in range(self.n_steps):
            TimeEvolution(gammas[i], self.cost ) | qureg
            TimeEvolution(betas[i] , self.mixer) | qureg
        
        return qureg
        
    def solve_angles(self):
        ''' Uses VQE to solve for the optimal angles [betas, gammas]. '''
        
        params = list(self.betas) + list(self.gammas)
        vqe = VQE(self.minimiser, self.min_args, self.min_kwargs)
        
        self.result = vqe.run(ansatz = self._prep_state,
                              hamiltonian = self.cost,
                              initial_params = params,
                              engine = self.engine,
                              **self.vqe_run_kwargs)
        
        self.betas  = self.result.x[:self.n_steps]
        self.gammas = self.result.x[self.n_steps:]
        
        return self.betas, self.gammas
    
    def likely_string(self, draws = 500):
        ''' Given the stored beta and gamma parameters (given initial ones or optimised ones if ran solve_angles()), samples prepared state to get most likely qubit string. '''
        
        # Check types
        if not isisntance(draws, int):
            raise TypeError('Argument `draws` must be a positive integer.')
        if draws <= 0:
            raise ValueError('Argument `draws` must be a positive integer.')
        
        # Sample
        params = list(self.betas) + list(self.gammas)
        
        strings = {}
        for i in range(draws):
            
            qureg = self._prep_state(params, self.engine)
            All(Measure) | qureg
            
            self.engine.flush()
            string = ''.join([str(int(q)) for q in reversed(qureg)])
            
            strings[string] = strings.get(string, 0) + 1
        
        # Return most common
        res = max(strings.keys(), key = lambda s: strings[s])
        return res, strings[res]
        

## Test
## ~~~~

if __name__ == "__main__":

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # For this test, let's do a simple MaxCut example with the following graph:
    # 0-1-2
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Just import networkx, useful for making graphs with edges and nodes
    # import time to time our operations
    import networkx as nx
    import time

    # Construct a graph with n nodes
    n = 3
    G = nx.Graph()
    G.add_nodes_from(list(range(0, n, 1)))

    # tuple is (i, j) where (i, j) is the edge
    # This line just tells our graph how the edges are connected to each other
    edge_list = [(0, 1), (1, 2)]

    # Feed the edges to our graph:
    G.add_edges_from(edge_list)

    # Now let's make our cost function (mixer handled as default argument already for QAOA MaxCut)
    cost_example = QubitOperator('', 0.5) + QubitOperator('Z0 Z1', -0.5) + QubitOperator('', 0.5) + QubitOperator('Z1 Z2', -0.5)


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Put it all together, create a QAOA instance
    # Use default initil state and mixer, put in our own cost
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    eng_example = MainEngine()
    inst = QAOA(cost = cost_example, engine = eng_example, n_qubits = 3, vqe_run_kwargs = {'verbose': 1, 'draws': 100})
    
    start = time.time()
    beta, gamma = inst.solve_angles()
    end = time.time()
    print("Beta: {}, Gamma: {}, Time: {}".format(beta, gamma, end - start))
    
    
    
    
    
    
    
    
    
    
    
