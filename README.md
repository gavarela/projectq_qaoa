# projectq_qaoa
Build a QAOA module in Project Q language, including a VQE for internal use.

## Files
`QAOA.py` is the main file here. This contains a class for the Quantum Approximate Optimization Algorithm (QAOA). The user inputs a cost and mixer hamiltonian, an initial quantum state, a number of Trotterization steps, and the number of draws one wants to perform

`VQE.py` contains a Variational Quantum Eigensolver (VQE) class. The user inputs an initial quantum state, a hamiltonian whose lowest eigensate's eigenvalue we want to approximate, and the number of draws one wants to perform

`GraphToHamiltonian.py` contains a bunch of functions for converting user-inputted graphs into hamiltonians for use in QAOA, VQE, etc. Right now the graphs must be networkx graphs, but we will add functionality for inputting lists of tuples defining the edges and nodes of a graph
