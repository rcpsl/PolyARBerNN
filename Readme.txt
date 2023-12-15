PolyARBerNN: A Neural Network Guided Solver and Optimizer for Bounded Polynomial Inequalities

1. Introduction
-----------------
This Python package contains the implementation of the algorithms described
in the paper "PolyARBerNN: A Neural Network Guided Solver and Optimizer for Bounded Polynomial Inequalities", Wael Fatnassi, Yasser Shoukry, ACM Trans. Embedd. Comput. Syst 2023. This file describes the contents of the package, and provides instructions regarding its use. 


2. Installation
-----------------
The tool was written for Python 3.7.6. Earlier versions may be sufficient, but not tested. In addition to Python 3.7.6, the solver requires the following:

- Z3 4.8.9 solver: pip install z3-solver
- Yices 2.6.2 solver: Please follow the instruction at this URL (https://yices.csl.sri.com) 
- scipy: pip install scipy
- autograd: pip install autograd 
- numpy: pip install numpy
- polytope: pip install polytope
- sympy: pip install sympy
- matplotlib: pip install matplotlib
- cvxopt: pip install cvxopt
- libpoly: Please follow the instruction at this URL (https://github.com/SRI-CSL/libpoly)
- gmp: brew install gmp 



3. Running the code
---------------------------------------------------
To run the solver, you can use the following command in a terminal:



Run > python3 PolyARBerNN.py

Outputs: UNSAT





 
