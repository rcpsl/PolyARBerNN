# PolyARBerNN

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

3. Solver
-----------------
PolyARBerNN is capable of solving a general multivariate polynomial constraints defined over a box using a neural network guided abstraction refinement procedure. The NN model used for the abtraction refinement is located in the "model" folder.

a) Define the box:

    num_vars = 2
    
    x_min = -1.0
    x_max = 1.0
    
    
    box=np.array(hypercube(num_vars, x_min,x_max))
    polype=pc.box2poly(box)
    boxx=pc.bounding_box(polype)
    pregion=[[{'A':polype.A,'b':polype.b}]]

b) Define the multivariate polynomial:
   
    # poly = 4 x^2 + 3 y^2 + 2
    poly = [
    {'coeff':4,      'vars':[{'power':2},{'power':0}]},
    {'coeff':3,    'vars':[{'power':0},{'power':2}]},
    {'coeff':2,    'vars':[{'power':0},{'power':0}]}
    ]

c) Define the orders for this polynomial: the maximum powers that x and y can take:

orders = [[2, 2]]

d) Call the solver which takes as parameters the number of variables, the box, the region, and the orders: num_vars, boxx, pregion, orders:

solver = PolyInequalitySolver(num_vars, boxx, pregion, orders)


e) Add the multivariate polynomial constraint, i.e., poly <= 0, to our solver PolyARBerNN:

   solver.addPolyInequalityConstraint(poly)

f) Run the solver:

   res=solver.solve()   


All these steps can be found in the main section of the code PolyARBerNN.py.

  
3. Running the code
---------------------------------------------------
To run the solver, you can use the following command in a terminal:

Run > python3 PolyARBerNN.py
