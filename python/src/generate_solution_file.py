
from pysat.formula import CNF
from pysat.solvers import Glucose3
import pickle

cnf = CNF(from_file="100_sat.txt")
solver_result = Glucose3(cnf)
if solver_result.solve():
    sol = solver_result.get_model()
    with open("./100_sat_sol.pkl", "wb") as file:
        pickle.dump(sol, file)