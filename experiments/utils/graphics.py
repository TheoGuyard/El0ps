def get_solver_name_color(solver_name):
    if solver_name == "mosek":
        return "royalblue"
    elif solver_name == "gurobi":
        return "forestgreen"
    elif solver_name == "l0bnb":
        return "orange"
    elif solver_name == "el0ps":
        return "tomato"
    else:
        return None