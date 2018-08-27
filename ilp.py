#- -*- coding: utf-8
# import gurobipy
import networkx as nx
import numpy as np

GUROBI_SUCCESS_STATE = [
    #gurobipy.GRB.OPTIMAL,
    #gurobipy.GRB.ITERATION_LIMIT,
    #gurobipy.GRB.NODE_LIMIT,
    #gurobipy.GRB.TIME_LIMIT,
    #gurobipy.GRB.SOLUTION_LIMIT
]


def evaluate_model(g, rqs, inventory, actions, degree_constraint,
                   timeout=None, set_initial_requests=False, edge_capacity=1,
                   keep_static=False):
    """
    ILP from paper. calculate optimal solution.

    Args:
        g (nx.Graph): Initial topology that should be evaluated.
        rqs (list): Of requests.Request objects.
        inventory (int): Maximum number of edges/capacity available to the ILP.
        actions (int): The maximum number of edges available to the ILP.
        degree_constraint (int): The maximum number of incident capacity to
            each node.
        timeout (int, optional): Maximum number of solving time in seconds.

    Returns:
        generated_graph (nx.DiGraph): Graph resulting from optimization
            containing also the mapping of requests to edges.
        num_accepted (int): Number of accepted requests.
    """
    import gurobipy
    if actions > inventory and actions % 2 != 0:
        actions -= 1
    m = gurobipy.Model("ilp")
    m.setParam("LogToConsole", 0)
    vars = {}
    gamma = 1e9
    # Create and add all necessary variables. The tmp prefixed and the delta
    # variable are needed to model the absolute values in constraint 6
    for u in g.nodes():
        for v in g.nodes():
            if u == v:
                continue
            # Needed to model absolute values
            n = "tmp_(p_({:d},{:d}))".format(u, v)
            vars[n] = m.addVar(name=n, vtype=gurobipy.GRB.INTEGER, lb=0)
            n = "tmp_(k_({:d},{:d}))".format(u, v)
            vars[n] = m.addVar(name=n, vtype=gurobipy.GRB.INTEGER, lb=0)
            # n = "tmp_({:d},{:d})".format(u, v)
            # vars[n] = m.addVar(name=n, vtype=gurobipy.GRB.INTEGER)
            n = "delta_({:d},{:d})".format(u, v)
            vars[n] = m.addVar(name=n, vtype=gurobipy.GRB.BINARY)

            name = "p_({:d},{:d})".format(u, v)
            vars[name] = m.addVar(
                name=name,
                vtype=gurobipy.GRB.INTEGER,
                lb=0
            )
            if g.has_edge(u, v):
                vars["k_({:d},{:d})".format(u, v)] = 1
                vars[name].start = int(g.edges[u, v]["capacity"] / edge_capacity)
            else:
                vars["k_({:d},{:d})".format(u, v)] = 0
                vars[name].start = 0

            for request in rqs:
                name = "x_({:d},{:d},{:d})".format(request.mask, u, v)
                vars[name] = m.addVar(
                    name=name,
                    vtype=gurobipy.GRB.BINARY
                )

    for request in rqs:
        r_d = "r_{:d}".format(request.mask)
        vars[r_d] = m.addVar(name=r_d, vtype=gurobipy.GRB.BINARY)
        if set_initial_requests:
            if g.has_edge(request.source, request.target):
                vars[r_d].start = 1

    m.update()

    # Flow Conservation Constraint 2
    for request in rqs:
        for i in g.nodes():
            linexpr = None
            r_d = "r_{:d}".format(request.mask)
            for j in g.nodes():
                if i == j:
                    continue
                x_dij = "x_({:d},{:d},{:d})".format(request.mask, i, j)
                x_dji = "x_({:d},{:d},{:d})".format(request.mask, j, i)
                if linexpr is None:
                    linexpr = vars[x_dij] - vars[x_dji]
                else:
                    linexpr += vars[x_dij] - vars[x_dji]
            if i == request.source:
                tmp = vars[r_d]
            elif i == request.target:
                tmp = -vars[r_d]
            else:
                tmp = 0
            m.addConstr(linexpr == tmp)

    # Constraint 3
    for i in g.nodes():
        for j in g.nodes():
            if i == j:
                continue
            linexpr = None
            for request in rqs:
                x_dij = "x_({:d},{:d},{:d})".format(request.mask, i, j)
                x_dji = "x_({:d},{:d},{:d})".format(request.mask, j, i)
                if linexpr is None:
                    linexpr = vars[x_dij] + vars[x_dji]
                else:
                    linexpr += vars[x_dij] + vars[x_dji]
            m.addConstr(linexpr <= edge_capacity * vars["p_({:d},{:d})".format(i, j)])

    # Adding constraints 4, 5, 6. For Constraint six we need additional binary
    # constraints to model the absolute value
    constraint5 = None
    constraint6 = None
    for i in g.nodes():
        constraint4 = None
        for j in g.nodes():
            if i == j:
                continue
            p_ij = "p_({:d},{:d})".format(i, j)
            p_ji = "p_({:d},{:d})".format(j, i)
            k_ij = "k_({:d},{:d})".format(i, j)
            tmp_p_ij = "tmp_(p_({:d},{:d}))".format(i, j)
            tmp_k_ij = "tmp_(k_({:d},{:d}))".format(i, j)
            # tmp_ij = "tmp_({:d},{:d})".format(i, j)
            delta_ij = "delta_({:d},{:d})".format(i,j)

            m.addConstr(
                vars[tmp_p_ij] - vars[tmp_k_ij] == vars[p_ij] - vars[k_ij]
            )
            # m.addConstr(vars[tmp_ij] == vars[tmp_p_ij] + vars[tmp_k_ij])
            m.addConstr(vars[tmp_p_ij] <= vars[delta_ij] * degree_constraint)
            m.addConstr(vars[tmp_k_ij] <= (1 - vars[delta_ij]) * degree_constraint)
            if keep_static:
                if g.has_edge(i, j):
                    m.addConstr(vars[p_ij] >= 1)
            # m.addConstr(vars[tmp_k_ij] >= 0)
            # m.addConstr(vars[tmp_p_ij] >= 0)
            if constraint6 is None:
                constraint6 = vars[tmp_p_ij] + vars[tmp_k_ij]
            else:
                constraint6 += vars[tmp_p_ij] + vars[tmp_k_ij]

            m.addConstr(vars[p_ij] == vars[p_ji])

            if constraint4 is None:
                constraint4 = vars[p_ij] - vars[k_ij]
            else:
                constraint4 += vars[p_ij] - vars[k_ij]

            if constraint5 is None:
                constraint5 = vars[p_ij] - vars[k_ij]
            else:
                constraint5 += vars[p_ij] - vars[k_ij]
        m.addConstr(constraint4 <= degree_constraint)
    m.addConstr(constraint5 <= 2 * inventory)
    m.addConstr(constraint6 <= 2 * actions)

    # Create objective function
    objective = None
    for request in rqs:
        for i in g.nodes():
            for j in g.nodes():
                if i == j:
                    continue
                x_dij = "x_({:d},{:d},{:d})".format(request.mask, i, j)
                if objective is None:
                    objective = vars[x_dij]
                else:
                    objective += vars[x_dij]
        objective -= gamma * vars["r_{:d}".format(request.mask)]
    m.setObjective(objective, gurobipy.GRB.MINIMIZE)
    if timeout is not None:
        m.setParam('TimeLimit', timeout)
    m.optimize()

    if m.status in GUROBI_SUCCESS_STATE:
        accepted = []
        generated_graph = nx.DiGraph()
        generated_graph.add_nodes_from(g.nodes())
        for i in g.nodes():
            for j in g.nodes():
                if i == j:
                    continue
                p_ij = "p_({:d},{:d})".format(i, j)
                if vars[p_ij].X > 0:
                    generated_graph.add_edge(
                            i,
                            j,
                            capacity=vars[p_ij].X * edge_capacity,
                            allocated_capacity=0,
                            requests=[]
                            )
                for request in rqs:
                    x_dij = "x_({:d},{:d},{:d})".format(request.mask, i, j)
                    if vars[x_dij].X == 1:
                        generated_graph.edges[i, j]["allocated_capacity"] += 1
                        generated_graph.edges[i, j]["requests"].append(request.mask)

        for request in rqs:
            r_d = "r_{:d}".format(request.mask)
            if vars[r_d].X == 1:
                accepted.append(request.mask)
        return generated_graph, accepted
    else:
        return None, None

#    for u, v, d in generated_graph.edges(data=True):
#        print u, v, d
#    print "Accepted Requets: ", num_accepted
