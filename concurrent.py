import numpy as np
import networkx as nx
import requests
import environment
import agents
import ilp


def embedd_requests_with_ilp(topo, rqs, degree_constraint):
    """
    Given a set of request find the maximum number of requests that fit into
    the topology.
    
    Args:
        topo (nx.Graph): Topology on which requests should be fit.
        ret (list): List of requests.Request objects.
        degree_constraint (int): Maximum degree of each node.
        
    Returns:
        embedded_requests (list): List of request ids.
    """
    _, ret = ilp.evaluate_model(
            g=topo,
            rqs=rqs,
            inventory=0,
            actions=0,
            degree_constraint=degree_constraint,
            timeout=None,
            set_initial_requests=True
            )
    return ret


def evaluate_sequence(args):
    num_nodes = args["num_nodes"]
    edge_data = args["edge_data"]
    seed = args["seed"]
    horizon = args["horizon"]
    inventory = args["inventory"]
    isize = args["inventory_size"]
    request_data = args["request_data"]
    num_sequences = args["num_sequences"]
    max_degree = args["max_degree"] if "max_degree" in args else None
    subsample = args["subsample"] if "subsample" in args else None
    sensor_threshold = args["sensor_threshold"] if "sensor_threshold" in args else None
    agent_type = args["type"]
    base_degrees = args["base_degrees"]
    with_ilp = args["with_ilp"]

    random = np.random.RandomState(seed=seed)
    random_shuffler = np.random.RandomState(seed=100)

    rqs = []
    if type(request_data[0]) == tuple:
        for u, v, i in request_data:
            rqs.append(requests.Request(u, v, i))
    else:
        for i, tm in enumerate(request_data):
            rqs.append(requests.TrafficMatrixRequest(tm, i))

    states = []
    average_embedded = 0
    counts = []
    for _ in range(num_sequences):
        topo = nx.Graph()
        topo.add_nodes_from(range(num_nodes))
        topo.add_edges_from(edge_data)
        env = environment.Environment(topo, max_degree, base_degrees)

        seed = random.randint(0, 1000000)
        agent = agents.AgentFactory.produce(
            order=agent_type,
            horizon=horizon,
            inventory_size=isize,
            requests=rqs,
            environment=env,
            subsample=subsample,
            num_sequences=num_sequences,
            seed=seed
        )
        agent.inventory = inventory
        agent.sensing_threshold = sensor_threshold

        actions = []
        for a in agent.random_action_sequence():
            a.apply()
            actions.append(a)

        current_topo = agent.environment.topology.copy()
        order = np.arange(len(rqs))

        max_count = None
        max_sensor = None
        if with_ilp:
            agent.environment = environment.Environment(
                current_topo.copy(),
                degree_limit=max_degree,
                base_degree=base_degrees
            )
            result_rqs = embedd_requests_with_ilp(
                    topo=agent.environment.topology,
                    rqs=rqs,
                    degree_constraint=max_degree
                    )
            for mask in result_rqs:
                agent.environment.routed_requests[mask] = []
            max_count = len(result_rqs)
            max_sensor = agent.sense()
        else:
            if type(rqs[0]) == requests.Request:
                for i in range(10):
                    count = 0
                    if subsample is None:
                        random_shuffler.shuffle(order)
                        sample = order
                    else:
                        sample = random_shuffler.choice(
                                order,
                                replace=False,
                                size=subsample
                                )
                    agent.environment = environment.Environment(
                        current_topo.copy(),
                        degree_limit=max_degree,
                        base_degree=base_degrees
                    )

                    for i, idx in enumerate(sample):
                        p = agent.environment.add_request(rqs[idx])
                        if p is not None:
                            count += 1

                    if count > max_count or max_count is None:
                        max_count = count
                        max_sensor = agent.sense()
            else:
                save = {}
                for i, r in enumerate(rqs):
                    p = agent.environment.add_request(r)
                    if p is not None:
                        save[r.mask] = []
                    agent.environment.reset_topology()
                max_count = len(save)
                agent.environment.routed_requests = save
                max_sensor = agent.sense()
        average_embedded += max_count
        counts.append(max_count)
        agent.add_new_reading(max_sensor, states)
    if len(states) == 0:
        states.append(-1)
    return [states, float(average_embedded) / float(num_sequences)]
    


def fit_requests(topo, rqs, action_budget, capacity_budget,
                 original_degrees, degree_limit):
    """ 
    Finds the maximum number of requests that fit on a given topology. This
    function is intended only for the scenario of unit capacity per edge
    and unit demand per request.
    
    Args:
        topo (nx.topology): Graph to route requests on.
        requests (list): List of requests.Requst objects. Requests that should
            be routed.
        capacity_budget (int): Budget of capacity budget that can be used.
        original_degrees (dict): Node to original degree mapping. I.e., the
            degree the node had before mutating the network.
        degree_limit (int): Maximum number of edges that can be added to
            a node.

    Returns:
        fitted_requests (list): List of requests.Request objects that
            fit into graph.
    """
    fitted_request_idx = []
    not_fitted = []
    for request in rqs:
        if topo.has_edge(request.source, request.target):
            fitted_request_idx.append(request.mask)
            topo.edges[request.source, request.target]["allocated_capacity"] += 1
        else:
            not_fitted.append(request)

    if len(not_fitted) == 0:
        return fitted_request_idx

    capacity = 0
    for _, _, d in topo.edges(data=True):
        capacity += d["capacity"]

    if capacity < capacity_budget:
        for i in range(min(capacity_budget - capacity, action_budget)):
            action_budget -= 1
            r = not_fitted.pop()
            topo.add_edge(r.source,
                          r.target,
                          static=False,
                          capacity=1,
                          allocated_capacity=1
                          )
            fitted_request_idx.append(r.mask)

    if len(not_fitted) == 0:
        return fitted_request_idx

    env = environment.Environment(topo, degree_limit, original_degrees)
    if env.routable_topology.number_of_edges() == 0:
        return fitted_request_idx  # No free edges to route or move --> done

    routed_requests = []
    not_routed_requests = []
    paths = []
    for r in not_fitted:
        p = env.add_request(r)
        if p is None:
            not_routed_requests.append(r)
        else:
            routed_requests.append(r.mask)
            paths.append((r.mask, p))

    if len(not_routed_requests) == 0:
        fitted_request_idx.extend(routed_requests)
        return fitted_request_idx

    if action_budget < 2:
        fitted_request_idx.extend(routed_requests)
        return fitted_request_idx

    paths.sort(key=lambda x: len(x[1]))
    routable_edges = [(u, v) for u, v in env.routable_topology.edges()]
    while action_budget >= 2 and (len(routable_edges) > 0 or len(paths) > 0) \
             and len(not_routed_requests) > 0:
        r = not_routed_requests.pop()
        deg_u = nx.degree(topo, weight="capacity")[r.source]
        deg_v = nx.degree(topo, weight="capacity")[r.target]
        
        if deg_u == degree_limit:
            free_edge_u = None
            for i, j, d in topo.edges(nbunch=u, data=True):
                if d["capacity"] - d["allocated_capacity"] > 0:
                    free_edge_u = (i, j)
                    break
        if deg_v == degree_limit:
            free_edge_v = None
            for i, j, d in topo.edges(nbunch=v, data=True):
                if d["capacity"] - d["allocated_capacity"] > 0:
                    free_edge_v = (i, j)
                    break
                
        if deg_u == degree_limit and deg_v == degree_limit and action_budget >= 3:
            if free_edge_u is None or free_edge_v is None:
                continue
            else:
                topo.remove_edge(*free_edge_u)
                topo.remove_edge(*free_edge_v)
                topo.add_edge(r.source,
                              r.target,
                              capacity=1,
                              allocated_capacity=1,
                              static=False
                              )
                capacity_budget += 1
                action_budget -= 3        
                fitted_request_idx.append(r.mask)
        elif ((deg_u < degree_limit and deg_v == degree_limit)
            or (deg_v < degree_limit and deg_u == degree_limit)) and action_budget >= 2:
            if deg_u < degree_limit:
                free_edge_tmp = free_edge_v
                node = v
            else:
                free_edge_tmp = free_edge_u
                node = u
            #  There is not unfree edge. Find longest path any incident
            #  edge takes part in.
            if free_edge_tmp is None:
                p = None
                for i, j in topo.edges(nbunch=node):
                    for k, tmp in enumerate(paths):
                        for x, y in zip(tmp[1][:-1], tmp[1][1:]):
                            if (x == i and y == j) or (x == j and y == i):
                                if p is None or len(p[1][1]) < len(tmp[1]):
                                    p = (k, tmp)
                                    free_edge_tmp = (i, j)
                                break
                #  Remove path from paths and routed requetss
                if p is None:
                    continue
                routed_requests.remove(p[0])
                tmp = paths[:p[0]]
                if len(paths) > p[0] + 1:
                    tmp.extend(paths[p[0] + 1:])
                for k, l in zip(p[1][1][:-1], p[1][1][1:]):
                    topo.edges[k, l]["allocated_capacity"] = 0
                    if k != free_edge_tmp[0] and l != free_edge_tmp[1]:
                        routable_edges.append(k, l)
            topo.remove_edge(*free_edge_tmp)
            topo.add_edge(
                    r.source,
                    r.target,
                    capacity=1,
                    allocated_capacity=1,
                    static=False
                    )
            action_budget -= 2
            fitted_request_idx.append(r.mask)
        elif deg_u < degree_limit and deg_v < degree_limit:
            taken = 0
            if capacity_budget == 0:
                if len(routable_edges) == 0:
                    continue
                else:
                    topo.remove_edge(*routable_edges.pop())
                    taken = 1
            topo.add_edge(
                    r.source,
                    r.target,
                    capacity=1,
                    allocated_capacity=1,
                    static=False
                    )
            action_budget -= 1 + taken
            fitted_request_idx.append(r.mask)
        else:
            continue
    fitted_request_idx.extend(routed_requests)
    return fitted_request_idx


if __name__ == "__main__":
    g = nx.random_regular_graph(d=3, n=6, seed=1)
    reqs = [rqs.Request(u, v, i) for i, (u, v) in enumerate(g.edges())]
    for u, v, d in g.edges(data=True):
        d["capacity"] = 1
        d["allocated_capacity"] = 0
        d["static"] = False
    f=fit_requests(g.copy(), reqs, 3, g.number_of_edges(), {i: 0 for i in range(6)}, 3)
    print f
    
    g.remove_edge(0, 3)
    g.remove_edge(0, 4)
    g.remove_edge(0, 5)
    f=fit_requests(g.copy(), reqs, 3, g.number_of_edges() + 3, {i: 0 for i in range(6)}, 3)
    print f
    
    g.remove_edge(1, 2)
    g.add_edge(0, 1, capacity=1, allocated_capacity=0, static=False)
    g.add_edge(0, 2, capacity=1, allocated_capacity=0, static=False)
    f=fit_requests(g.copy(), reqs, 5, g.number_of_edges() + 3, {i: 0 for i in range(6)}, 3)
    print f
    
    
    g.add_edge(3, 4, capacity=1, allocated_capacity=0, static=False)
    g.add_edge(5, 0, capacity=1, allocated_capacity=0, static=False)
    print dict(nx.degree(g)).values()
    f=fit_requests(g.copy(), reqs, 5, g.number_of_edges() + 3, {i: 0 for i in range(6)}, 3)
    print f
    
    random = np.random.RandomState(seed=5)
    base_degrees = {i: 0 for i in range(30)}
    for i in range(100):
        g = nx.random_regular_graph(d=3, n=30, seed=i)
        tmp = nx.random_regular_graph(d=3, n=30, seed=i+1)
        reqs = [rqs.Request(u, v, i) for i, (u, v) in enumerate(tmp.edges())]
        for u, v, d in g.edges(data=True):
            d["capacity"] = 1
            d["allocated_capacity"] = 0
            d["static"] = False
        x = np.random.choice(
                np.arange(g.number_of_edges()), 
                replace=False,
                size=np.random.randint(3, 20)
                )
        e = [(u, v) for u, v in g.edges()]
        for i in x:
            g.remove_edge(*e[i])
        f=fit_requests(g.copy(), reqs, 10, g.number_of_edges(), base_degrees, 3)
        
        
    
            
            
    
    
