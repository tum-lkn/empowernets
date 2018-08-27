import environment
import requests
import networkx as nx
import agents

topo = nx.Graph()
topo.add_edge(0, 1, capacity=10, allocated_capacity=0, static=True)
topo.add_edge(2, 1, capacity=10, allocated_capacity=0, static=True)
topo.add_edge(3, 1, capacity=10, allocated_capacity=0, static=True)

tm = {
    0: {
        1: 5,
        2: 5,
        3: 5
    },
    1: {
        0: 5,
        2: 5,
        3: 5
    },
    2: {
        0: 5,
        1: 5,
        3: 5
    },
    3: {
        0: 5,
        1: 5,
        2: 5
    }
}

env = environment.Environment(topo.copy(), 20)
ret = env.add_request(requests.TrafficMatrixRequest(tm, 1))
assert ret is None, "Ret is not None!"
env.reset_topology()

env.add_edge((0, 2), 5)
env.add_edge((2, 0), 5)

env.add_edge((3, 0), 5)
env.add_edge((0, 3), 5)

env.add_edge((1, 2), 5)
env.add_edge((1, 3), 5)
env.add_edge((2, 1), 5)
env.add_edge((3, 1), 5)

assert len(env.nodes) == 0

edges = env.get_changeable_edges(include_static=False)
assert (0, 2) in edges
assert (0, 3) in edges

env.remove_edge((1, 2), 10)
env.remove_edge((1, 3), 10)

env.add_edge((2, 3), 5)
env.add_edge((3, 2), 5)

ret = env.add_request(requests.TrafficMatrixRequest(tm, 1))
assert ret == 1, "Ret is not None!"


t = nx.erdos_renyi_graph(5, 0.7, seed=5)
for u, v, d in t.edges(data=True):
    d["capacity"] = 100
    d["allocated_capacity"] = False
    d["static"] = False
env = environment.Environment(t)
rq1 = requests.TrafficMatrixRequest(tm.copy(), 1)
rq2 = requests.TrafficMatrixRequest(tm.copy(), 2)
agent = agents.RequestAwareAgent(5, 10, [rq1, rq2], env, None)
agent.estimate_empowerment()
