import environment
import requests
import networkx as nx

g = nx.Graph()
g.add_nodes_from(range(5))
# g.add_edge(0, 1, static=False, capacity=1, allocated_capacity=0)
# g.add_edge(1, 2, static=False, capacity=1, allocated_capacity=0)
# g.add_edge(2, 3, static=False, capacity=1, allocated_capacity=0)
# g.add_edge(3, 0, static=False, capacity=1, allocated_capacity=0)
# g.add_edge(4, 1, static=False, capacity=1, allocated_capacity=0)
# g.add_edge(4, 2, static=False, capacity=1, allocated_capacity=0)

edge_value = 1000
env = environment.Environment(g, 3 * edge_value)

# env.add_edge((1, 3))
# env.add_edge((0, 2))
env.add_edge((0, 3), edge_value)
assert env.routable_topology.has_edge(0, 3)
assert env.topology.has_edge(0, 3)
assert env.topology.edges[0, 3]["capacity"] == 1 * edge_value
assert env.topology.edges[0, 3]["allocated_capacity"] == 0
assert not env.topology.edges[0, 3]["static"]

env.add_edge((0, 3), edge_value)
assert env.routable_topology.has_edge(0, 3)
assert env.topology.has_edge(0, 3)
assert env.topology.edges[0, 3]["capacity"] == 2 * edge_value
assert env.topology.edges[0, 3]["allocated_capacity"] == 0
assert not env.topology.edges[0, 3]["static"]

env.remove_edge((0, 3), edge_value)
assert env.routable_topology.has_edge(0, 3)
assert env.topology.has_edge(0, 3)
assert env.topology.edges[0, 3]["capacity"] == 1 * edge_value
assert env.topology.edges[0, 3]["allocated_capacity"] == 0
assert not env.topology.edges[0, 3]["static"]

env.remove_edge((0, 3), edge_value)
assert not env.routable_topology.has_edge(0, 3)
assert not env.routable_topology.has_edge(0, 3)

env.add_edge((0, 3), edge_value)
assert env.routable_topology.has_edge(0, 3)
assert env.topology.has_edge(0, 3)
assert env.topology.edges[0, 3]["capacity"] == 1 * edge_value
assert env.topology.edges[0, 3]["allocated_capacity"] == 0
assert not env.topology.edges[0, 3]["static"]

env.add_edge((0, 1), edge_value)
assert env.routable_topology.has_edge(0, 1)
assert env.topology.has_edge(0, 1)
assert env.topology.edges[0, 1]["capacity"] == 1 * edge_value
assert env.topology.edges[0, 1]["allocated_capacity"] == 0
assert not env.topology.edges[0, 1]["static"]

p1 = env.add_request(requests.Request(1, 3, 1))
if edge_value == 1:
    assert not env.routable_topology.has_edge(0, 3)
    assert not env.routable_topology.has_edge(1, 3)
else:
    assert env.routable_topology.has_edge(0, 3)
    assert env.routable_topology.has_edge(1, 0)
assert env.topology.has_edge(0, 3)
assert env.topology.has_edge(0, 1)
assert env.topology.edges[0, 3]["capacity"] == 1 * edge_value
assert env.topology.edges[0, 3]["allocated_capacity"] == 1
assert env.topology.edges[0, 1]["capacity"] == 1 * edge_value
assert env.topology.edges[0, 1]["allocated_capacity"] == 1

env.add_edge((0, 3), edge_value)
if edge_value == 1:
    assert env.routable_topology.has_edge(0, 3)
assert env.topology.has_edge(0, 3)
assert env.topology.edges[0, 3]["capacity"] == 2 * edge_value
assert env.topology.edges[0, 3]["allocated_capacity"] == 1

env.remove_edge((0, 3), edge_value)
if edge_value == 1:
    assert not env.routable_topology.has_edge(0, 3)
assert env.topology.has_edge(0, 3)
assert env.topology.edges[0, 3]["capacity"] == 1 * edge_value
assert env.topology.edges[0, 3]["allocated_capacity"] == 1

print env.topology.edges[0, 3]["allocated_capacity"]
p2 = env.add_request(requests.Request(0, 3, 2))
if edge_value == 1:
    assert not env.routable_topology.has_edge(0, 3)
    assert not env.routable_topology.has_edge(1, 3)
assert env.topology.has_edge(0, 3)
assert env.topology.has_edge(0, 1)
assert env.topology.edges[0, 3]["capacity"] == 1 * edge_value
print env.topology.edges[0, 3]["allocated_capacity"]
assert env.topology.edges[0, 3]["allocated_capacity"] == 1
assert env.topology.edges[0, 1]["capacity"] == 1 * edge_value
assert env.topology.edges[0, 1]["allocated_capacity"] == 1

env.remove_request(requests.Request(1, 3, 1), path=p1)
assert env.routable_topology.has_edge(0, 3)
assert env.routable_topology.has_edge(0, 1)
assert env.topology.has_edge(0, 3)
assert env.topology.has_edge(0, 1)
assert env.topology.edges[0, 3]["capacity"] == 1 * edge_value
assert env.topology.edges[0, 3]["allocated_capacity"] == 0
assert env.topology.edges[0, 1]["capacity"] == 1 * edge_value
assert env.topology.edges[0, 1]["allocated_capacity"] == 0

env.add_request(requests.Request(0, 3, 2))
assert not env.routable_topology.has_edge(0, 3)
assert env.routable_topology.has_edge(0, 1)
assert env.topology.has_edge(0, 3)
assert env.topology.has_edge(0, 1)
assert env.topology.edges[0, 3]["capacity"] == 1 * edge_value
assert env.topology.edges[0, 3]["allocated_capacity"] == 1
assert env.topology.edges[0, 1]["capacity"] == 1 * edge_value
assert env.topology.edges[0, 1]["allocated_capacity"] == 0

env.remove_edge((0, 1), edge_value)
assert not env.routable_topology.has_edge(0, 3)
assert not env.routable_topology.has_edge(1, 3)
assert env.topology.has_edge(0, 3)
assert not env.topology.has_edge(0, 1)
assert env.topology.edges[0, 3]["capacity"] == 1 * edge_value
assert env.topology.edges[0, 3]["allocated_capacity"] == 1

try:
    env.remove_edge((0, 3))
    AssertionError
except:
    pass
env.remove_request(requests.Request(0, 3, 2))
assert env.routable_topology.has_edge(0, 3)
assert not env.routable_topology.has_edge(1, 3)
assert env.topology.has_edge(0, 3)
assert not env.topology.has_edge(0, 1)
assert env.topology.edges[0, 3]["capacity"] == 1 * edge_value
assert env.topology.edges[0, 3]["allocated_capacity"] == 0

env.remove_edge((0, 3), edge_value)
assert not env.routable_topology.has_edge(0, 3)
assert not env.routable_topology.has_edge(1, 3)
assert not env.topology.has_edge(0, 3)
