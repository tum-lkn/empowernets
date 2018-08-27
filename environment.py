import networkx as nx
import copy
import requests


class Environment(object):

    def __init__(self, topology, degree_limit=None, base_degree=None, base_capacity=1000):
        """
        Initializes object.

        Args:
            topology (networkx.Graph): Graph representing environment agent
                operates on.
            degree_limit (int, optional): Maximum degree of adjacent capacity
                of each node. Limit is inclusive. That is, this is the value of
                the maximum attainable degree.
            base_degree (dict, optional): The base degree of the topology. I.e.,
                the degree of each node before agent starts modifying things.
                If not given, will be extracted from the topology.

        Note:
            It is assumed that the given topology satisfies the degree constraint
            if any. This is not checked.
        """
        self.topology = topology
        self.routable_topology = nx.Graph()
        self.state = 0
        self.routed_requests = {}
        self.commodity_sfps = {}
        self.degree_limit = degree_limit if degree_limit is not None else 10e6
        if base_degree is None:
            self.base_degree = dict(nx.degree(self.topology, weight="capacity"))
        else:
            self.base_degree = base_degree
        self._nodes = [n for n, d in nx.degree(self.topology, weight="capacity")
                      if d < self.degree_limit + self.base_degree[n]]

        self.routable_topology.add_nodes_from(self.topology.nodes())
        for (u, v, d) in self.topology.edges(data=True):
            if d["allocated_capacity"] < d["capacity"]:
                self.routable_topology.add_edge(u, v)
        self.base_capacity = 1000

    @property
    def nodes(self):
        return [n for n, d in nx.degree(self.topology, weight="capacity")
                if d < self.degree_limit + self.base_degree[n]]

    @property
    def number_of_edges(self):
        return self.topology.number_of_edges()

    def copy(self):
        """
        Creates a copy of the environment.
        :return:
        """
        new_env = Environment(copy.deepcopy(self.topology), self.degree_limit, base_degree=self.base_degree.copy())
        new_env.routable_topology = copy.deepcopy(self.routable_topology)

        new_env.routed_requests = self.routed_requests.copy()

        return new_env

    def get_edges(self, include_static=False):
        if include_static:
            return [(u, v) for u, v, d in self.topology.edges(data=True) \
                    if d["allocated_capacity"] == 0]
        else:
            ret = []
            for u, v, d in self.topology.edges(data=True):
                if d["static"]:
                    if d["capacity"] > self.base_capacity:
                        ret.append((u, v))
                    else:
                        pass
                elif d["allocated_capacity"] == 0:
                    ret.append((u, v))
                else:
                    pass
            return ret

    def get_changeable_edges(self, include_static=False, increase=False):
        """
        Get the edges of the topology that can be updated, that is, for which
        the agent can increase capacity.

        Args:
            include_static (bool): When this flag is set to true then static
                edges are included as well.
            increase (bool): Indicates whether the capacity should be increased
                If set to false when decrease is assumed.

        Returns:
            edges (list): List of tuples representing edges that can be changed.

        """
        degrees = dict(nx.degree(self.topology, weight="capacity"))
        edges = []
        for u, v, d in self.topology.edges(data=True):
            if d["static"] and not include_static:
                # Make sure that static edges are only chosen if respective flag
                # is set to true.
                continue
            elif increase and (degrees[u] >= self.degree_limit + self.base_degree[u]
                               or degrees[v] >= self.degree_limit + self.base_degree[v]):
                # Make sure that only those edges can be chosen from for which
                # do degree constraint holds true.
                continue
            elif not increase and include_static and d["static"] and d["capacity"] < d["original_capacity"]:
                # Make sure that only those static edges are added in case of
                # an decrease that would not drop below their original capacity.
                continue
            elif not increase and d["allocated_capacity"] - d["capacity"] >= 0:
                # Make sure that only those edges can be chosen which still have
                # available capacity.
                continue
            else:
                edges.append((u, v))
        return edges

    def add_edge(self, edge, value):
        """
        Add edge to topology if edge does not exist, else increase capacity on
        existing edge by one.

        Args:
            edge (tuple): Int ids.
            value (numeric): Amount of which capacity on edge should be
                increased or a new edge should have.

        Returns:
            value (int): Value of added edge.
        """
        deg_u = nx.degree(self.topology, nbunch=edge[0], weight="capacity")
        deg_v = nx.degree(self.topology, nbunch=edge[1], weight="capacity")
        assert deg_u + value <= self.degree_limit + self.base_degree[edge[0]], ("Adding edge {:d}, {:d} " +
            "violates degree limit on node {:d}").format(edge[0], edge[1], edge[0])
        assert deg_v + value <= self.degree_limit + self.base_degree[edge[1]], ("Adding edge {:d}, {:d} " +
            "violates degree limit on node {:d}").format(edge[0], edge[1], edge[1])

        if self.topology.has_edge(*edge):
            self.topology.edges[edge[0], edge[1]]["capacity"] += value
        else:
            self.topology.add_edge(*edge, capacity=value, static=False, allocated_capacity=0)

        if not self.routable_topology.has_edge(edge[0], edge[1]):
            self.routable_topology.add_edge(*edge)

        if deg_u + value == self.degree_limit + self.base_degree[edge[0]]:
            self._nodes.remove(edge[0])
        if deg_v + value == self.degree_limit + self.base_degree[edge[1]]:
            self._nodes.remove(edge[1])

        self.commodity_sfps = {}

        return value

    def remove_edge(self, edge, value):
        """
        Remove existing edge from graph if capacity is one, else reduce capacity
        by one.

        Args:
            edge (tuple): Node ids.
            value (numeric): Amount of which capacity on edge should be
                decreased or a new edge should have.

        Returns:
            value (int): Value of removed edges.
        """
        # (edge[0] == 0 and edge[1] == 2) or (edge[0] == 2 and edge[1] == 0)
        if self.topology.has_edge(*edge):
            ctaken = self.topology.edges[edge[0], edge[1]]["allocated_capacity"]
            ctotal = self.topology.edges[edge[0], edge[1]]["capacity"]
            assert ctotal - ctaken >= value

            deg_u = nx.degree(self.topology, nbunch=edge[0], weight="capacity")
            deg_v = nx.degree(self.topology, nbunch=edge[1], weight="capacity")

            c = self.topology.edges[edge[0], edge[1]]["capacity"]
            if c > value:
                self.topology.edges[edge[0], edge[1]]["capacity"] = c - value
            else:
                self.topology.remove_edge(*edge)

            if ctaken >= ctotal - value:
                self.routable_topology.remove_edge(*edge)

            # If degree was previously equal to the degree limit then it is now
            # less and node can again be used to attach an edge.
            if deg_u == self.degree_limit + self.base_degree[edge[0]]:
                self._nodes.append(edge[0])
            if deg_v == self.degree_limit + self.base_degree[edge[1]]:
                self._nodes.append(edge[1])

            self.commodity_sfps = {}

        return value

    def reserve_capacity(self, path):
        """
        Reserve Capacity on edges, i.e., remove edges for which capacity drops
        to zero.

        Args:
            path (list): List of node ids forming path from start to target.

        Returns:
            None
        """
        for u, v in zip(path[:-1], path[1:]):
            c_allocated = self.topology.edges[u, v]["allocated_capacity"]
            c_free = self.topology.edges[u, v]["capacity"]
            assert c_allocated + 1 <= c_free, "Capacity out of bounds"
            self.topology.edges[u, v]["allocated_capacity"] = c_allocated + 1

            if c_allocated + 1 >= c_free:
                self.routable_topology.remove_edge(u, v)

    def free_capacity(self, path):
        """
        For the given path free the allocated capacity. That is, increment the
        capacity of existing edges and add edges with capacity one for not
        existing edges.

        Args:
            path (list): List of node ids.

        Returns:
            None

        """
        for u, v in zip(path[:-1], path[1:]):
            assert self.topology.edges[u, v]["allocated_capacity"] > 0, \
                "Try to free from edge that has no allocated capacity"
            if not self.routable_topology.has_edge(u, v):
                self.routable_topology.add_edge(u, v)
            self.topology.edges[u, v]["allocated_capacity"] -= 1

    def add_simple_request(self, request, path=None):
        """
        Add, i.e., route, request through topology. If path is given use that one.
        If path is not given then try to find the shortest path for given nodes.
        Bandwidth on edges is then allocated.

        Args:
            request (requests.Request): Request to route.
            path (list, optional): List of nodes forming path from start to
                target.

        Returns:
            path (list or None): None if no path could be found, else path in
                form of list of node ids.

        """
        # assert request.mask & self.state == 0, "Try to add already added request"
        assert request.mask not in self.routed_requests
        if path is None and request.commodity is not None and \
                request.commodity in self.commodity_sfps:
            if self.routable_over_path(self.commodity_sfps[request.commodity]):
                path = self.commodity_sfps[request.commodity]
            else:
                # not routable --> remove registered route
                self.commodity_sfps.pop(request.commodity)

        if path is None:
            try:
                path = nx.shortest_path(
                    self.routable_topology,
                    request.source,
                    request.target
                )
                if request.commodity is not None:
                    if request.commodity not in self.commodity_sfps:
                        self.commodity_sfps[request.commodity] = path
            except nx.NetworkXNoPath:
                path = None

        if path is not None:
            self.reserve_capacity(path)
            self.state = self.state | request.mask
            self.routed_requests[request.mask] = path
        return path

    def add_tm_request(self, request, path=None):
        """
        Add a traffic matrix request. Or try fitting it. Or whatever. For
        success all infinitesimal small flows need to be routed through
        the topology.

        Args:
            request (requests.TrafficMatrixRequest): Request to fit.
            path (list, optional): List of node ids describing a path.

        Returns:
            ret (int): id of request if successful else None.
        """
        s = 0
        ret = request.mask
        for r in request.request_iter():
            s += 1
            p = self.add_simple_request(r)
            if p is None:
                ret = None
                break
        return ret

    def add_request(self, request, path=None):
        """
        Dispatcher to handle different types of requests.

        Args:
            request (requests.AbstractRequest): An request type.
            path (list): List of node ids describing a path through the network.

        Returns:
            object
        """
        if type(request) == requests.Request:
            return self.add_simple_request(request, path)
        elif type(request) == requests.TrafficMatrixRequest:
            return self.add_tm_request(request, path)
        else:
            raise AssertionError

    def remove_request(self, request, path=None):
        """
        Remove routing for request. That is, free allocated bandwidth.

        Args:
            request (requests.Request): Request that should be removed.
            path (list, optional): Path for request.

        Returns:
            None
        """
        assert type(request) == requests.Request, "Unsupported Request type {}".format(type(request))
        # assert self.state & request.mask > 0, "Try to remove not routed requst"
        tmp = self.routed_requests.pop(request.mask)
        self.state = self.state & (~request.mask)
        if path is None:
            path = tmp
        self.free_capacity(path)
        return path

    def routable_over_path(self, path):
        """
        Checks whether or not a demand can be routed over a unit demand.

        Args:
            path (list): List of node identifiers.

        Returns:
            is_routable (bool): True if demand can be routed over path else False.
        """
        is_routable = True
        for u, v in zip(path[:-1], path[1:]):
            if self.routable_topology.has_edge(u, v):
                continue
            else:
                is_routable = False
        return is_routable

    def reset_topology(self):
        """
        Reset topology, that is, remove all routed requests and set
        capacity values accordingly and enable edges in routable topology.

        Returns:
            None
        """
        self.mask = 0
        self.routed_requests = {}
        self.commodity_sfps = {}
        for u, v, d in self.topology.edges(data=True):
            d['allocated_capacity'] = 0
            self.routable_topology.add_edge(u, v)
