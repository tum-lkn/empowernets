import json
import logging

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import environment
import requests
import visualization.generic_visualization as gen_vis


class SimulatedAnnealing(object):
    def __init__(self, env, reqs, num_extra_edges, num_steps, temperature0, cooling_factor, seed=0, subsample=None,
                 rnd_routing_order=False, outpath="", edge_capacity=1
                 ):
        """

        :param env:
        :param reqs: List of requests
        :param num_extra_edges: Number of edges/capacity that can be added
        :param num_steps: Number of steps to do in inner loop
        :param temperature0: Starting temperature
        :param cooling_factor:
        :param seed: Random seed
        :param subsample: Number of samples in subset if requests should be subsampled
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.outpath = outpath

        self.seed = seed
        self.randomState = np.random.RandomState(self.seed)

        self.env = env
        self.requests = reqs
        self.num_extra_edges = num_extra_edges  # total number of edges that we can add
        self.inventory = self.num_extra_edges  # Currently available edges that we can add
        self.num_moves = 0
        self.edge_cap = edge_capacity

        self.num_steps = num_steps  # Number of iterations of inner loop
        self.temperature0 = temperature0  # Initial temperature
        self.temperature = temperature0
        self.cooling_factor = cooling_factor

        self.subsample = subsample  # Number of requests in subsample
        self.rnd_routing_order = rnd_routing_order

        self.states = list()

    def num_routed_demands_for_sequence(self, request_sequence, tmp_env):
        """
        Route a list of requests in the given order by their shortest paths and return the number of successful routings
        :param request_sequence:
        :param tmp_env:
        :return: Number of successfully routed requests
        """
        num_routed = 0
        local_env = tmp_env.copy()
        for request in request_sequence:
            path = local_env.add_request(request)
            num_routed += (path is not None)

        return num_routed

    def determine_routing_rnd(self, tmp_env):
        """
        Routes as many requests as possible in the current topology. Shuffle the requests randomly
        :return:
        """

        best_sequence = None
        max_num_routed = 0

        for i in xrange(10):
            tmp_sequence = list(self.requests)
            if self.subsample is None:
                self.randomState.shuffle(tmp_sequence)
                sample = tmp_sequence
            else:
                sample = self.randomState.choice(tmp_sequence, replace=False, size=self.subsample)

            tmp_numrouted = self.num_routed_demands_for_sequence(sample, tmp_env)
            if tmp_numrouted > max_num_routed:
                max_num_routed = tmp_numrouted
                best_sequence = sample
        return max_num_routed, best_sequence

    def determine_routing(self, tmp_env):
        """
        Routes as many requests as possible in the current topology. Sort the requests by their expected SP length
        :return:
        """
        tmp_sequence = list(self.requests)
        if self.subsample is None:
            self.randomState.shuffle(tmp_sequence)
            sample = tmp_sequence
        else:
            sample = self.randomState.choice(tmp_sequence, replace=False, size=self.subsample)

        sp_lengths = dict(nx.all_pairs_shortest_path_length(tmp_env.topology))
        requests_sp_lengths = list()
        routable_requests = list()
        for req in sample:
            try:
                requests_sp_lengths.append(sp_lengths[req.source][req.target])
                routable_requests.append(req)
            except KeyError:
                pass

        tmp_sequence = sorted(list(zip(requests_sp_lengths, routable_requests)))
        sequence = [r[1] for r in tmp_sequence]
        return self.num_routed_demands_for_sequence(sequence, tmp_env), sequence

    def choose_edge_to_add(self, env_to_use=None):
        """
        Returns the edge that should be added to the topology. Selects the source node from the list of candidate nodes
         by weighting the selection probability with the number of terminated requests. The destination is chosen
         uniformly from the request destinations of that node
        :param env_to_use:
        :return:
        """
        if env_to_use is None:
            env_to_use = self.env

        node_candidates = env_to_use.nodes
        if len(node_candidates) <= 1:
            self.logger.info(" Not enough node candidates -> return")
            raise ValueError

        nodes_usage = [0] * len(env_to_use.topology.nodes)
        dsts_per_node = dict()
        for req in self.requests:
            src = req.source
            dst = req.target
            if dst not in env_to_use.nodes:
                # This ensures that prob > 0 only for nodes where we can still add an edge
                continue
            nodes_usage[src] += 1
            nodes_usage[dst] += 1

            if src not in dsts_per_node:
                dsts_per_node[src] = [dst]
            else:
                dsts_per_node[src].append(dst)

        final_nodes = np.zeros(len(env_to_use.topology.nodes))
        np.put(final_nodes, node_candidates, np.array(nodes_usage)[node_candidates])

        self.logger.debug(" Node usage: %s" % final_nodes)
        if np.sum(final_nodes) == 0:
            # Nodes are not used, choose uniformly from candidates
            new_edge_src = self.randomState.choice(node_candidates)
            new_edge_dst = self.randomState.choice(node_candidates)
            while new_edge_dst == new_edge_src:
                new_edge_dst = self.randomState.choice(node_candidates)
        else:
            probs = 1.0 * np.array(final_nodes) / np.sum(final_nodes)
            new_edge_src = self.randomState.choice(list(env_to_use.topology.nodes), p=probs)

            new_edge_dst = self.randomState.choice(dsts_per_node.get(new_edge_src, node_candidates))
            while new_edge_dst == new_edge_src:
                new_edge_dst = self.randomState.choice(dsts_per_node.get(new_edge_src, node_candidates))
        return new_edge_src, new_edge_dst

    def modify_network(self):
        """
        Modifies the topology on a copy of the environment and returns this updated copy along with the number of
        edges taken from the inventory. Either adds a new edge or moves an existing one.
        :return: Updated environment
        """
        env_copy = self.env.copy()
        inventory_change = 0
        # Flip coin to decide if we add a new edge or modify an existing one.
        # The probability is dependent on the inventory and the number of move actions that were performed before
        prob_moving_edge = (1 - (1.0 * self.inventory / self.num_extra_edges)) ** (1+self.num_moves)

        # Calculate available Capacity for logging purpose:
        cap = 0
        for u, v, d in env_copy.topology.edges(data=True):
            cap += d["capacity"]
        self.logger.info(" Prob_add_edge = {}/{}; #Edges {}".format(self.inventory, self.num_extra_edges, cap))

        if self.randomState.uniform(0, 1) < 1 - prob_moving_edge:
            # Add a new edge
            self.logger.info(" Add new edge")
            try:
                new_edge = self.choose_edge_to_add()
                env_copy.add_edge(new_edge, self.edge_cap)
                inventory_change = -1
            except ValueError:
                self.logger.error(" Not enough nodes to create meaningful edges. Set inventory to zero")
            self.num_moves = 0
        else:
            # Modify existing edge
            self.logger.info(" Modify edge")

            # Remove random edge (uniformly chosen)
            edge_idx_to_remove = self.randomState.choice(range(env_copy.number_of_edges))
            edge_to_remove = list(env_copy.topology.edges)[edge_idx_to_remove]
            env_copy.remove_edge(edge_to_remove, self.edge_cap)

            # Add a new edge
            new_edge = self.choose_edge_to_add(env_to_use=env_copy)
            env_copy.add_edge(new_edge, self.edge_cap)

            self.num_moves += 1

        return env_copy, inventory_change

    def reduce_temperature(self):
        self.temperature = self.cooling_factor * self.temperature * 1.0

    def visualize_topo(self, env, sequence, filename=None, close=False):
        """
        Plot the topo for debugging purposes
        :param env:
        :param sequence:
        :return:
        """
        topo_copy = env.topology.copy()
        edge_labels = dict()
        sum_alloc = 0

        taken_edges = set()
        free_edges = []
        for request in sequence:
            try:
                path = nx.shortest_path(topo_copy, request.source, request.target)
            except nx.NetworkXNoPath:
                continue
            # Every edge has only a capacity of one. so remove it from the graph when a flow is routed on it
            old_node = path[0]
            for e in path[1:]:
                # topo_copy.edges[old_node, e]["allocated_capacity"] += 1
                edge_labels[(old_node, e)] = topo_copy.edges[old_node, e]["capacity"]
                taken_edges.add((old_node, e))
                if topo_copy.edges[old_node, e]["allocated_capacity"] == topo_copy.edges[old_node, e]["capacity"]:
                    topo_copy.remove_edge(old_node, e)
                old_node = e
                sum_alloc += 1

        self.logger.info(" Sum allocated capacity: %s " % sum_alloc)

        positions = nx.spring_layout(env.topology, iterations=300)
        fig, ax = gen_vis.make_fig(scale_height=1)

        nx.draw_networkx_edges(env.topology, pos=positions,
                               edge_color="green", ax=ax)
        nx.draw_networkx_edges(env.topology, pos=positions, edgelist=taken_edges,
                               edge_color="orange", ax=ax)
        nx.draw_networkx_edge_labels(env.routable_topology, pos=positions,
                                    edge_labels=edge_labels)
        nx.draw_networkx_nodes(env.topology, pos=positions, node_color="blue",
                               node_size=10, ax=ax)

        if filename is not None:
            plt.savefig("{:s}.pdf".format(filename), format="pdf")
        if close:
            plt.close()

    def run(self):
        """
        Performs the simulated annealing. Keeps track of the best topology found and eventually stores the edges along
         with the sequence of requests to a file.
        :return: Number of routed flows and corresponding environment
        """
        best_num_routed, best_sequence = self.determine_routing_rnd(self.env)
        best_env = self.env.copy()
        last_best_change = 0
        env_num_routed = best_num_routed

        decrease_temp = True

        while decrease_temp:
            # Outer loop -> temperature
            for step in xrange(self.num_steps):
                # Inner loop modify neighborhood
                tmp_env, inventory_update = self.modify_network()
                if self.rnd_routing_order:
                    num_routed, sequence = self.determine_routing_rnd(tmp_env)
                else:
                    num_routed, sequence = self.determine_routing(tmp_env)
                delta_num_routed = env_num_routed - num_routed
                if delta_num_routed < 0:
                    self.env = tmp_env
                    self.inventory += inventory_update
                    env_num_routed = num_routed
                    if env_num_routed > best_num_routed:
                        best_num_routed = env_num_routed
                        best_env = self.env.copy()
                        best_sequence = sequence
                        last_best_change = 0
                elif self.randomState.uniform(0, 1) < np.exp(-delta_num_routed / 1.0 / self.temperature):
                    self.env = tmp_env
                    self.inventory += inventory_update
                last_best_change += 1

                self.logger.info(" Temp %i - Routed Flows: %s" % (self.temperature, best_num_routed))

            self.reduce_temperature()

            # Save current state
            self.states.append({
                "topo": [(u, v, d) for u, v, d in best_env.topology.edges(data=True)],
                "paths": best_env.routed_requests.copy()
            })

            if self.temperature < 1:
                decrease_temp = False
            if last_best_change > 100:
                decrease_temp = False

        # Print results to file
        filename = self.outpath+("simulated_annealing-random-requests-all-nodes-source-{}-edges-{}" +
                    "-subsamples-{}-t0-{}-cooling-seed-{}.json").format(
            self.num_extra_edges, self.subsample, self.temperature0, self.cooling_factor, self.seed
        )
        # Not all request from the stored sequence might fit on the topology. however this is the sequence of requests
        # that resulted in the best result.
        with open(filename, "w") as fh:
            json.dump([self.states, [(r.source, r.target) for r in best_sequence]], fh)

        return best_num_routed, best_env, best_sequence


def evaluate_model(graph, rqs, inventory, degree_constraint, edge_cap, t0, cooling, iterations, seed, routing_order_rnd=False, subsample=None):
    env = environment.Environment(graph, degree_limit=degree_constraint)
    myalgo = SimulatedAnnealing(
        env=env,
        reqs=rqs,
        num_extra_edges=inventory,
        num_steps=iterations,
        temperature0=t0,
        cooling_factor=cooling,
        seed=seed,
        subsample=subsample,
        rnd_routing_order=routing_order_rnd,
        outpath="data/",
        edge_capacity=edge_cap
    )
    res_num_routed, res_env, res_sequence = myalgo.run()
    return res_env.topology, res_sequence

if __name__ == "__main__":
    PLT_OUT_PATH = "plots/"

    logging.basicConfig(level=logging.INFO)
    NUM_EDGES = 45
    NUM_NODES = 30
    NUM_REQUESTS = 45

    random = np.random.RandomState(seed=1000)
    g = nx.Graph()
    g.add_nodes_from(range(NUM_NODES))

    env = environment.Environment(g, degree_limit=3)

    rqs = []

    #while len(rqs) < NUM_REQUESTS:
    #    u = random.randint(0, NUM_NODES)
    #    # u = len(rqs)
    #    v = random.randint(0, NUM_NODES)
    #    if u == v:
    #        continue
    #    else:
    #        rqs.append(requests.Request(u, v, 2 ** len(rqs)))

    rqs = []
    count = 0
    for i in range(NUM_NODES):
        for j in range(i + 1, NUM_NODES):
            rqs.append(requests.Request(i, j, count))
            count += 1

    srcs = [0] * NUM_NODES
    dsts = [0] * NUM_NODES

    for req in rqs:
        srcs[req.source] += 1
        dsts[req.target] += 1

    # plt.subplot(1,3,1)
    # plt.bar(range(NUM_NODES), srcs)
    # plt.subplot(1,3,2)
    # plt.bar(range(NUM_NODES), dsts)
    # plt.subplot(1,3,3)
    # plt.bar(range(NUM_NODES), np.array(srcs)+np.array(dsts))
    # plt.show()

    # print np.sum((np.array(srcs) + np.array(dsts)) > 3)

    for seed in range(1000, 1020):
        myalgo = SimulatedAnnealing(
            env=env,
            reqs=rqs,
            num_extra_edges=NUM_EDGES,
            num_steps=100,
            temperature0=1000,
            cooling_factor=0.99,
            seed=seed,
            subsample=45,
            rnd_routing_order=False,
            outpath="data/"
        )

        res_num_routed, res_env, res_sequence = myalgo.run()
        myalgo.visualize_topo(res_env, res_sequence, filename=PLT_OUT_PATH+"topo_%s" % seed, close=True)
        with open("data/topo_%s.json" % seed, "w") as fp:
            json.dump(nx.jit_data(res_env.topology), fp=fp)
        print seed, res_num_routed
