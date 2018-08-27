"""
Defines agents with different embodiments.
"""

import numpy as np
import multiprocessing
import actions
import concurrent
import logging
import requests as emprequests


logging.basicConfig(level=logging.DEBUG)


REQUEST_SENSING_AGENT = 1
REQUEST_AWARE_AGENT = 2
REQUEST_SENSING_PERMUTATION_UNAWARE_AGENT = 3
REQUEST_AWARE_PERMUTATION_UNAWARE_AGENT = 4


class AgentFactory(object):

    @classmethod
    def produce(cls, order, horizon, inventory_size, requests, environment,
                subsample=None, num_sequences=1000, seed=1):
        """
        Generate an Agent.

        Args:
            order (int): Type of the agent, must be in {1, 2, 3, 4}.
            horizon (int): Length of action sequences.
            inventory_size (int): Capacity available to the agent to deploy
                in the topology.
            requests (list): List of requests that might be happening.
            environment (environment.Environment): Environment in which the
                agent resides.
            subsample (int, optional): How many of the possible requests should
                be considered. Default is None, corresponding to all requests.
            num_sequences (int, optional): Number of random action sequences
                the agent should perform. Default is 1000.
            seed (numeric, optional): Random seed that should be used to produce
                random actions. Default is 1.

        Returns:
            agent (AbstractAgent).

        Raises:
            RuntimeError if order is unknown.

        """
        if order == REQUEST_SENSING_AGENT:
            return RequestSensingAgent(
                horizon=horizon,
                inventory_size=inventory_size,
                requests=requests,
                environment=environment,
                num_sequences=num_sequences,
                seed=seed
            )
        elif order == REQUEST_SENSING_PERMUTATION_UNAWARE_AGENT:
            return RequestSensingPermutationUnawareAgent(
                horizon=horizon,
                inventory_size=inventory_size,
                requests=requests,
                environment=environment,
                num_sequences=num_sequences,
                seed=seed
            )
        elif order == REQUEST_AWARE_AGENT:
            return RequestAwareAgent(
                horizon=horizon,
                inventory_size=inventory_size,
                requests=requests,
                environment=environment,
                subsample=subsample,
                num_sequences=num_sequences,
                seed=seed
            )
        elif order == REQUEST_AWARE_PERMUTATION_UNAWARE_AGENT:
            return RequestAwarePermutationUnawareAgent(
                horizon=horizon,
                inventory_size=inventory_size,
                requests=requests,
                environment=environment,
                subsample=subsample,
                num_sequences=num_sequences,
                seed=seed
            )
        else:
            raise RuntimeError("Unknown Order")


class AbstractAgent(object):
    """
    Abstract base class for concrete agent implementations.
    """

    def __init__(self, horizon, inventory_size, requests, environment,
                 num_sequences, seed):
        """
        Initializes object.

        Args:
            horizon (int): Length of randomly sampled sequence, i.e., length of
                agent horizon.
            inventory_size (int): Number of elements agent can store in his
                inventory.
            requests (Array of requests.Request): Requests the agent can sample
                from.
            environment (environment.Environment): Environment object.
            subsample (int, optional): Subsample the requests instead of
                shuffling them.
            num_sequences (int, optional): Number of random sequences to perform
                for empowerment estimation.
            with_ilp (bool, optional): Whether to find maximum number of reqeusts
                that can be routed with ILP. If false SPF routing is used.
            seed (int, optional): Seed for random number generator.
        """
        self.horizon = horizon
        self.inventory_size = inventory_size
        self.inventory = 0
        self.requests = requests
        self.random = np.random.RandomState(seed=seed)
        self.environment = environment
        self.num_sequences = num_sequences
        self.actions = []
        self.sensing_threshold = None
        self.edge_value = 1

    def sample_request(self):
        """
        Samples a requests uniformly at random from `requests`.

        Returns requests.Request:
        """
        return self.requests[self.random.randint(0, len(self.requests))]

    def sample_request_to_remove(self):
        """
        From all embedded requests randomly chose one request.

        Returns:
            requests.Request
        """
        ids = self.environment.routed_requests.keys()
        if len(ids) == 0:
            return None
        else:
            return self.requests[ids[self.random.randint(0, len(ids))]]

    def sample_action(self):
        """
        Samples an action uniformly at random.

        Returns:
            actions.Action

        Note:
            I make use of functional programming here and directly instantiate
            an object. I.e., actions stores the class names for which I call
            the constructor.
        """
        idx = self.random.randint(0, len(self.actions))
        return self.actions[idx](self)  # Get class and instantiate new object

    def random_action_sequence(self):
        """
        Generates a random sequence of actions of length `horizon`.

        Yields: requests.Request
        """
        for i in range(self.horizon):
            yield self.sample_action()

    def sample_random_existing_edge(self, include_static=False):
        """
        Samples an edge uniform at random of the existing edges that do not
        route any traffic.

        Returns:
            edge (tuple): Tuple incident node ids.
            include_static (bool): Whether static edges should be included in
                the set of edges the agent may choose from.
        """
        edge = None
        edges = self.environment.get_edges(include_static)
        if len(edges) > 0:
            edge = edges[self.random.randint(0, len(edges))]
        return edge

    def sample_random_edge(self):
        """
        From all possible edges choose one at random, i.e., let V be the set
        of nodes, then the edge returned is chosen at random from  the
        cross-product V x V.

        Returns:
            edge (tuple): Tuple with start and end node.
        """
        if len(self.environment.nodes) < 2:
            return None
        else:
            return self.random.choice(self.environment.nodes, replace=False, size=2).tolist()

    def sample_random_not_existing_edge(self):
        """
        Sample an edge at random that is not yet added to the topology.

        Returns:
            edge (tuple): Tuple specifying edge with start and end-node.
        """
        exists = True
        edge = None
        # Probability of sampling an existing edge by chance is quite low. So
        # this should be rather efficient.
        count = 0
        while exists and count < 50:
            count += 1
            edge = self.sample_random_edge()
            if edge is None:
                exists = False
            else:
                exists = self.environment.topology.has_edge(*edge)
        return edge

    def choose_edge_to_take(self):
        """
        Choose an edge that should be put in an inventory.

        Returns:
            edge (tuple): Tuple of incident node ids.
        """
        return self.sample_random_existing_edge(include_static=False)

    def choose_edge_to_place(self, exclude_existing=False):
        """
        Choose an edge that should be realized.

        Args:
            exclude_existing (bool): If set to True then an edge is sampled that
                does not yet exist in the graph. If set to False then any edge
                can be returned, even if it already exists in the graph.

        Returns:
            edge (tuple): Tuple of incident node ids.
        """
        if exclude_existing:
            return self.sample_random_not_existing_edge()
        else:
            return self.sample_random_edge()

    def choose_edge_to_change(self, include_static=False, increase=False):
        """
        Choose an edge that should be realized.

        Args:
            include_static (bool): Include static edges in the candidate set.
            increase (bool): Indicate whether capacity on edge should be in-
                or decreased.

        Returns:
            edge (tuple): Tuple of incident node ids.
        """
        edges = self.environment.get_changeable_edges(include_static, increase)
        if len(edges) == 0:
            return None
        else:
            return edges[self.random.randint(0, len(edges))]

    def filter(self, environment_state):
        """
        Filter out uninteresting states. In this case uninteresting means states
        in which less then a specific number of requests could be served.

        Args:
            environment_state (list): Set of routed requests.

        Returns:
            filtered_signal (np.ndarray)

        """
        if self.sensing_threshold is None:
            filtered_signal = environment_state
        else:
            count = len(environment_state)
            if count < self.sensing_threshold:
                filtered_signal = []
            else:
                filtered_signal = environment_state
        return np.array(filtered_signal)

    def choose_next_action(self):
        """
        Choose the next action.

        Raises:
            NotImplementedError: Must be implemented in subclass.

        """
        raise NotImplementedError()

    def estimate_empowerment(self):
        """
        Procedure to estimate empowerment. Must be implemented in subclass.

        Raises:
            NotImplementedError.

        """
        raise NotImplementedError()

    def sense(self):
        """
        Obtain a sensor reading from the environment.

        Raises:
            NotImplementedError: Must be implemented in subclass.
        """
        raise NotImplementedError()

    def add_new_reading(self, reading, readings):
        """
        Checks whether reading is a new reading or not.

        Args:
            reading (numpy.ndarray): Array of current sensor reading.
            readings (list of numpy.ndarray): Previous sensor readings.

        Returns:
            None
        """
        reading = np.sort(reading)
        if len(readings) == 0:
            readings.append(reading)
        elif readings[-1].size < reading.size:
            readings.append(reading)
        else:
            for i, r in enumerate(readings):
                if r.size < reading.size:
                    continue
                elif r.size == reading.size:
                    c = np.sum(np.equal(r, reading))
                    if c == reading.size:
                        continue
                    else:
                        readings.insert(i, reading)
                        break
                else:  # r.size > reading.size:
                    break


class RequestSensingAgent(AbstractAgent):
    """
    Agent that can manipulate edges and requests. It perceives exactly the set
    of requests that are currently served in the topology.
    """

    def __init__(self, horizon, inventory_size, requests, environment,
                 num_sequences=1000, seed=1):
        """
        Initializes object.

        Args:
            horizon (int): Length of randomly sampled sequence, i.e., length of
                agent horizon.
            inventory_size (int): Number of elements agent can store in his
                inventory.
            requests (Array of requests.Request): Requests the agent can sample
                from.
            environment (environment.Environment): Environment object.
            num_sequences (int, optional): Number of random sequences to perform
                for empowerment estimation.
            seed (int, optional): Seed for random number generator.
        """
        super(RequestSensingAgent, self).__init__(horizon, inventory_size, requests,
                                                  environment, num_sequences, seed)
        self.actions = [
            actions.AddEdge,
            actions.TakeEdge,
            actions.RemoveRequest,
            actions.AddRequest,
            actions.DoNothing
        ]
        self.type = REQUEST_SENSING_AGENT

    def sense(self):
        """
        Obtain the ids of all routed requests.

        Returns:
            sensor_reading (list): List of request ids.
        """
        return self.filter(self.environment.routed_requests.keys())

    def estimate_empowerment(self):
        """
        Estimate the empowerment of the current state.

        Returns:
            empowerment (int)
        """
        states = []
        for i in range(self.num_sequences):
            actions = []
            for action in self.random_action_sequence():
                actions.append(action)
                action.apply()

            state = self.sense()
            self.add_new_reading(state, states)

            actions.reverse()
            for action in actions:
                action.rollback()
        if self.sensing_threshold is None:
            average_embedded = 0 # does not matter
        else:
            if type(states[0]) == int:
                average_embedded = np.mean(states)
            else:
                average_embedded = np.mean([a.size for a in states])
        return np.log(len(states)), average_embedded

    def choose_next_action(self):
        """
        Perform each registered action and estimated the empowerment of the
        resulting state. Return the action with largest empowerment.

        Returns:
            action (actions.Action)
            empowerment (float)
        """
        best_actions = []
        best_empowerment = 0
        average_embedded = []

        for action_class in self.actions:
            action = action_class(self)
            action.apply()
            empowerment, avg_embd = self.estimate_empowerment()
            average_embedded.append(avg_embd)
            action.rollback()

            if empowerment == best_empowerment:
                best_actions.append(action)
            elif empowerment > best_empowerment:
                best_actions = [action]
                best_empowerment = empowerment
            else:
                pass

        if self.sensing_threshold is not None:
            tmp = np.max(average_embedded)
            if tmp > self.sensing_threshold:
                self.sensing_threshold = tmp

        return best_actions[self.random.randint(0, len(best_actions))],\
               best_empowerment

    def choose_request_to_place(self):
        return self.sample_request()

    def choose_request_to_remove(self):
        return self.sample_request_to_remove()


class RequestAwareAgent(AbstractAgent):
    """
    Can place or take edges but has no influence on requests. For evaluation
    all requests are used. Agent has only the actions AddEdge, TakeEdge and
    DoNothing.
    Intuitiviely I would guess that this agent moves to a state of the environment
    from which it can easily reach states that are able cover as many different
    combinations of requests as possible.
    """

    def __init__(self, horizon, inventory_size, requests, environment, subsample=None,
                 num_sequences=1000, with_ilp=False, seed=1):
        """
        Initializes object.

        Args:
            horizon (int): Length of randomly sampled sequence, i.e., length of
                agent horizon.
            inventory_size (int): Number of elements agent can store in his
                inventory.
            requests (Array of requests.Request): Requests the agent can sample
                from.
            environment (environment.Environment): Environment object.
            subsample (int, optional): Subsample the requests instead of
                shuffling them.
            num_sequences (int, optional): Number of random sequences to perform
                for empowerment estimation.
            with_ilp (bool, optional): Whether to find maximum number of reqeusts
                that can be routed with ILP. If false SPF routing is used.
            seed (int, optional): Seed for random number generator.
        """
        super(RequestAwareAgent, self).__init__(horizon, inventory_size, requests,
                                                environment, num_sequences, seed)
        self.actions = [
            actions.AddEdge,
            actions.TakeEdge,
            actions.DoNothing
            # actions.IncreaseCapacity,
            # actions.DecreaseCapacity
        ]
        if type(self.requests[0]) == emprequests.Request:
            self.request_data = [(r.source, r.target, r.mask) for r in self.requests]
        else:
            self.request_data = [r.tm for r in self.requests]
        self.subsample = subsample
        self.type = REQUEST_AWARE_AGENT
        self.with_ilp = with_ilp

    def sense(self):
        """
        Obtain the ids of all routed requests.

        Returns:
            sensor_reading (list): List of request ids.
        """
        return self.filter(self.environment.routed_requests.keys())

    def estimate_empowerment(self):
        """
        Estimate the empowerment of the current state.

        Returns:
            empowerment (int)
        """
        num_processors = 8
        edge_data = [(u, v, d) for u, v, d in self.environment.topology.edges(data=True)]
        nsq = int(self.num_sequences / num_processors)
        nsq += self.num_sequences - nsq * num_processors
        prototype = {
            "num_nodes": self.environment.topology.number_of_nodes(),
            "edge_data": edge_data,
            "seed": None,
            "horizon": self.horizon,
            "inventory": self.inventory,
            "inventory_size": self.inventory_size,
            "request_data": self.request_data,
            "num_sequences": nsq,
            "max_degree": self.environment.degree_limit,
            "type": self.type,
            "base_degrees": self.environment.base_degree,
            "with_ilp": self.with_ilp
        }
        if self.subsample is not None:
            prototype["subsample"] = self.subsample
        if self.sensing_threshold is not None:
            prototype["sensor_threshold"] = self.sensing_threshold
        jobs = []
        for i in range(num_processors):
            tmp = prototype.copy()
            tmp["seed"] = self.random.randint(0, 1000000)
            jobs.append(tmp)

        # pool = multiprocessing.Pool(processes=num_processors)
        # ret = None
        # try:
        #     reduce = pool.map(concurrent.evaluate_sequence, jobs)
        #     pool.close()
        reduce = [concurrent.evaluate_sequence(jobs[0])]
        l = []
        for s in reduce:
           if type(s[0][0]) == int:
               l.append(s[0])
           else:
               l.extend(s[0])
        states = np.unique(np.concatenate(l))
        average_embedded = np.mean([s[1] for s in reduce])
        ret = np.log(states.size)
        # except Exception as e:
        #     pool.close()
        #     logging.exception(e)
        #     raise e
        return ret, average_embedded

    def choose_next_action(self):
        """
        Perform each registered action and estimated the empowerment of the
        resulting state. Return the action with largest empowerment.

        Returns:
            action (actions.Action)
            empowerment (float)
        """
        best_actions = []
        best_empowerment = -1
        average_embedded = []

        for action_class in self.actions:
            action = action_class(self)
            action.apply()
            empowerment, avg_embd = self.estimate_empowerment()
            average_embedded.append(avg_embd)
            action.rollback()
            
            if np.isinf(empowerment):
                empowerment = 0

            if empowerment == best_empowerment:
                best_actions.append(action)
            elif empowerment > best_empowerment:
                best_actions = [action]
                best_empowerment = empowerment
            else:
                pass

        if self.sensing_threshold is not None:
            tmp = np.max(average_embedded)
            if tmp > self.sensing_threshold:
                self.sensing_threshold = tmp

        return best_actions[self.random.randint(0, len(best_actions))], \
               best_empowerment


class RequestSensingPermutationUnawareAgent(RequestSensingAgent):
    """
    This agent can manipulate edges and requests, but perceives only the number
    of currently routed requests and not the set of requests.
    """

    def __init__(self, horizon, inventory_size, requests, environment,
                 num_sequences=1000, seed=1):
        """
        Initializes object.

        Args:
            horizon (int): Length of randomly sampled sequence, i.e., length of
                agent horizon.
            inventory_size (int): Number of elements agent can store in his
                inventory.
            requests (Array of requests.Request): Requests the agent can sample
                from.
            environment (environment.Environment): Environment object.
            num_sequences (int, optional): Number of random sequences to perform
                for empowerment estimation.
            seed (int, optional): Seed for random number generator.
        """
        super(RequestSensingPermutationUnawareAgent, self)\
            .__init__(horizon, inventory_size, requests,
                      environment, num_sequences, seed)
        self.type = REQUEST_SENSING_PERMUTATION_UNAWARE_AGENT

    def add_new_reading(self, reading, readings):
        """
        Add a new sensor reading to readings if it is not already contained.

        Args:
            reading (int): Number representing number of routed requests.
            readings (list of ints): Previous readings.

        Returns:
            None
        """
        if len(readings) == 0:
            readings.append(reading)
        elif not np.in1d(reading, readings)[0]:
            readings.append(reading)
        else:
            pass

    def filter(self, environment_state):
        """
        Filter out uninteresting sensor readigs. A reading is filtered if it
        is smaller than the average number of accepted requests.

        Args:
            environment_state (int): Perceived state of the environment.

        Returns:
            environment_state (int): Updated signal.

        """
        if self.sensing_threshold is None:
            return environment_state
        elif environment_state < self.sensing_threshold:
            return 0
        else:
            return environment_state

    def sense(self):
        """
        Obtaine new sensor reading.

        Returns:
            sensor_reading (int): Number of requests in topology.
        """
        return self.filter(len(self.environment.routed_requests))


class RequestAwarePermutationUnawareAgent(RequestAwareAgent):

    def __init__(self, horizon, inventory_size, requests, environment, subsample=None,
                 num_sequences=1000, with_ilp=False, seed=1):
        """
        Initializes object.

        Args:
            horizon (int): Length of randomly sampled sequence, i.e., length of
                agent horizon.
            inventory_size (int): Number of elements agent can store in his
                inventory.
            requests (Array of requests.Request): Requests the agent can sample
                from.
            environment (environment.Environment): Environment object.
            subsample (int, optional): Subsample the requests instead of
                shuffling them.
            num_sequences (int, optional): Number of random sequences to perform
                for empowerment estimation.
            seed (int, optional): Seed for random number generator.
        """
        super(RequestAwarePermutationUnawareAgent, self) \
            .__init__(horizon, inventory_size, requests,
                      environment, subsample, num_sequences, with_ilp, seed)
        self.type = REQUEST_AWARE_PERMUTATION_UNAWARE_AGENT

    def add_new_reading(self, reading, readings):
        """

        Args:
            reading (int): Number representing number of routed requests.
            readings (list of ints): Previous readings.

        Returns:
            None
        """
        if len(readings) == 0:
            readings.append(reading)
        elif not np.in1d(reading, readings)[0]:
            readings.append(reading)
        else:
            pass

    def filter(self, environment_state):
        """
        Filter out uninteresting sensor readigs. A reading is filtered if it
        is smaller than the average number of accepted requests.

        Args:
            environment_state (int): Perceived state of the environment.

        Returns:
            environment_state (int): Updated signal.

        """
        if self.sensing_threshold is None:
            return environment_state
        elif environment_state < self.sensing_threshold:
            return 0
        else:
            return environment_state

    def sense(self):
        """
        Obtaine new sensor reading.

        Returns:
            sensor_reading (int): Number of requests in topology.
        """
        return self.filter(len(self.environment.routed_requests))

