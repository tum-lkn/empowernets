

class Action(object):

    def __init__(self, agent, name):
        """
        Initialzes object.

        Args:
            agent (agents.RequestSensingAgent): Agent performing the action.
        """
        self.agent = agent
        self.name = name

    def apply(self):
        """
        Apply action to environment.

        Returns:
            True if successful else False
        """
        raise NotImplementedError()

    def rollback(self):
        """
        Rollback action, i.e., move environment to state before action was
        applied.

        Returns:
            True if successful else False.
        """
        raise NotImplementedError()


class AddEdge(Action):

    def __init__(self, agent):
        """
        Initializes object.

        Args:
            agent (agents.AbstractAgent): An agent that performs those edges.
        """
        super(AddEdge, self).__init__(agent, "AddEdge")
        self.applicable = self.agent.inventory > 0
        self.applied = False
        self.edge = None
        self.edge_value = 0

    def apply(self):
        """
        Tries to add an edge to the topology. The action succeeds if it can
        be applied. If the action is not applicable nothing happens. If it is
        applicable then the agents inventory is reduced.

        Returns:
            applied (bool):

        """
        self.applied = False
        if self.applicable:
            if self.edge is None:
                # Let the agent select an edge to place. If this returns None
                # then the agent was not able to pick a suitable edge and
                # this action becomes a noop.
                self.edge = self.agent.choose_edge_to_place(exclude_existing=False)
            if self.edge is None:
                self.edge = -1
            if self.edge != -1:
                self.edge_value = self.agent.environment.add_edge(self.edge, value=self.agent.edge_value)
                self.applied = True
                self.agent.inventory -= self.edge_value
            self.applied = True
        return self.applied

    def rollback(self):
        """
        Rollback this action, i.e., remove edge again from the topology.

        Returns:
            None
        """
        if self.applied:
            value = self.agent.environment.remove_edge(self.edge, value=self.agent.edge_value)
            assert value == self.edge_value, "Value of added and removed " \
                    + "do not match"
            self.agent.inventory += value
            self.applied = False
            self.edge_value = 0


class TakeEdge(Action):

    def __init__(self, agent):
        """
        Initializes object.

        Args:
            agent (agents.AbstractAgent): An agent that performs those edges.
        """
        super(TakeEdge, self).__init__(agent, "TakeEdge")
        self.applied = False
        self.applicable = self.agent.environment.number_of_edges > 0
        self.edge = None
        self.edge_value = 0

    def apply(self):
        """
        Remove an edge from the topology.

        Returns:
            applied

        """
        if self.applicable:
            if self.edge is None:
                self.edge = self.agent.choose_edge_to_take()
            # choose edge might return None if no edge could be found. Thus
            # check again.
            if self.edge is None:
                # Set to minus one as None has other meanings in later contexts
                self.edge = -1
            if self.edge != -1:
                self.edge_value = self.agent.environment.remove_edge(self.edge, value=self.agent.edge_value)
                self.agent.inventory += self.edge_value
                self.applied = True
        return self.applied

    def rollback(self):
        if self.applied:
            self.applied = False
            value = self.agent.environment.add_edge(self.edge, value=self.agent.edge_value)
            assert value == self.edge_value, "Value of removed and readded " + \
                "edge to not match"
            self.agent.inventory -= value
            self.edge_value = 0


class AddRequest(Action):

    def __init__(self, agent):
        """
        Initializes object.

        Args:
            agent (agents.AbstractAgent): An agent that performs those edges.
        """
        super(AddRequest, self).__init__(agent, "AddRequest")
        self.request = self.agent.choose_request_to_place()
        # If request is already in environment do nothing
        self.applicable = self.request.mask not in self.agent.environment.routed_requests.keys()
        self.applied = False
        self.path = None

    def apply(self):
        """
        Add a s-t pair to the topology using shortest path routing. Depending
        on the value of self.path a preconfigured path is chosen. in case of
        success corresponding resources are allocated.

        Returns:
            applied (bool): True if request has been added zero false.

        """
        if self.applicable:
            if self.path is None:
                self.path = self.agent.environment.add_request(self.request)
            else:
                self.agent.environment.add_request(self.request, self.path)
            if self.path is not None:
                self.applied = True
        return self.applied

    def rollback(self):
        """
        Remove request again from the topology and release resources.

        Returns:
            None

        """
        if self.applied:
            self.agent.environment.remove_request(self.request)
            self.applied = False


class RemoveRequest(Action):

    def __init__(self, agent):
        """
        Initializes object.

        Args:
            agent (agents.AbstractAgent): An agent that performs those edges.
        """
        super(RemoveRequest, self).__init__(agent, "RemoveRequest")
        self.request = self.agent.choose_request_to_remove()
        # If request is already in environment do nothing
        if self.request is None:
            self.applicable = False
        else:
            self.applicable = self.request.mask in self.agent.environment.routed_requests.keys()
        self.applied = False
        self.path = None

    def apply(self):
        """
        Remove request from the topology and free resources.

        Returns:
            applied (bool): True if successful else false.

        """
        if self.applicable:
            self.path = self.agent.environment.remove_request(self.request)
            self.applied = True
        return self.applied

    def rollback(self):
        """
        Add removed request againt to the topology.

        Returns:
            None
        """
        if self.applied:
            self.agent.environment.add_request(self.request, self.path)
            self.applied = False


class DoNothing(Action):

    def __init__(self, agent):
        """
        Initializes object.

        Args:
            agent (agents.AbstractAgent): An agent that performs those edges.
        """
        super(DoNothing, self).__init__(agent, "DoNothing")

    def apply(self):
        """
        Do nothing always succeeds.

        Returns:
            True

        """
        return True

    def rollback(self):
        """
        Do nothing again.

        Returns:
            True

        """
        return True


class IncreaseCapacity(Action):

    def __init__(self, agent):
        """
        Initializes object.

        Args:
            agent (agents.AbstractAgent): An agent that performs those edges.
        """
        super(IncreaseCapacity, self).__init__(agent, "IncreaseCapacity")
        self.applicable = self.agent.inventory > 0
        self.applied = False
        self.edge = None
        self.edge_value = 0

    def apply(self):
        """
        Try to increase capacity on an edge of the agents choice.

        Returns:
            applied (bool): True if successful else False.
        """
        if self.applicable:
            if self.edge is None:
                self.edge = self.agent.choose_edge_to_change(True, True)
            if self.edge is None:
                self.edge = -1
            if self.edge != -1:
                # Since an existing edge is chosen by the agent enviornment will
                # increase the capacity.
                self.edge_value = self.agent.environment.add_edge(self.edge)
                self.applied = True
                self.agent.inventory -= self.edge_value
        return self.applied

    def rollback(self):
        """
        Remove previously added capacity back to topology.

        Returns:
            None
        """
        if self.applied:
            value = self.agent.environment.remove_edge(self.edge)
            assert value == self.edge_value, "Value of added and removed " \
                                             + "do not match"
            self.agent.inventory += 1
            self.applied = False
            self.edge_value = 0


class DecreaseCapacity(Action):

    def __init__(self, agent):
        """
        Initializes object.

        Args:
            agent (agents.AbstractAgent): An agent that performs those edges.
        """
        super(DecreaseCapacity, self).__init__(agent, "DecreaseCapacity")
        self.applied = False
        self.applicable = self.agent.environment.number_of_edges > 0
        self.edge = None
        self.edge_value = 0

    def apply(self):
        """
        Decrease capacity on an edge of the agent's choice.

        Returns:
            applied (bool): True if successful else False.
        """
        if self.applicable:
            if self.edge is None:
                self.edge = self.agent.choose_edge_to_change(True, False)
            if self.edge is None:
                self.edge = -1
            if self.edge != -1:
                self.edge_value = self.agent.environment.remove_edge(self.edge)
                self.agent.inventory += self.edge_value
                self.applied = True
        return self.applied

    def rollback(self):
        """
        Add repviously removed capacity back to topology.

        Returns:
            None
        """
        if self.applied:
            self.applied = False
            value = self.agent.environment.add_edge(self.edge)
            assert value == self.edge_value, "Value of removed and readded " + \
                                             "edge to not match"
            self.agent.inventory -= value
            self.edge_value = 0
