import numpy as np


class AbstractRequest(object):
    
    def __init__(self, mask):
        self.mask = mask


class Request(AbstractRequest):

    def __init__(self, s, t, mask, commodity=None):
        """
        Initializes object.

        Args:
            s (int): Id of source node.
            t (int): Id of target node.
            mask (int): Boolean mask for request. Only one single bit of the
                int is set to one.
            commodity (int, optional): Identifier for commodity. Enables to have
                different ids for requests for the same s-t pair.
        """
        super(Request, self).__init__(mask)
        # tmp = np.log2(float(mask))
        # assert np.abs(tmp - int(tmp)) < 1e-6, "More than one bit in mask set to one"
        self.source = s
        self.target = t
        self.mask = mask
        self.commodity = commodity


class TrafficMatrixRequest(AbstractRequest):
    """
    Requests represents a traffic matrix. That is a set of s-t pairs. The
    demand between each pair is integral. Each demand is modeled as a set of
    s-t pairs with same source and destination.
    """

    def __init__(self, tm, mask, seed=1):
        """
        Initialize object.

        Args:
            tm (dict): Dict of dicts of ints, traffic matrix where keys are
                nodes and values demand between each node pair.
            mask (int): Identifier for the mask.
            seed (int, optional): Integer for random number generator used to
                generate unit flows.
        """
        super(TrafficMatrixRequest, self).__init__(mask)
        self.tm = tm
        self.mask = mask
        self.seed = seed

        self.commodities = {}
        """
        Use this attribute to map each OD flow to a unique commodity id.
        """
        self.od_pairs = []
        count = 0
        for l1 in self.tm:
            if l1 not in self.commodities:
                self.commodities[l1] = {}
            for l2 in self.tm[l1]:
                self.commodities[l1][l2] = count
                if self.tm[l1][l2] > 0:
                    self.od_pairs.append((l1, l2))
                count += 1

    def copy_tm(self):
        cpy = {}
        for k in self.tm:
            if k not in cpy:
                cpy[k] = {}
            for l in self.tm[k]:
                cpy[k][l] = self.tm[k][l]
        return cpy

    def request_iter(self):
        """
        Generate as many unit flows as routing units are requested. Randomly
        generate unit flows for each commodity, each pair in the TM.

        Yields:
            Request
        """
        random = np.random.RandomState(seed=self.seed)
        indices = np.arange(len(self.od_pairs))
        local_tm = {}
        for k, d in self.tm.iteritems():
            local_tm[k] = d.copy()
        count = 0
        while indices.size > 0:
            idx = random.choice(indices)
            l1, l2 = self.od_pairs[idx]
            local_tm[l1][l2] -= 1
            if local_tm[l1][l2] == 0:
                if indices.size == 1:
                    indices = np.array([])
                else:
                    indices = indices[indices != idx]

            rq = Request(l1, l2, count, self.commodities[l1][l2])
            count += 1
            yield rq



