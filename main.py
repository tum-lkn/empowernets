import numpy as np
import networkx as nx
import multiprocessing
import json
import agents
import environment
import requests


def evaluate(args):
    degree_constraint = args["degree_constraint"]
    num_nodes = args["num_nodes"]
    num_edges = args["num_edges"]
    num_requests = args["num_requests"]
    num_seq = args["num_seq"]
    run = args["run"]
    filtering = args["filtering"]
    order = args["order"]
    horizon = args["horizon"]

    filename = ("sensing-all-requests-{:d}-edges-{:d}" +
                "-sequences-run-{:d}-filtering-{:s}-order-{:d}-horizon-{:d}.json").format(
        num_edges, num_seq, run, "yes" if filtering else "no", order, horizon
    )
    print num_edges, num_seq, filename

    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    env = environment.Environment(g, degree_limit=degree_constraint)

    rqs = []
    count = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            rqs.append(requests.Request(i, j, count))
            count += 1

    agent = agents.AgentFactory.produce(
        order=order,
        horizon=horizon,
        inventory_size=num_edges,
        requests=rqs,
        environment=env,
        subsample=None,
        num_sequences=num_seq,
        seed=run
    )
    agent.inventory = num_edges
    if filtering:
        agent.sensing_threshold = 0

    chosen_actions = []
    sensed_states = []
    environment_states = []
    empowerments = []
    filter_values = []

    best_empowerment = 0
    best_step = 0
    filter_threshold = agent.sensing_threshold
    i = 0
    while i < 500 or i - best_step < 100:
        i += 1
        a, e = agent.choose_next_action()
        a.apply()

        if agent.sensing_threshold != filter_threshold:
            print "\tupdated threshold {:.2f} --> {:.2f}".format(filter_threshold, agent.sensing_threshold)
            best_step = i
            best_empowerment = 0
            filter_threshold = agent.sensing_threshold

        chosen_actions.append(a.name)
        empowerments.append(e)
        sensed_states.append(env.state)
        environment_states.append({
            "topo": [(u, v, d) for u,v,d in env.topology.edges(data=True)],
            "paths": env.routed_requests.copy()
        })
        if filter_threshold is not None:
            filter_values.append(filter_threshold)

        num_edges = agent.inventory
        for u, v, d in agent.environment.topology.edges(data=True):
            num_edges += d["capacity"]
        for d in agent.environment.routed_requests.itervalues():
            num_edges += len(d) - 1

        if e > best_empowerment:
            best_step = i
            best_empowerment = e

        if i % 10 == 0:
            print i, "\t",
            print e, "\t", np.exp(e), "\t", best_empowerment, "\t", np.exp(best_empowerment), "\t",
            print agent.inventory, "\t", i - best_step,
            if agent.sensing_threshold is None:
                print ""
            else:
                print "\t", agent.sensing_threshold
            print "----"

        if i % 10 == 0:
            print "dump"
            with open(filename, "w") as fh:
                json.dump([chosen_actions, empowerments, sensed_states, environment_states, filter_values], fh)
            print "dump-finished"

    with open(filename, "w") as fh:
        json.dump([chosen_actions, empowerments, sensed_states, environment_states, filter_values], fh)


if __name__ == "__main__":
    num_nodes = 30
    degree_constraint = 3
    job = {
        "degree_constraint": degree_constraint,
        "num_nodes": num_nodes,
        "num_edges": num_nodes * degree_constraint / 2,
        "num_requests": 45,
        "num_seq": 1000,
        "horizon": 15,
        "filtering": False,
        "run": 5,
        "order": agents.REQUEST_SENSING_PERMUTATION_UNAWARE_AGENT
    }
    jobs = []
    for horizon in [5, 10, 15]:
        tmp = job.copy()
        tmp["horizon"] = horizon
        evaluate(tmp)
        jobs.append(tmp)

    # pool = multiprocessing.Pool(processes=len(jobs))
    # pool.map(evaluate, jobs)
    # pool.close()


