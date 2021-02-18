import networkx as nx
import numpy as np
import random

def read_data(meta_str,data_str):
    G = nx.Graph()
    node_label = []
    with open(meta_str, 'r') as f:
        line = f.readline().strip()
        while(line!=''):
            data = line.split('\t')
            node_label.append(data[0])
            G.add_node(data[0],private_op=0,public_op=0,decision=0,\
                              old_private_op=0,old_public_op=0,payoff=0)
            line = f.readline().strip()
    f.close()
    edges = []
    with open(data_str, 'r') as f:
        line = f.readline().strip()
        while(line!=''):
            data = line.split('\t')
            if len(data) == 1:
                data = line.split(' ')
            if node_label.index(data[1])<node_label.index(data[2]):
                edges.append((data[1],data[2]))
            else:
                edges.append((data[2], data[1]))
            line = f.readline().strip()
    f.close()
    return G,node_label,edges

def edge_weights(edges,pc):
    from collections import Counter
    edges_w = Counter(edges)
    new_edges =edges_w.most_common(int(len(edges_w) * pc))
    return new_edges

def construct_graph(G,edges_w):
    for edge in edges_w:
        G.add_edge(edge[0][0],edge[0][1])
    return G

def initialize_graph(G):
    private_op = np.random.random(len(G.nodes))*2-1
    i = 0
    for node in G.nodes:
        G.nodes[node]['private_op'] = private_op[i]
        G.nodes[node]['old_private_op'] = private_op[i]
        G.nodes[node]['public_op'] = 0.0
        G.nodes[node]['decision'] =0
        G.nodes[node]['old_public_op'] = 0.0
        G.nodes[node]['payoff'] = 0.0
        G.nodes[node]['state'] = 1
        i+=1
    return G

def initialize_graph_with_deceivers(G, percent):
    node_label = G.nodes
    num_deceivers = int(len(node_label)*percent)
    private_op = np.random.random(len(G.nodes)) * 2 - 1
    i = 0
    for node in node_label:
        G.nodes[node]['private_op'] = private_op[i]
        G.nodes[node]['old_private_op'] = private_op[i]
        G.nodes[node]['public_op'] = 0
        G.nodes[node]['decision'] =0
        G.nodes[node]['old_public_op'] = 0
        G.nodes[node]['payoff'] = 0.0
        G.nodes[node]['state'] = 1
        i+=1
    nodes = list(node_label)
    deceivers = random.sample(nodes,num_deceivers)
    for nd in deceivers:
        G.nodes[nd]['state'] = -1
        G.nodes[nd]['private_op'] = -1
    # if not sum(nx.get_node_attributes(G, 'state').values())==160:
    #     print("error")
    return G

def evole(args):
    graph2,p,alpha,beta,gamma,tau,lmd,b,c,round_num,t0 = args
    graph = graph2.copy()
    graph = initialize_graph(graph)
    update_decision(graph,lmd,b,c,tau,True)
    cal_payoff(graph,b,c)
    roll(graph)
    t = 1
    results = []
    while t<round_num:
        update_public_op(graph, p)
        for i in range(t0):
            update_decision(graph, lmd,b,c,tau)
        update_private_op(graph, alpha, beta, gamma)
        roll(graph)
        t+=1
    re = statistic(graph)

    t=0
    while t<t0:
        update_decision(graph, lmd, b,c, tau)
        t += 1
        re = statistic(graph)
        results.append(re)
    a1 = sum([x[0] for x in results])/len(results)
    b1 = sum([x[1] for x in results])/len(results)
    c1 = sum([x[2] for x in results])/len(results)
    return [a1,b1,c1]

def deceiver_evolve(args):
    graph2, p, alpha, beta, gamma, tau, lmd, b, c, round_num, t0, per = args
    graph = graph2.copy()
    graph = initialize_graph_with_deceivers(graph,per)
    update_decision(graph, lmd, b, c, tau, True)
    cal_payoff(graph, b, c)
    roll(graph)
    t = 1
    results = []
    while t < round_num:
        update_public_op(graph, p)
        for i in range(t0):
            update_decision(graph, lmd, b, c, tau)
        update_private_op(graph, alpha, beta, gamma)
        roll(graph)
        t += 1
    re = statistic(graph)

    t = 0
    while t < t0:
        update_decision(graph, lmd, b, c, tau)
        t += 1
        re = statistic(graph)
        results.append(re)
    a1 = sum([x[0] for x in results]) / len(results)
    b1 = sum([x[1] for x in results]) / len(results)
    c1 = sum([x[2] for x in results]) / len(results)
    return [a1, b1, c1]

def evole2(graph,p,alpha,beta,gamma,tau,lmd,b,c,round_num):
    initialize_graph_with_deceivers(graph)
    update_decision(graph,lmd,b,c,tau,True)
    print(statistic(graph))
    roll(graph)
    t = 1
    while t<round_num:
        update_public_op(graph, p)
        update_private_op(graph,alpha,beta,gamma)
        update_decision(graph, lmd,b,c,tau)
        roll(graph)
        re = statistic(graph)
        print(str(t)+" "+ str(re))
        t+=1
    return

def get_opinion(graph,p,alpha,beta,gamma,tau,lmd,b,c,round_num,t0):
    graph = initialize_graph(graph)
    r_fit = []
    r_fit.append(list(nx.get_node_attributes(graph, 'private_op').values()))
    update_decision(graph, lmd, b, c, tau, True)
    cal_payoff(graph, b, c)
    roll(graph)
    t = 1
    while t < round_num:
        update_public_op(graph, p)
        for i in range(t0):
            update_decision(graph, lmd, b, c, tau)
        update_private_op(graph, alpha, beta, gamma)
        r_fit.append(list(nx.get_node_attributes(graph, 'private_op').values()))
        roll(graph)
        t += 1
    return r_fit

def update_private_op(graph,alpha,beta,gamma):
    for node in graph.nodes:
        nd = graph.nodes[node]
        if nd['state'] == -1:
            continue
        neighbors = list(nx.neighbors(graph,node))
        pub = 0
        deb = 0
        for neigh in neighbors:
            pub += graph.nodes[neigh]['public_op']
            deb += graph.nodes[neigh]['decision']
        nd['private_op'] = alpha*pub/len(neighbors)+beta*nd['old_private_op']+ \
                           gamma*deb/len(neighbors)
    return

def update_public_op(graph,p):
    avg_op = sum(list(nx.get_node_attributes(graph,'old_public_op').values()))/len(graph.nodes)
    for node in graph.nodes:
        nd = graph.nodes[node]
        if nd['state'] == -1:
            if random.random()<0.5:
                nd['public_op'] = 1
            else:
                nd['public_op'] = avg_op
            continue
        nd['public_op'] = (1-p)*nd['old_private_op']+ p*avg_op
    return 0

def update_decision(graph,lmd,b,c,tau,first=False):
    if first == True:
        for node in graph.nodes:
            nd = graph.nodes[node]
            p_1 = (1 + nd['private_op']) / 2
            if random.random() < p_1:
                nd['decision'] = 1
            else:
                nd['decision'] = -1
    else:
        node = random.sample(graph.nodes,1)[0]
        fit_pro = cal_fit_pro(graph,tau,node)
        nd = graph.nodes[node]
        fit = lmd*(1+nd['private_op']) / 2+  (1-lmd)*fit_pro

        if random.random()<fit:
            nd['decision'] = 1
        else:
            nd['decision'] = -1
        cal_s_payoff(graph,b,c,node)
        neighbors = list(nx.neighbors(graph, node))
        for neighbor in neighbors:
            cal_s_payoff(graph,b,c,neighbor)
    return

def cal_fit_pro(graph,tau,node):
    pi_1 = 0
    pi_2 = 0
    neighbors = list(nx.neighbors(graph, node))
    for neigh in neighbors:
        if graph.nodes[neigh]['decision'] == 1:
            pi_1 += 1-tau+tau*graph.nodes[neigh]['payoff']
        else:
            pi_2 += 1-tau+tau*graph.nodes[neigh]['payoff']

    if pi_1+pi_2 == 0:
        if pi_1>0:
            fit = 1
        else: fit = 0
    else:fit = pi_1/(pi_1+pi_2)
    return fit

def cal_payoff(graph,b,c):
    for node in graph.nodes:
        nd = graph.nodes[node]
        neighbors = list(nx.neighbors(graph, node))
        payoff = 0.0
        if nd['decision'] == 1:
            for neigh in neighbors:
                payoff += 1 / 2 * ((b - c) * (1 + graph.nodes[neigh]['decision']) + (-c) * (
                        1 - graph.nodes[neigh]['decision']))
        else:
            for neigh in neighbors:
                payoff += 1 / 2 * b * (1 + graph.nodes[neigh]['decision'])
        nd['payoff'] = payoff
    return

def cal_s_payoff(graph,b,c,node):
    nd = graph.nodes[node]
    neighbors = list(nx.neighbors(graph, node))
    payoff = 0
    if nd['decision'] == 1:
        for neigh in neighbors:
            payoff += 1 / 2 * ((b - c) * (1 + graph.nodes[neigh]['decision']) + (-c) * (
                    1 - graph.nodes[neigh]['decision']))
    else:
        for neigh in neighbors:
            payoff += 1 / 2 * b * (1 + graph.nodes[neigh]['decision'])
    nd['payoff'] = payoff

def roll(graph):
    for node in graph.nodes:
        nd = graph.nodes[node]
        nd['old_private_op'] = nd['private_op']
        nd['old_public_op'] =nd['public_op']

def statistic(graph):
    re = []
    re.append(sum(nx.get_node_attributes(graph, 'private_op').values())/len(graph.nodes))
    re.append(sum(nx.get_node_attributes(graph, 'public_op').values())/len(graph.nodes))
    decision = (sum(nx.get_node_attributes(graph, 'decision').values())+len(graph.nodes))/2

    # 合作的个体比例
    re.append(decision/len(graph.nodes))
    # print(nx.get_node_attributes(graph, 'private_op').values())
    # print(nx.get_node_attributes(graph, 'public_op').values())
    # print(nx.get_node_attributes(graph, 'decision').values())
    return re

def deal_unconnected(graph):
    connected_p = list(nx.connected_components(graph))
    i = 0
    while i<len(connected_p)-1:
        a1 = random.sample(connected_p[i],1)[0]
        a2 = random.sample(connected_p[i+1],1)[0]
        graph.add_edge(a1,a2)
        i+=1

# data_file = "/home/randy/Documents/social net/data/tij_pres_InVS13.dat"
# meta_file = "/home/randy/Documents/social net/data/metadata_InVS13.dat"
# G,node_label,edges = read_data(meta_file,data_file)
# edges_w = edge_weights(edges,1)
# G = construct_graph(G,edges_w)
# deal_unconnected(G)
# evole(G,p=0.1,alpha=0.3,beta=0.5,gamma=0.2,tau=20,lmd=0.1,rho=1,round_num=1000)
