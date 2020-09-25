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
                              old_private_op=0,old_public_op=0,old_decision=0)
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
    node_label = G.nodes
    np.random.seed(1)
    private_op = np.random.random(len(node_label))*2-1
    i = 0
    for node in node_label:
        G.nodes[node]['private_op'] = private_op[i]
        G.nodes[node]['old_private_op'] = private_op[i]
        G.nodes[node]['public_op'] = 0
        G.nodes[node]['decision'] =0
        G.nodes[node]['old_public_op'] = 0
        G.nodes[node]['old_decision'] = 0
        i+=1
    return G

def evole(args):
    graph2,p,alpha,beta,gamma,tau,lmd,rho,round_num = args
    graph = graph2.copy()
    initialize_graph(graph)
    update_decision(graph,lmd,rho,tau,True)
    roll(graph)
    t = 1
    while t<round_num:
        update_public_op(graph, p)
        update_private_op(graph,alpha,beta,gamma)
        update_decision(graph, lmd,rho,tau)
        roll(graph)
        t+=1
    t= 0
    results = []
    while t<1000:
        update_public_op(graph, p)
        update_private_op(graph, alpha, beta, gamma)
        update_decision(graph, lmd, rho, tau)
        roll(graph)
        t += 1
        re = statistic(graph)
        results.append(re)
    a = sum([x[0] for x in results])/len(results)
    b = sum([x[1] for x in results])/len(results)
    c = sum([x[2] for x in results])/len(results)
    return [a,b,c]

def evole2(graph,p,alpha,beta,gamma,tau,lmd,rho,round_num):
    initialize_graph(graph)
    update_decision(graph,lmd,rho,tau,True)
    print(statistic(graph))
    roll(graph)
    t = 1
    while t<round_num:
        update_public_op(graph, p)
        update_private_op(graph,alpha,beta,gamma)
        update_decision(graph, lmd,rho,tau)
        roll(graph)
        re = statistic(graph)
        print(str(t)+" "+ str(re))
        t+=1
    return


def update_private_op(graph,alpha,beta,gamma):
    for node in graph.nodes:
        nd = graph.nodes[node]
        neighbors = list(nx.neighbors(graph,node))
        pub = 0
        deb = 0
        for neigh in neighbors:
            pub += graph.nodes[neigh]['public_op']
            deb += graph.nodes[neigh]['old_decision']
        nd['private_op'] = alpha/len(neighbors)*pub+beta*nd['old_private_op']+ \
                           gamma/len(neighbors)*deb
    return

def update_public_op(graph,p):
    avg_op = sum(list(nx.get_node_attributes(graph,'old_public_op').values()))/len(graph.nodes)
    for node in graph.nodes:
        nd = graph.nodes[node]
        nd['public_op'] = (1-p)*nd['old_private_op']+ p*avg_op
    return 0

def update_decision(graph,lmd,rho,tau,first=False):
    if first == True:
        for node in graph.nodes:
            nd = graph.nodes[node]
            p_1 = (1 + nd['private_op']) / 2
            if random.random() < p_1:
                nd['decision'] = 1
            else:
                nd['decision'] = -1
    else:
        for node in graph.nodes:
            nd = graph.nodes[node]
            pi_1,pi_2 = cal_payoff(graph,node,rho)
            p_1 = lmd * (1 + nd['private_op']) / 2 + (1 - lmd)/(1+pow(np.e,tau*(pi_2-pi_1)))
            if random.random() < p_1:
                nd['decision'] = 1
            else:
                nd['decision'] = -1
    return

def cal_payoff(graph,node,rho):
    pi_1 = 0
    pi_2 = 0
    neighbors = list(nx.neighbors(graph, node))
    for neigh in neighbors:
        pi_1 += 1/2*(1+rho)*(1+graph.nodes[neigh]['old_decision'])
        pi_2 += 1/2 * (1-graph.nodes[neigh]['old_decision'])
    pi_1 = pi_1/len(neighbors)
    pi_2 = pi_2 / len(neighbors)
    return pi_1,pi_2

def roll(graph):
    for node in graph.nodes:
        nd = graph.nodes[node]
        nd['old_private_op'] = nd['private_op']
        nd['old_public_op'] =nd['public_op']
        nd['old_decision'] =nd['decision']

def statistic(graph):
    re = []
    re.append(sum(nx.get_node_attributes(graph, 'private_op').values())/len(graph.nodes))
    re.append(sum(nx.get_node_attributes(graph, 'public_op').values())/len(graph.nodes))
    re.append(sum(nx.get_node_attributes(graph, 'decision').values())/len(graph.nodes))
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
