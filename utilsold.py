import random
import networkx as nx
from tqdm import tqdm_notebook
import time

def diff_of_list(lst1,lst2):
    return list(set(lst1)-set(lst2))

def cal_temporality(set1,set2):
    setU = set1|set2
    setC = set1&set2
    return float(len(set1)+len(set2)-2*len(setC))/float(len(setU))

def constructG(meta_str,data_str):
    G = nx.Graph()
    node_label = []
    with open(meta_str, 'r') as f:
        line = f.readline().strip()
        while(line!=''):
            data = line.split('\t')
            node_label.append(data[0])
            G.add_node(data[0],state=0,char=0)
            line = f.readline().strip()
    f.close()
    edge_with_time = []
    with open(data_str, 'r') as f:
        line = f.readline().strip()
        while(line!=''):
            data = line.split('\t')
            if len(data) == 1:
                data = line.split(' ')
            if node_label.index(data[1])<node_label.index(data[2]):
                edge_with_time.append([int(data[0]),data[1],data[2]])
            else:
                edge_with_time.append([int(data[0]), data[2], data[1]])
            line = f.readline().strip()
    f.close()
    return G,node_label,edge_with_time

def initialAgent(G,node_label,a1,a2):
    a3 = 1- a1 -a2
    stub_agent = random.sample(node_label,int(a1*len(node_label)))
    x2 = diff_of_list(node_label,stub_agent)
    con_agent = random.sample(x2,int(a2*len(node_label)))
    x2 = diff_of_list(x2,con_agent)
    uncon_agent = random.sample(x2,int(a3*len(node_label)))
    for node in node_label:
        G.nodes[node]['state'] = random.sample([0,1],1)[0]
        G.nodes[node]['state2'] = 0
    for node in stub_agent:
        G.nodes[node]['state'] = 1
        G.nodes[node]['char'] = 0
    for node in con_agent:
        G.nodes[node]['char'] = 1
    for node in uncon_agent:
        G.nodes[node]['char'] = 2
    return G

def initialAgent2(G,a1):
    node_label = G.nodes
    stub_agent = random.sample(node_label,int(a1*len(node_label)))
    uncon_agent = diff_of_list(node_label,stub_agent)
    for node in node_label:
        G.nodes[node]['state'] = random.sample([0,1],1)[0]
        G.nodes[node]['state2'] = 0
    for node in stub_agent:
        G.nodes[node]['state'] = 1
        G.nodes[node]['char'] = 0
    for node in uncon_agent:
        G.nodes[node]['char'] = 1
    return G

def evolve(G,g,beta,alpha):
    # 开始更新
    tx = g
    while (tx > 0):
        for node in G.nodes:
            if G.nodes[node]['char'] == 0:
                continue
            else:
                neighbors = list(G.neighbors(node))
                if len(neighbors) > 0:
                    prob = random.random()
                    if prob < beta:
                        G.nodes[node]['state'] = 1 - G.nodes[node]['state']
                    elif prob < alpha + beta:
                        continue
                    else:
                        selected = random.sample(neighbors, 1)[0]
                        G.nodes[node]['state'] = G.nodes[selected]['state']
        tx -= 1
    states = nx.get_node_attributes(G, 'state')
    pos_state = sum(list(states.values()))
    return G,pos_state

def evolve2(G,g,beta,alpha):
    # 开始更新
    tx = g
    while (tx > 0):
        for node in G.nodes:
            if G.nodes[node]['char'] == 0:
                continue
            else:
                neighbors = list(G.neighbors(node))
                if len(neighbors) > 0:
                    prob = random.random()
                    if prob < beta:
                        G.nodes[node]['state2'] = 1 - G.nodes[node]['state']
                    elif prob < alpha + beta:
                        continue
                    else:
                        selected = random.sample(neighbors, 1)[0]
                        G.nodes[node]['state2'] = G.nodes[selected]['state']
        for node in G.nodes:
            if G.nodes[node]['char'] == 0:
                continue
            else:
                G.nodes[node]['state'] = G.nodes[node]['state2']
        tx -= 1
    states = nx.get_node_attributes(G, 'state')
    pos_state = sum(list(states.values()))
    return G,pos_state

def evolution (args):
    G,edge_with_time0,node_label,alpha,g,Gt,delta_t,epoch,a1,a2,beta = args
    w = int(Gt/g)
    t_max = edge_with_time0[-1][0]
    s = []
    for k in range(epoch):
        print(a1,k)
        G2 = G.copy()
        G2 = initialAgent(G2,node_label,a1,a2)
        init_n = 0
        # 网络更新的次数
        for i in range(w):
            t0 = edge_with_time0[init_n][0]
            temp_net = nx.Graph(G)

            # 网络结构更新
            while(edge_with_time0[init_n][0]<=t0+delta_t):
                temp_net.add_edge(edge_with_time0[init_n][1],edge_with_time0[init_n][2])
                if len(edge_with_time0) == init_n+1:
                    init_n = 0
                    t0 = t0-t_max+edge_with_time0[0][0]
                    continue
                init_n+=1
            #print(nx.number_of_edges(temp_net))

            # 开始更新
            tx = g
            while(tx>0):
                for node in node_label:
                    if G2.nodes[node]['char'] == 0:
                        continue
                    #elif G2.nodes[node]['char'] == 1:
                        #prob = random.random()
                        #if prob<beta:
                           # G2.nodes[node]['state'] = 1-G2.nodes[node]['state']
                        #elif prob<1+beta:
                         #   G2.nodes[node]['state'] = 1
                        #else:
                         #   neighbors = list(temp_net.neighbors(node))
                         #   if len(neighbors)>0:
                         #       selected = random.sample(neighbors,1)[0]
                         #       G2.nodes[node]['state'] = G2.nodes[selected]['state']
                    else:
                        neighbors = list(temp_net.neighbors(node))
                        if len(neighbors)>0:
                            prob = random.random()
                            if prob<beta:
                                G2.nodes[node]['state'] = 1-G2.nodes[node]['state']
                            elif prob<alpha+beta:
                                continue
                            else:
                                selected = random.sample(neighbors,1)[0]
                                G2.nodes[node]['state'] = G2.nodes[selected]['state']
                tx-=1
        
        states = nx.get_node_attributes(G2,'state')
        pos_state = sum(list(states.values()))
        s.append(pos_state)
        time.sleep(0.01)
    return (sum(s)/len(s))

def evolution2 (args):
    G,edge_with_time0,node_label,alpha,g,Gt,delta_t,epoch,a1,a2,beta = args
    w = int(Gt/g)
    t_max = edge_with_time0[-1][0]
    s = []
    for k in range(epoch):
        print(a1,k)
        G2 = G.copy()
        G2 = initialAgent(G2,node_label,a1,a2)
        init_n = 0
        # 网络更新的次数
        for i in range(w):
            t0 = edge_with_time0[init_n][0]
            temp_net = nx.Graph(G)

            # 网络结构更新
            while(edge_with_time0[init_n][0]<=t0+delta_t):
                temp_net.add_edge(edge_with_time0[init_n][1],edge_with_time0[init_n][2])
                if len(edge_with_time0) == init_n+1:
                    init_n = 0
                    t0 = t0-t_max+edge_with_time0[0][0]
                    continue
                init_n+=1
            #print(nx.number_of_edges(temp_net))

            # 开始更新
            tx = g
            while(tx>0):
                for node in node_label:
                    if G2.nodes[node]['char'] == 0:
                        continue
                    #elif G2.nodes[node]['char'] == 1:
                        #prob = random.random()
                        #if prob<beta:
                           # G2.nodes[node]['state'] = 1-G2.nodes[node]['state']
                        #elif prob<1+beta:
                         #   G2.nodes[node]['state'] = 1
                        #else:
                         #   neighbors = list(temp_net.neighbors(node))
                         #   if len(neighbors)>0:
                         #       selected = random.sample(neighbors,1)[0]
                         #       G2.nodes[node]['state'] = G2.nodes[selected]['state']
                    else:
                        neighbors = list(temp_net.neighbors(node))
                        if len(neighbors)>0:
                            prob = random.random()
                            if prob<beta:
                                G2.nodes[node]['state2'] = 1-G2.nodes[node]['state']
                            elif prob<alpha+beta:
                                G2.nodes[node]['state2'] = G2.nodes[node]['state']
                                continue
                            else:
                                selected = random.sample(neighbors,1)[0]
                                G2.nodes[node]['state2'] = G2.nodes[selected]['state']
                for node in node_label:
                    if G2.nodes[node]['char'] == 0:
                        continue
                    else:
                        G2.nodes[node]['state'] = G2.nodes[node]['state2']
                tx-=1
        
        states = nx.get_node_attributes(G2,'state')
        pos_state = sum(list(states.values()))
        s.append(pos_state)
        time.sleep(0.01)
    return (sum(s)/len(s))

def evolution_onstatic (args):
    G,edge_with_time0,node_label,alpha,g,Gt,delta_t,epoch,a1,a2,beta = args
    t_max = edge_with_time0[-1][0]
    s = []
    # 开始更新
    for k in range(epoch):
        G2 = initialAgent(G,node_label,a1,a2)
        tx = Gt
        print(a1,k)
        for ed in edge_with_time0:
            G2.add_edge(ed[1],ed[2])
        while(tx>0):
            for node in node_label:
                if G2.nodes[node]['char'] == 0:
                    continue
                else:
                    neighbors = list(G2.neighbors(node))
                    if len(neighbors)>0:
                        prob = random.random()
                        if prob<beta:
                            G2.nodes[node]['state2'] = 1-G2.nodes[node]['state']
                        elif prob<alpha+beta:
                            continue
                        else:
                            selected = random.sample(neighbors,1)[0]
                            G2.nodes[node]['state2'] = G2.nodes[selected]['state']
            for node in node_label:
                    if G2.nodes[node]['char'] == 0:
                        continue
                    else:
                        G2.nodes[node]['state'] = G2.nodes[node]['state2']
            tx-=1
        states = nx.get_node_attributes(G2,'state')
        pos_state = sum(list(states.values()))
        s.append(pos_state)
        time.sleep(0.01)
    return (sum(s)/len(s))

def weigh_G(edge_with_time,G):
    G2 = G.to_directed()
    print(len(G2))
    edge_with_w = {}
    for ed in edge_with_time:
        if (ed[1],ed[2]) in edge_with_w.keys():
            edge_with_w[(ed[1],ed[2])] = edge_with_w[(ed[1],ed[2])]+1
        else:
            edge_with_w[(ed[1],ed[2])] = 1
    for key,value in edge_with_w.items():
        G2.add_edge(key[0],key[1])
        G2.add_edge(key[1], key[0])
        G2[key[0]][key[1]]['w'] = value
        G2[key[1]][key[0]]['w'] = value
    for node in G2.nodes:
        neighbors = list(nx.neighbors(G2,node))
        sumw = 0
        for n in neighbors:
            sumw += G2[node][n]['w']
        for n in neighbors:
            G2[node][n]['w'] = G2[node][n]['w']/sumw
    return G2

def evolution_onaggregate (args):
    G,edge_with_w,node_label,alpha,g,Gt,delta_t,epoch,a1,a2,beta = args
    # 加权处理
    G2 = G.copy()
    s= []
    # 开始更新
    for k in range(epoch):
        G2 = initialAgent(G,node_label,a1,a2)
        tx = Gt
        print(a1,k)
        while(tx>0):
            for node in node_label:
                if G2.nodes[node]['char'] == 0:
                    continue
                else:
                    neighbors = list(G2.neighbors(node))
                    if len(neighbors)>0:
                        prob = random.random()
                        if prob<beta:
                            G2.nodes[node]['state'] = 1-G2.nodes[node]['state']
                        elif prob<alpha+beta:
                            continue
                        else:
                            p_l = []
                            for nei in neighbors:
                                p_l.append(G2[node][nei]['w'])
                            selected = sample_with_p(p_l,1)
                            G2.nodes[node]['state'] = G2.nodes[neighbors[selected[0]]]['state']
            tx-=1
        states = nx.get_node_attributes(G2,'state')
        pos_state = sum(list(states.values()))
        s.append(pos_state)
        time.sleep(0.01)
    return (sum(s)/len(s))

def sample_with_p(p_l,n):
    l = []
    for i in range(n):
        r = random.random()
        for p in p_l:
            r-= p
            if r<=0:
                l.append(p_l.index(p))
                break
    return l

def evolution_on_synthetic(args):
    N, p, zeta, g, beta, alpha = args
    results1 = []
    results2 = []
    for epoch in range(500):
        print(zeta,epoch)
        G0 = nx.erdos_renyi_graph(N,p)
        G0 = initialAgent2(G0,zeta)
        G1 = G0.copy()
        G2 = G0.copy()
        r1 = 0
        r2 = 0
        for i in range(100):
            #print(len(G1.edges))
            for edge in G1.edges:
                if random.random()>0.3:
                    G1.remove_edge(edge[0],edge[1])
                if random.random()>0.8:
                    G2.remove_edge(edge[0], edge[1])
            G11,r1 = evolve2(G1,g,beta,alpha)
            G22,r2 = evolve2(G2,g,beta,alpha)

            G00 = nx.erdos_renyi_graph(N, p)

            chars = nx.get_node_attributes(G11, 'char')
            states = nx.get_node_attributes(G11, 'state')
            G1 = nx.Graph()
            G1.add_nodes_from(G11.nodes)
            for key in chars:
                G1.nodes[key]['char'] = chars[key]
                G1.nodes[key]['state'] = states[key]
                G1.nodes[key]['state2'] = 0
            G1.add_edges_from(G00.edges)

            chars = nx.get_node_attributes(G22, 'char')
            states = nx.get_node_attributes(G22, 'state')
            G2 = nx.Graph()
            G2.add_nodes_from(G22.nodes)
            for key in chars:
                G2.nodes[key]['char'] = chars[key]
                G2.nodes[key]['state'] = states[key]
                G2.nodes[key]['state2'] = 0
            G2.add_edges_from(G00.edges)

        results1.append(r1)
        results2.append(r2)
    r11 = sum(results1)/len(results1)
    r22 = sum(results2)/len(results2)
    return r11,r22

def evolution_on_synthetic2(args):
    N, p, zeta, g, beta, alpha = args
    results1 = []
    for epoch in range(500):
        print(alpha,epoch)
        G0 = nx.barabasi_albert_graph(N,p)
        G0 = initialAgent2(G0,zeta)
        G1 = G0.copy()
        G11,r1 = evolve2(G1,g*100,beta,alpha)
        results1.append(r1)
    r11 = sum(results1)/len(results1)
    return r11