from utils import *
import tqdm
import time
from multiprocessing import Pool
from matplotlib import pyplot as plt

# data_file = "tij_pres_InVS13.dat"
# meta_file = "metadata_InVS13.dat"
# G,node_label,edges = read_data(meta_file,data_file)
# edges_w = edge_weights(edges,1)
# G = construct_graph(G,edges_w)
# deal_unconnected(G)
#

for lam in tqdm.tqdm(range(0,11,2),desc="a:"):
    for b in tqdm.tqdm(range(0,21,2),desc="b:"):
        args = []
        time = 0
        while time<100:
            G = nx.nx.erdos_renyi_graph(200,0.05)
            deal_unconnected(G)
            args.append([G, 0.2, 0.3, 0.4, 0.3, 0.05, lam*0.1, 10, 0.5, 100,10000,b*0.01])

            # graph2, p, alpha, beta, gamma, tau, lmd, b, c, round_num,t0
            time += 1
        pool = Pool(processes=10)
        result = pool.map(evole, args)
        private_op = [r0[0] for r0 in result]
        public_op = [r0[1] for r0 in result]
        fitness = [r0[2] for r0 in result]
        with open('results_weak_er_de.txt', 'a') as f:
            f.write(str(sum(fitness)/len(fitness))+",")
            f.write(str(sum(private_op) / len(private_op)) + ",")
            f.write(str(sum(public_op) / len(public_op)) + ",")
    with open('results_weak_er_de.txt', 'a') as f:
        f.write("\n")
# graph2,p,alpha,beta,gamma,tau,lmd,b,c,round_num = args
#     graph = graph2.copy()

