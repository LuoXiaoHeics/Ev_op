from utils import *
import tqdm
import time
from multiprocessing import Pool
from matplotlib import pyplot as plt

data_file = "tij_pres_InVS13.dat"
meta_file = "metadata_InVS13.dat"
G,node_label,edges = read_data(meta_file,data_file)
edges_w = edge_weights(edges,1)
G = construct_graph(G,edges_w)
deal_unconnected(G)

args = []
for r in range(21):
    for i in range(100):
        args.append([G,0.1,0.3,0.4,0.3,5,r*0.05,0.2,10000])
    pool = Pool(processes=11)
    result = pool.map(evole, args)
    re = []
    for j in range(3):
        re.append(sum([x[j] for x in result])/len(result))
    with open('results.txt','w') as f:
        f.write(str(re))
