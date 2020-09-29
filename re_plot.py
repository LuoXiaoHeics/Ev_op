import matplotlib.pyplot as plt
import re
import seaborn as sns
import numpy as np

with open('/home/randy/Documents/socialnet/Ev_op/results/InVS13/re.txt','r') as f:
    line = f.readline()
    l = []
    while(not line==""):
        pattern = r'\d+\.\d+'
        values = re.findall(pattern,line)
        val = []
        for i in range(int((len(values)+1)/3)):
            val.append(float(values[i*3+2]))
        l.append(val)
        line = f.readline()
    print(l)
    for a in l:
        k = 7-len(a)
        for m in (range(k)):
            a.append(0.4)
    print(l)

data = np.array(l)



#
# np.random.seed(0)
# x = np.random.randn(4, 4)
#
f, ax1= plt.subplots(figsize=(6, 6))
#
# sns.heatmap(x, annot=True, ax=ax1)
#
sns.heatmap(data,ax = ax1,square=True,cmap='YlGnBu')

# ax2.set_title('matplotlib colormap')
# ax2.set_xlabel('region')
# ax2.set_ylabel('kind')
#
plt.show()