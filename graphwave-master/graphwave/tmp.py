import networkx as nx
import copy
a=[]
b=[]
def createGraph(filename):
    G = nx.DiGraph()
    weightlist=Get_Mdimensions_NlengthArray_initValuex(2,[295,295],0)
    

   

    for line in open(filename):
        strlist0=line.replace(',',' ')
        strlist =strlist0.split()
        n1 = int(strlist[0])
        n2 = int(strlist[1])      
        weightlist[n1][n2] +=1      
        if n1 not in a:
            a.append(n1)
        if n2 not in a:
            a.append(n2)
    b=sorted(a)

    
    for i in range (0,295):
        for j in range(0,295):
            if weightlist[i][j] != 0:
                G.add_weighted_edges_from([(i,j,weightlist[i][j])])
              
    return G,b,weightlist

def Get_Mdimensions_NlengthArray_initValuex(m,n,x):
    if m!=len(n):
        print("Error")
    else:
        result=[x for i in range(n[-1])]
        dimensions_num=1
        while dimensions_num<m:
            result=[copy.deepcopy(result) for i in range(n[-1-dimensions_num])]
            dimensions_num+=1
        return result
