# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 10:41:16 2023

@author: Ali ŞENOL
"""

import numpy as np
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import silhouette_score
import scipy
from IPython import get_ipython
import warnings
warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')
get_ipython().magic('clear all -sf')
xt=np.array([])
from sklearn import metrics

class MCMSTCluster:
    N=int()

    r=float()    
    d=int()
    MC_Num=int
    MacroC_Num=int
    colors=np.empty((0,4),int)
    def __init__(self,X,N,r,n_micro,d):
        #algorithm parameters###########################################
        self.XX=X
        self.N = N #minimum number of data to define a MC 
        self.r = r # radius of MC
        self.n_micro=n_micro # minimum number MC to define Macro Cluster
        ##################################################################
        self.d=d
        self.MC_Num=0
        self.MacroC_Num=0
        self.processed_data=np.empty((0,d+3),float) #[index | MC No | isActive | features={d1,d2,d3...}]
        self.MCs=np.empty((0,d+3),float) #[MC No | #of data it has | Macro Cluster # |centerCoordinates={d1,d2,d3,...}]
        self.MacroClusters=np.empty((0,4)) #[MacroClusterNo | #of data it has | isActive ]
        self.labels_=[] #Cluster labels_
        self.AddNode()
        self.DefineMC()
        self.RegulateClusters()
        self.DefMacros()
        self.assingMacoN()
    def AddNode(self): # add the data to processed_data
        self.processed_data=np.hstack((np.arange(self.XX.shape[0]).reshape(self.XX.shape[0],1)+1,np.repeat([[0,0]], self.XX.shape[0],0),self.XX))
            
    def DefineMC(self):       
        while True:
            X=self.processed_data[self.processed_data[:,1]==0,:]#data that do not belong to any cluster
            num_MCs=self.MC_Num;
            if(X.shape[0]>=self.N): # if # of data that do not belong to any cluster greater than N
                tree=KDTree(X[:,3:]) #construct kdtree
                for i in range(X.shape[0]): # for each data of tree do reangeserach
                    if(X[i,2]==0):
                        ind=tree.query_radius(X[i:,3:], self.r) #rangesearch
                        points=X[ind[0],3:]
                        # print(points)
                        if(len(points)>=self.N):  
                            center=np.mean(np.array(points),axis=0) #calculate the center of candidate MC
                            # if(self.MCs[:,0].shape[0]==0):
                            self.MC_Num=self.MC_Num+1
                            # print("MC number ",self.MC_Num," is defined")
                            self.MCs=np.vstack((self.MCs,np.hstack((np.array([self.MC_Num,len(points),0]),center)))) # define new MC
                            for j in range(len(points)):                  
                                self.processed_data[np.where((self.processed_data[:,3:] == points[j]).all(axis=1))[0],1]= self.MC_Num #assign data to new MC     
                            break
            if(num_MCs==self.MC_Num):
                return   
    def MST(self,mc_id):
        P=self.MCs[self.MCs[:,2]==0,:]
        indices=self.MCs[self.MCs[:,2]==0,0]
        # print(indices)
        G=squareform(pdist(P[:,3:]))
        # number of vertices in graph
        N = G.shape[0]
        selected_node = np.zeros(N)
        no_edge = 0
        # Starting node
        selected_node[P[:,0]==mc_id] = True
        spanning_edges = []
        while (no_edge < N - 1):
            minimum = 2*self.r###########################################Buraya bak
            a = 0
            b = 0
            for m in range(N):
                if selected_node[m]:
                    for n in range(N):
                        if ((not selected_node[n]) and G[m][n]):  
                            # not in selected and there is an edge
                            if G[m][n] < minimum:
                                a = m
                                b = n
            if(int(indices[a])==int(indices[b])):
                       break
            spanning_edges.append([int(indices[a]),int(indices[b])])
            selected_node[b] = True
            no_edge += 1
        return spanning_edges
    
    def DefMacros(self):
        edge_lists=[]
        for a in range(self.MCs.shape[0]):
            if(self.MCs[a,2]==0):
                edge_lists=self.MST(self.MCs[a,0])
                if(len(edge_lists)>0):
                    edge_list=np.empty((0,2),int)
                    for index in edge_lists:
                        i,j=index
                        edge_list=np.vstack((edge_list,(i,j)))
                    summ=0
                    edges=np.unique(edge_list)
                    for e in edges:
                        summ=summ+self.MCs[self.MCs[:,0]==e,1]
                    if(summ>=n_micro*N or len(np.unique(edge_list))>=n_micro):
                        self.MacroC_Num=self.MacroC_Num+1
                        colors = np.array(sns.color_palette(None, self.MacroC_Num+1))
                        self.MacroClusters=np.vstack((self.MacroClusters,  np.array([self.MacroC_Num,len(np.unique(edge_list)),edge_list,colors[-1,:]],dtype=object)   ))
                        print("--Macro Cluster #",self.MacroC_Num," is defined--")
                        for i in np.unique(edge_list):
                            self.MCs[self.MCs[:,0]==i,2]=self.MacroC_Num
        if len(edge_lists)>=self.N:
            for a in range(self.MCs.shape[0]):
                if(self.MCs[a,2]==0 and self.MCs[a,1]>=self.N*self.n_micro):
                        self.MacroC_Num=self.MacroC_Num+1
                        colors = np.array(sns.color_palette(None, self.MacroC_Num+1))
                        self.MacroClusters=np.vstack((self.MacroClusters,np.array([self.MacroC_Num,len(np.unique(edge_list)),edge_list,colors[-1,:]],dtype=object)))
                        print("--Macro Cluster #",self.MacroC_Num," is defined--")
                        self.MCs[a,2]=self.MacroC_Num
                
    def SearchforClusters(self):
        for i in range(self.MCs.shape[0]):
           if (self.MCs[i,2]==0): 
                self.DefineMacroC(self.MCs[i,0])
    def assingMacoN(self):
        for i in self.MCs:
            self.processed_data[self.processed_data[:,1]==i[0],2]=i[2]
        self.labels_=self.processed_data[:,2].reshape(-1)
    def RegulateClusters(self):
        X=self.processed_data[self.processed_data[:,1]==0,:]#data that do not belong to any cluster
        if(X.shape[0]>0 and self.MCs.shape[0]>0):
            tree=KDTree(self.MCs[:,3:]) #construct kdtree
            for i in X:
                distance,index=tree.query([i[3:]])
                if(distance[0]<=2*self.r):
                    self.processed_data[self.processed_data[:,0]==i[0],1]=self.MCs[self.MCs[:,0]==index[0]+1,0]
    def plotGraph(self,X,labels,r,N,n_micro,index,index_value,dataset_name,dpi=100):
        plt.rcParams['figure.dpi'] = dpi
        plt.rcParams["figure.figsize"] = (4,4) 
        plt.scatter([X[labels!=0, 0]], [X[labels!=0, 1]],s=40, c=labels[labels!=0],edgecolors='k',cmap="jet") #nipy_spectral
        plt.scatter([X[labels==0, 0]], [X[labels==0, 1]],s=10, c="black",edgecolors='k',cmap="jet") #nipy_spectral
        eps=str("r=%0.2f,n_micro=%d,N=%d){%s=%0.4f}"%(r,n_micro,N,index,index_value)) 
        s="MCMSTClustering (eps="+eps
        plt.title(s,fontsize = 8)
        plt.rcParams.update({'font.size': 8})
        plt.grid()
        plt.rcParams['axes.axisbelow'] = True
        plt.xlabel('x values')
        plt.ylabel('y values')
        plt_name=str("img/"+dataset_name+"_"+index+".png")
        plt.savefig(plt_name,bbox_inches='tight') 
        plt.show()        
    def plotMCs(self,N,r,n_micro,maxARI,maxPurity,maxSI,dpi=100):
        ax = plt.gca()
        plt.rcParams['figure.dpi'] = dpi
        plt.rcParams["figure.figsize"] = (4,4) 
        ax.cla() # clear things for fresh plot 
        plt.rcParams['figure.dpi'] = dpi
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)
        for i in range(len(self.processed_data)):
            mc=self.processed_data[i,1]
            Macro=self.MCs[self.MCs[:,0]==mc,2]
            
            if(int(mc)==0 or Macro==0):
                colr=(0,0,0,1) 
            else:  
                # print(Macro)
                colr=self.MacroClusters[self.MacroClusters[:,0]==Macro,3][0]
                # print(colr)
            plt.plot(self.processed_data[i, 3], self.processed_data[i, 4],color=colr, marker='o',markeredgecolor='k',alpha=.25, markersize=2)
        for i in range(len(self.MCs)):
            if(self.MCs[i, 2]==0):
                col=(0,0,0,1) # black MC means that it does not belong to any MacroC
            else:
                col=self.MacroClusters[self.MacroClusters[:,0]==self.MCs[i,2],3][0].tolist()
            plt.plot(self.MCs[i, 3], self.MCs[i, 4],'rd',markeredgecolor='k', markersize=4)
            circle1=plt.Circle((self.MCs[i,3],self.MCs[i,4]),self.r,color=col, clip_on=False,fill=False,linewidth=2)
            plt.text(self.MCs[i,3],self.MCs[i,4], str(int(self.MCs[i,0])))
            ax.add_patch(circle1)      
        for edge in self.MacroClusters[:,2]:
            for e in edge:
                i, j = e
                plt.plot([self.MCs[self.MCs[:,0]==i, 3], self.MCs[self.MCs[:,0]==j, 3]], [self.MCs[self.MCs[:,0]==i, 4], self.MCs[self.MCs[:,0]==j, 4]], c=self.MacroClusters[self.MacroClusters[:,0]==self.MCs[self.MCs[:,0]==i,2],3][0].tolist(),markersize=500)     
        plt.title("MCMSTClustering Micro-Clusters") 
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")        
        plt.xlim([-.05,1.05]) 
        plt.ylim([-.05,1.05])            
        # plt.show()
        plt.subplots_adjust(bottom=0.25) 
        text = plt.text(0.45, -0.25, 
                        str("{N=%d,r=%.2f,NM=%d}=>{ARI=%.4f,Pur=%.4f,SI=%.4f}"%(optN,optR,optNMicro,maxARI,maxPurity,maxSI)), 
        horizontalalignment='center', wrap=True ) 
        plt.tight_layout(rect=(0,.05,1,1)) 
        text.set_bbox(dict(facecolor='red', alpha=0.5, edgecolor='black'))
        plt_name=str("img/"+dataset_name+".png")
        plt.savefig(plt_name)
        plt.show()
        plt.clf()
    def purity_score(self,y_true, y_pred):
        # compute contingency matrix (also called confusion matrix)
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        # return purity
        return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
##------------main---------------------------------------------------------------
data_sets={1,2,3,4} ##tested datasets
for dataset in data_sets:
    print("dataset=",dataset)
    plotFigure=1
    loop=0
    if (dataset==1):
        data = scipy.io.loadmat("Datasets/halfkernel.mat")
        data=data["halkernel"];
        labels_true = data[:,2]
        X = data[:,0:2]
        dataset_name="1_HalfKernel_"
        N=4 #minimum number of data to define a MC 
        r=0.084 # radius of MC
        n_micro=8 # minimum number MC to define Macro Cluster
    elif dataset==2:
        data = np.genfromtxt("Datasets/Three-Spirals.txt", delimiter=',', dtype=float,usecols=range(3))
        labels_true = data[:,2]
        X = data[:,0:2]
        dataset_name="2_Three_Spirals_"
        N=3 #minimum number of data to define a MC 
        r=0.066 # radius of MC
        n_micro=6 # minimum number MC to define Macro Cluster
    elif dataset==3:
        data = np.genfromtxt("Datasets/corners.txt", delimiter=',', dtype=float,usecols=range(3))
        labels_true = data[:,2]
        X = data[:,0:2]
        dataset_name="3_Corners_"
        N=15 #minimum number of data to define a MC 
        r=0.066 # radius of MC
        n_micro=13 # minimum number MC to define Macro Cluster
    elif dataset==4:
        data = scipy.io.loadmat("Datasets/moon.mat")
        X=data["data"]
        labels_true = np.mat(data["target"])
        labels_true=np.ravel(labels_true)
        dataset_name="4_Moon_"
        N=3 #minimum number of data to define a MC 
        r=0.054# radius of MC
        n_micro=14 # minimum number MC to define Macro Cluster---->1.0000
   
    ####MinMaxNormalization#######################################################
    scaler = MinMaxScaler()
    scaler.fit(X)
    MinMaxScaler()
    X=scaler.transform(X[:,:])

    loop=0; 
    maxARI=float('-inf')
    maxPurity=float('-inf')
    maxSI=float('-inf')
    optN=float('-inf')
    optR=float('-inf')
    optNMicro=float('-inf')
    maxEpsilon=float('-inf') 
    SI=float('-inf') 
   
    kds=MCMSTCluster(X,N,r,n_micro,X.shape[1])    
    labels=kds.labels_
    
    if len(np.unique(labels))>1:
        ARI=adjusted_rand_score(labels_true.reshape(-1), labels)
        if ARI>maxARI:
            maxARI=ARI
            maxSI=silhouette_score(X,labels)
            maxPurity=kds.purity_score(labels_true.reshape(-1), labels)
            optN=N
            optR=r
            optNMicro=n_micro
            print("=========Better parameters are detected=========")
            print("Max ARI=%0.4f"%maxARI)
            print("Purity=%0.4f"%maxPurity)
            print("Silhouette Index=%0.4f"%maxSI)  
            print("The best N=%d"%N)
            print("The best r=%0.4f"%r)
            print("the best n_micro=%d"%n_micro)
            kds.plotMCs(N,r,n_micro,maxARI,maxPurity,maxSI)
            kds.plotGraph(X,labels,r,N,n_micro,"ARI",ARI,dataset_name)
