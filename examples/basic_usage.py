from mcmst_clust import MCMSTClustering
from sklearn.datasets import load_wine
from sklearn.metrics import adjusted_rand_score

data = load_wine()
X = data.data
y = data.target

model = MCMSTClustering(N=19, r=0.49, n_micro=3, random_state=42) 
labels = model.fit_predict(X)

print("n_micro:", model.n_micro_clusters_, "n_macro:", model.n_macro_clusters_)
print("ARI:", adjusted_rand_score(y, labels))