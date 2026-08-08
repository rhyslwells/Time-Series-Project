from tslearn.metrics import dtw
from tslearn.clustering import TimeSeriesKMeans

# compute pairwise DTW distance
dist = dtw(series1, series2)

# clustering example
model = TimeSeriesKMeans(n_clusters=3, metric="dtw", random_state=0)
labels = model.fit_predict(time_series_dataset)


-------------------------


Example: of computing distacnes between time series

#compare two

T = 100 # Length of the time series
N = 30  # Time series per set

# Generate the first set of time series
Y1 = np.zeros((T, N))
for i in range(N):
    Y1[:,i] = sm.tsa.arma_generate_sample(ar=[1, -.9], ma=[1], nsample=T, scale=1) 

# Generate the second set of time series
Y2 = np.zeros((T, N))
for i in range(N):
    Y2[:,i] = sm.tsa.arma_generate_sample(ar=[1, .9], ma=[1], nsample=T, scale=1)

# - Let's see how the mean and std of the two groups look like.


fig, axes = plt.subplots(2,1, figsize=(10, 5))
axes[0].plot(np.mean(Y1, axis=1))
axes[0].fill_between(range(T), np.mean(Y1, axis=1) - np.std(Y1, axis=1), np.mean(Y1, axis=1) + np.std(Y1, axis=1), alpha=0.3)
axes[0].set_title("First set")
axes[1].plot(np.mean(Y2, axis=1))
axes[1].fill_between(range(T), np.mean(Y2, axis=1) - np.std(Y2, axis=1), np.mean(Y2, axis=1) + np.std(Y2, axis=1), alpha=0.3)
axes[1].set_title("Second set");



# - We can visualize the path $\pi$ on the cost matrix $\mathbf{D}_{\boldsymbol{x}, \boldsymbol{y}}$ for the time series we generated.
# - First, we let $\boldsymbol{x}$ and $\boldsymbol{y}$ be two time series from the same group.
# - Notice how the path $\pi$ crosses the darker areas, corresponding to smaller dissimilarity values.

s1 = Y2[:,1]
s2 = Y2[:,2]
fig = plt.figure(figsize=(5, 5))
d, paths = dtw.warping_paths(s1, s2)
best_path = dtw.best_path(paths)
dtwvis.plot_warpingpaths(s1, s2, paths, best_path, figure=fig);

# Concatenate the two sets of time series
Y = np.concatenate((Y1, Y2), axis=1).T

# Compute the distance matrix
dtw_dist = dtw.distance_matrix_fast(Y)

plt.figure(figsize=(4,4))
plt.imshow(dtw_dist, cmap='viridis')
plt.colorbar();


-----

# classification

from tck.datasets import DataLoader
