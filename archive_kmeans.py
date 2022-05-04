def dist(ts_1, ts_2, sigma):
    #mask = (np.isnan(ts_1) | np.isnan(ts_2)) == False
    #length = mask.sum()
    #if length == 0:
    #    print('wtf')
    #    return 1
    #else:
    #return np.sqrt(np.max((ts_1 - ts_2) ** 2))
    return np.dot(ts_1 - ts_2, np.linalg.pinv(sigma) @ (ts_1 - ts_2))


def initialize(data, k):
    centroids = []
    centroids.append(data[np.random.randint(
            data.shape[0]), :])
    for c_id in range(k - 1):
        distances = []
        for i in range(data.shape[0]):
            point = data[i, :]
            d = sys.maxsize
            for j in range(len(centroids)):
                temp_dist = dist(point, centroids[j], np.eye(data.shape[1]))
                d = min(d, temp_dist)
            distances.append(d)
             
        distances = np.array(distances)
        next_centroid = data[np.argmax(distances), :]
        centroids.append(next_centroid)
        distances = []

    return centroids, [np.eye(data.shape[1]) for i in range(k)]

def kmeans(X, data, k):
    diff = 1
    cluster = np.zeros(X.shape[0])
    centroids, sigmas = initialize(X, k)
    while diff:
        # for each observation
        for i, row in enumerate(X):
            mn_dist = float('inf')
            # dist of the point from all centroids
            for idx, centroid in enumerate(centroids):
                sigma = sigmas[idx]
                d = dist(centroid, row, sigma)
                # store closest centroid
                if mn_dist > d:
                    mn_dist = d
                    cluster[i] = idx
        new_centroids = pd.DataFrame(X).groupby(by=cluster).mean().values
        new_sigmas = pd.DataFrame(X).groupby(by=cluster).var().values # Sample variance
        print(new_sigmas)
        # if centroids are same then leave
        if np.count_nonzero(centroids-new_centroids) == 0:
            diff = 0
        else:
            centroids = new_centroids
    return centroids, cluster

def get_clusters(data):
    #data = data.fillna(0)
    X = data.values
    gotcha = True
    """while gotcha:
        try:
            centroids, clusters = kmeans(X, data, 12)
            gotcha = False
        except Exception as e:
            print("Retry")
            pass
    """
    centroids, clusters = kmeans(X, data, 5)
    return clusters
