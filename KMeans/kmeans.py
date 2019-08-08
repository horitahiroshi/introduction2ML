import numpy as np
import matplotlib.pyplot as plt
import ipdb

"""
    ...     x
    ..

                    ...
                    ..
                    x

'x': initial points/centroids
'.': points/examples from dataset 
"""
dataset_example = np.array([[10, 78],
                            [11, 66],
                            [9, 69],
                            [8, 77],
                            [10, 75],
                            [13, 80],
                            [21, 7],
                            [21, 9],
                            [19, 13],
                            [18, 12],
                            [19, 10],
                            [21, 8],
                            [20, 6]], dtype=float)

def euclidena_distance(example, centers):
    difs = example - centers
    return np.sqrt(np.sum(np.square(difs), axis=1))

def compute_div(prev, curr):
    difs = prev - curr
    return np.sum(np.sqrt(np.sum(np.square(difs),axis=1)))

def kmeans(dataset, clusters=2, threshold=1e-1):
    # sample with size 'clusters' containing random examples ids from dataset
    ids = np.random.choice(len(dataset),clusters, replace=False)

    # clusters centroids
    centers = dataset[ids]
    
    # array of clusters labels of each instance from dataset
    closest = np.zeros(len(dataset))

    div = 2 * threshold
    while(div > threshold):
        # for each example in dataset
        for i in range(len(dataset)):
            # compute the distance to each of the clusters centroids
            distances = euclidena_distance(dataset[i], centers)

            # find the closest centroid
            nearest_center = np.argsort(distances)[0]

            # label the example with cluster number
            closest[i] = nearest_center

        prev_centers = np.copy(centers)
        # update centroids by calculating the mean of coordinates from found cluster
        for j in range(len(centers)):
            centers[j] = np.mean(dataset[np.where(closest == j)], axis=0)
            
        # compute variation between previous centers and updated centers
        div = compute_div(prev_centers, centers)
        print(div)        
            
    plt.figure(0)
    plt.scatter(dataset[:,0],dataset[:,1])
    plt.scatter(centers[:,0],centers[:,1],marker="+")
    plt.show()

    return 0

def main():
    kmeans(dataset_example, clusters=2)
    pass

if __name__ == "__main__":
    main()