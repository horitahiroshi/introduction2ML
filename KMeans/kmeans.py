import numpy as np
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

def kmeans(dataset, clusters=2):
    # sample with size 'clusters' containing random examples ids from dataset
    ids = np.random.choice(len(dataset),clusters, replace=False)

    # clusters centroids
    centers = dataset[ids]

    return

def main():
    pass

if __name__ == "__main__":
    main()