import numpy as np
import ipdb

debug = True

dataset_example = np.zeros((10,2))
for i in range(len(dataset_example)):
    dataset_example[i] = [i+1, i+1]

# function to calculate euclidean distances between query and all examples from dataset
def euclidian_distance(example, query):
    difs = example - query
    
    try:     # in case there are 2 or more attributes per example
        euclidian = np.sqrt(np.sum(np.square(difs),axis=1))
    except:  # in case the there is only one attribute per example
        euclidian = np.sqrt(np.square(difs))

    return euclidian

# function to calculate the weights on each example from dataset, according to their distances
# from query point.
def weighting(example, query, sigma):
    euclidian = euclidian_distance(example[:,-1], query)
    return (np.exp(-np.square(euclidian) / (2 * np.square(sigma))))

# calculate the class estimation using weighted average with weights based on the distance
# between examples and query point.
def dwknn(dataset, query, sigma=0.5):
    w = weighting(dataset, query, sigma)
    print("dataset: ", dataset)
    print("weights: ", w)

    Y = dataset[:,-1]
    y = np.sum(w * Y) / np.sum(w)
    print("classification = ", y)
    
    return

def main():
    dwknn(dataset_example, 5.3, 0.5)

if __name__=="__main__":
    main()