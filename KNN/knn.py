import numpy as np
import ipdb

debug = True

"""
The example dataset is about the chance of playing a sport
game according to weather's temperature and umidity.

Temperature Umidity Play?
34          78      0
32          65      0
26          89      1
22          56      1
"""
dataset_example = np.array([[34, 78, 0],
                            [32, 65, 0],
                            [26, 89, 1],
                            [22, 56, 1]], dtype=float)

def euclidian_distance(example, query):
    difs = example - query
    return np.sqrt(np.sum(np.square(difs),axis=1))
    

def knn(dataset, query, k=1):
    # calculate the distances between query and each example from dataset
    distances = euclidian_distance(dataset[:,:-1], query)
    
    # separate the k nearest examples
    k_nearest_indexes = np.argsort(distances)[:k]

    # take the classes from the last column from k nearest examples/rows from dataset
    classes = dataset[k_nearest_indexes, -1]

    # list all available nearest labels
    U = np.unique(classes)

    # count frequency of each label occured in these k nearest examples
    R = np.zeros_like(U)
    for i in range(len(U)):
        R[i] = np.sum(U[i] == classes)

    result = U[np.argsort(R)[-1]]

    return [result, U, R]
    
def main():
    query = np.array([22, 56], dtype=float)
    result,U,R = knn(dataset_example, query, k=1)

    print("The query is from class: ")
    print(int(result))

    if(debug):
        print("classes: ")
        print(U)
        print(R)

if __name__ == "__main__":
    main()