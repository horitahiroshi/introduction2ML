K-Nearest Neighbors
===================

[KNN](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) is an algorithm of classification, i.e., it is in the context of supervised machine learning.

## Brief explanation

Given a [space](https://en.wikipedia.org/wiki/Space_(mathematics)), it is possible to classify a query point based on its k nearest neighbors most frequent class, where k is a positive integer number.

For this it is necessary a concept of [distance](https://en.wikipedia.org/wiki/Distance#Theoretical_distances) (e.g. [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance)) to find the k nearest neighbors. Then the classification works like a voting, i.e., the most frequent class within the k neighbors will be the the class to be attributed to the query point.

## Example:

Imagine a person is in an event full of people and you want to find out in which class/group (s)he belongs to based on the k=10 (physicaly) nearest people. If 7 of them are from class A and 3 from class B, then the person (query point) will be from class A.