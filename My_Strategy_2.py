import numpy as np
import scipy.io
import math
import matplotlib.pyplot as plt

#Initializing randomly the First Centroid
def centroid_init(k, data_sets):
    index_list = []
    centroids = []
    temp = np.zeros([len(data_sets), k - 1])
    randomly = np.random.choice(data_sets.shape[0], 1, replace=False)
    index_list.append(randomly[0])
    centroids.append(data_sets[randomly])
    # Selecting Centroids which are at maximum distance from each other
    for i in range(k - 1):
        temp[:, i] = np.linalg.norm(centroids[i] - data_sets, axis=1)
        condition = True
        temp2 = np.mean(temp[:, :i + 1], axis=1)
        condition = True
        i = np.argmax(temp2)
        while (condition):
            if i in index_list:
                temp2[i] = 0
                i = np.argmax(temp2)
            else:
                condition = False
        centroids.append(np.asarray(data_sets[i]))
        index_list.append(i)
    centroidss = data_sets[index_list]
    return centroidss

# Importing the data from the "AllSamples" file for the learning algorithm
def dataext():
    dataset = scipy.io.loadmat("C:\\Users\\kunal\\Desktop\\Statistical Machine Learning\\Project 2\\AllSamples.mat")
    data = dataset['AllSamples']
    return data

# Objective Function for the K-means Algorithm
def objFun(data_sets, centroids):
    objectiveval = []
    for r in data_sets:
        objectiveval.append(((np.linalg.norm((r - centroids), axis=0) ** 2)))
    return np.sum(objectiveval)

# Calculating the Euclidean Distance from the given Coordinates
def dist_euc(x_cord, y_cord, x_cent, y_cent):
    x_new = (x_cent - x_cord) ** 2
    y_new = (y_cent - y_cord) ** 2
    disteuc = math.sqrt(x_new + y_new)
    return disteuc

# Extracting the x and y coordinates from the given dataset
dat=dataext()
print(dat)
x_cordinate = dat.take(0, axis=1)
y_cordinate = dat.take(1, axis=1)

# Initializing Color Map
color_map = ['orange','red', 'green', 'blue', 'yellow', 'grey', 'purple', 'maroon', 'yellowgreen', 'skyblue', 'wheat',
             'pink', 'cyan']

# initializing the k values from k = 2-10 and objective function
obj_plot = []
k_value = [2, 3, 4, 5, 6, 7, 8, 9, 10]

for k in k_value:
    centroids = centroid_init(k, dat)
    y_centroid = centroids.take(1, axis=1).tolist()
    x_centroid = centroids.take(0, axis=1).tolist()
    # plot scatter graph for given data set with initial k
    figure = plt.figure(figsize=(5, 5))
    print(" Graph for k = ", k)
    plt.scatter(x_centroid, y_centroid, marker='+', s=100, color='black')
    plt.scatter(x_cordinate, y_cordinate,color='orange')
    condition = True
    plt.ylim(-2, 10)
    plt.xlim(-2, 10)
    plt.show()
    temp1 = np.zeros(len(dat))
    condition = True
    all = 0

    while (condition):
        Distt = []
        # For every data point in the given data
        for point in dat:
            EDistt = []
            for cc in centroids:
                EDistt.append(dist_euc(point[0], point[1], cc[0], cc[1]))
            Distt.append(EDistt)
        Distt = np.asarray(Distt)
        y_centroid = centroids.take(1, axis=1).tolist()
        x_centroid = centroids.take(0, axis=1).tolist()
        # Updating the centroid
        centroid_old = np.copy(centroids)
        # plot scatter graph
        temp1 = np.argmin(Distt, axis=1)
        figures = plt.figure(figsize=(5, 5))

        for i in range(k):
            tempp = np.asarray([dat[j] for j in range(len(dat)) if temp1[j] == i])
            centroids[i] = np.mean(tempp, axis=0)
            # printing map
            y_c = tempp.take(1, axis=1).tolist()
            x_c = tempp.take(0, axis=1).tolist()
            plt.scatter(x_c, y_c, c=color_map[i])
        condition_temp = (np.array_equal(centroid_old, centroids))
        if condition_temp:
            condition = False
        else:
            condition = True
        plt.xlim(-2, 10)
        plt.ylim(-2, 10)
        plt.scatter(x_centroid, y_centroid, marker='+', s=100, color="black")
        plt.show()

    # Calculating Objective Function and its Graph
    for i in range(k):
        tempp = np.asarray([dat[j] for j in range(len(dat)) if temp1[j] == i])
        val = objFun(tempp, centroids[i])
        all = all + val
    obj_plot.append(all)
    print(obj_plot)
    print("Clusters is ", all)

# Plot for Objective Function
plt.plot(k_value, obj_plot, 'bx-')
plt.title('Plot of Objective Function ( Strategy 2 )')
plt.ylabel('Objective Function')
plt.xlabel('K Value')
plt.show()