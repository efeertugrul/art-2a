import numpy as np
import math
import time


#call for the data set.
def calculate_norm(array):
    return math.sqrt(sum(array**2))

def normalize(input_array):
    norms = sum(input_array.T**2)
    for i in range(0,len(norms)):
        norms[i] = math.sqrt(norms[i])

    normalized = input_array / norms.reshape(len(norms),1)
    normalized[np.where(np.isfinite(normalized)==False)] = 0
    return normalized

#call for each normalized input value.
def get_winner_index(normalized, weights, vigilance):
    net_input = []
    for i in range(0, len(weights)):
        net_input.append(np.dot(normalized, weights[i].T))
    winner_i = net_input.index(max(net_input))
    if(net_input[winner_i] < vigilance):
        return None
    else:
        return winner_i

def weight_update(weights, last_input):
    learning_rate = 0.01
    calculation = (learning_rate*last_input+(1-learning_rate)* weights)
    weights = calculation/calculate_norm(calculation)
    return weights

def calc_rep(weights, inputs, labels, pure_data = None):
    """
    This function finds the closest data to the cluster angle.
    Returns Numpy array
    """


    normalized_inputs = normalize(inputs)
    if((pure_data!=None).all()):
        inputs_ = pure_data
        representers = np.zeros((weights.shape[0],pure_data.shape[1]))
    else:
        representers = np.zeros((weights.shape[0],weights.shape[1]))
        inputs_ = inputs
    for i in range(0, max(labels)+1):
        selected = inputs_[np.where(labels == i)]
        selected_n = normalized_inputs[np.where(labels == i)]
        result = np.dot(selected_n, weights[i])
        if(len(np.where(result == max(result))[0])==1):
            representers[i] = selected[np.where(result == max(result))]
        else:
            representers[i] = selected[np.where(result == max(result))[0][0]]

    return representers


def art2a_predict(vigi, inputs, feature_size, weights,file=None):

    """
    Predict function of ART2A
    If input can't be clustered, it is labeled as -1

    Returns 1D numpy array
    """

    normalized_inputs = normalize(inputs)
    weight_array = weights
    clusters = []
    start = time.time()
    for i in range(0,len(normalized_inputs)):
        w_i = get_winner_index(normalized_inputs[i], weight_array, vigi)
        if(w_i!=None):
            clusters.append(w_i)
        else:
            #label the input as noise
            clusters.append(-1)
    end = time.time()
    print("Art2A ended in: ", end-start)
    if(file!=None):
        csv_str = "{}.csv".format(file)
        np.savetxt(csv_str, clusters, delimiter=",")
    print("Vigilance value {}: {} clusters".format(vigi, max(clusters)+1))
    return np.array(clusters)

def art2a_train(vigi, inputs, feature_size, file=None ,cycle=1):
    """
    This wersion returns weight array also for labeling test data in predict art function
    Returns clusters and 2D weight array
    """
    normalized_inputs = normalize(inputs)
    weight_array = normalized_inputs[0].reshape(1,feature_size).copy()
    clusters = []
    cycle_start = 0
    cycle_end = 0
    start = time.time()
    for c in range(0, cycle):
        cycle_start = time.time()
        for i in range(0,len(normalized_inputs)):
            w_i = get_winner_index(normalized_inputs[i], weight_array, vigi)
            if(w_i!=None):
                weight_array[w_i] = weight_update(weight_array[w_i], inputs[i])
                if(c == cycle-1):
                    clusters.append(w_i)
            else:
                #create a new weight
                weight_array.resize(len(weight_array)+1, feature_size)
                weight_array[len(weight_array)-1] = normalized_inputs[i]
                if(c == cycle-1):
                    clusters.append(len(weight_array)-1)
        cycle_end = time.time() - cycle_start
        print("Cycle {} has ended in {} for vigilance value {}".format(c+1, cycle_end, vigi))
    end = time.time()
    print("Art2A ended in: ", end-start)
    if(file!=None):
        csv_str = "{}.csv".format(file)
        np.savetxt(csv_str, clusters, delimiter=",")
    print("Vigilance value {}: {} clusters".format(vigi, max(clusters)+1))
    return np.array(clusters), weight_array

def yield_art2a(vigi, inputs, feature_size):
    normalized_inputs = normalize(inputs)
    weight_array = normalized_inputs[0].reshape(1,feature_size).copy()
    c=0
    while(True):
        c+=1
        cycle_start = 0
        cycle_end = 0
        clusters = []
        cycle_start = time.time()
        for i in range(0,len(normalized_inputs)):
            w_i = get_winner_index(normalized_inputs[i], weight_array, vigi)
            if(w_i!=None):
                weight_array[w_i] = weight_update(weight_array[w_i], inputs[i])
                clusters.append(w_i)
            else:
                #create a new weight
                weight_array.resize(len(weight_array)+1, feature_size)
                weight_array[len(weight_array)-1] = normalized_inputs[i]
                clusters.append(len(weight_array)-1)
        cycle_end = time.time() - cycle_start
        clusters = fix_cluster_numbers(np.array(clusters))
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        print("Cycle {} has ended in {} for vigilance value {}\n Total # of clusters {}".format(c, cycle_end, vigi, max(clusters)+1))
        yield clusters

def fix_cluster_numbers(cluster_labels):
    n_c = len(np.unique(cluster_labels))
    index = 0
    t_label = 0
    minimum = min(cluster_labels)
    if(minimum!=0):
        cluster_labels = cluster_labels - minimum
        print(cluster_labels,"\n")
    for i in range(0, n_c):
        if(i not in cluster_labels):
            index = np.argmax(cluster_labels > i)
            t_label = cluster_labels[index]
            cluster_labels[cluster_labels == t_label] = i
    return cluster_labels

def calculate_centroids(input_array, cluster_array):

    if(len(input_array)!=len(cluster_array)):
        raise Exception("Non-Equal input lengths input_array={}, cluster_array={}".format(len(input_array), len(cluster_array)))

    elif(len(input_array.shape)==1):
        print(input_array.shape)
        raise Exception("Array must be 2D")

    #Get feature size
    feature_size = input_array.shape[1]

    #create an array of zeros with shape (number of clusters, number of features)
    centroids = np.zeros(((int)(max(cluster_array)-min(cluster_array)+1), feature_size))

    #the array below is created for counting the number of elements in a cluster
    #its 2-D because of doing the division at ease
    no_of_elements = np.zeros(((int)(max(cluster_array)-min(cluster_array)+1), feature_size))
    cluster_no = 0

    #for adding 1
    ones = np.ones(feature_size)

    for i in range(0,len(input_array)):
        #get the cluster of the input element
        cluster_no = (int)(cluster_array[i])

        #add the element to the right index of the centroids array
        centroids[cluster_no] += input_array[i]

        #increment the number of elements by 1
        no_of_elements[cluster_no] += ones

    #finally divide each summation to the number of elements.
    centroids = centroids / no_of_elements

    return centroids
