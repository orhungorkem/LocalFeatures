import numpy as np

def ssd(desc1, desc2):
    '''
    Sum of squared differences
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - distances:    - (q1, q2) numpy array storing the squared distance
    '''
    assert desc1.shape[1] == desc2.shape[1]
    # TODO: implement this function please

    M = desc1.shape[0]
    N = desc2.shape[0]  

    # d1.d1 + d2.d2 - 2.d1.d2 should be calculated
    
    d1_d1 = (desc1*desc1).sum(axis=1).reshape((M,1))*np.ones(shape=(1,N))
    d2_d2 = (desc2*desc2).sum(axis=1)*np.ones(shape=(M,1))
    return d1_d1 + d2_d2 -2*desc1.dot(desc2.T)
    

def match_descriptors(desc1, desc2, method = "one_way", ratio_thresh=0.5):
    '''
    Match descriptors
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - matches:      - (m x 2) numpy array storing the indices of the matches
    '''
    assert desc1.shape[1] == desc2.shape[1]
    distances = ssd(desc1, desc2)
    q1 = desc1.shape[0]
    if method == "one_way": # Query the nearest neighbor for each keypoint in image 1
        # TODO: implement the one-way nearest neighbor matching here
        matches = []
        for i in range(q1):
            col = np.argmin(distances[i]) #column with minimum distance
            matches += [[i,col]]
        
        return np.array(matches)



    elif method == "mutual":
        # TODO: implement the mutual nearest neighbor matching here
        matches = []
        for i in range(q1):
            col = np.argmin(distances[i])
            if np.argmin(distances[:,col]) == i:  # also checking the minimum argument in the column
                matches += [[i,col]]
        
        return np.array(matches)
        
    elif method == "ratio":
        # TODO: implement the ratio test matching here
        matches = []
        for i in range(q1):
            col = np.argmin(distances[i])
            second = np.partition(distances[i],1)[1]  # gives the second minimum distance
            if distances[i,col] < ratio_thresh * second:
                matches += [[i,col]]
        
        return np.array(matches)
    else:
        raise NotImplementedError
    

