from sqlite3 import SQLITE_CREATE_INDEX
import numpy as np
import scipy
import cv2

# Harris corner detector
def extract_harris(img, sigma = 1.0, k = 0.05, thresh = 1e-5):
    '''
    Inputs:
    - img:      (h, w) gray-scaled image
    - sigma:    smoothing Gaussian sigma. suggested values: 0.5, 1.0, 2.0
    - k:        Harris response function constant. suggest interval: (0.04 - 0.06)
    - thresh:   scalar value to threshold corner strength. suggested interval: (1e-6 - 1e-4)
    Returns:
    - corners:  (q, 2) numpy array storing the keypoint positions [x, y]
    - C:     (h, w) numpy array storing the corner strength
    '''
    # Convert to float
    img = img.astype(float) / 255.0

    # Compute image gradients
    # TODO: implement the computation of the image gradients Ix and Iy here.
    # You may refer to scipy.signal.convolve2d for the convolution.
    # Do not forget to use the mode "same" to keep the image size unchanged.
    kernel_x = np.array([[0,0,0],[-0.5,0,0.5],[0,0,0]])   # kernels to get image gradients
    kernel_y = np.array([[0,-0.5,0],[0,0,0],[0,0.5,0]])

    I_x = scipy.signal.convolve2d(img, kernel_x, mode='same', boundary='symm') 
    I_y = scipy.signal.convolve2d(img, kernel_y, mode='same', boundary='symm')
    

    
    M = np.zeros((len(img), len(img[0]), 2, 2))

    # Compute local auto-correlation matrix
    # TODO: compute the auto-correlation matrix here
    # You may refer to cv2.GaussianBlur for the gaussian filtering (border_type=cv2.BORDER_REPLICATE)

    kernel_size = 3

    for i in range(len(img)):
        for j in range(len(img[0])):
            cov = np.array([[I_x[i][j] ** 2, I_x[i][j] * I_y[i][j]],[I_y[i][j] * I_x[i][j], I_y[i][j] ** 2]])
            M[i][j] = cov
    
    

    sliced_m = np.zeros((2,2,len(img[0]),len(img)))
    for i in range(2):
        for j in range(2):
            sliced = np.array([[M[t][z][i][j] for t in range(len(img))] for z in range(len(img[0]))])  # getting slices since blur function gets 2d matrix
            sliced = cv2.GaussianBlur(sliced, (kernel_size,kernel_size), sigma, sigma, cv2.BORDER_REPLICATE)
            sliced_m[i][j] = sliced

    for i in range(len(img)):
        for j in range(len(img)):
            for z in range(2):
                for t in range(2):
                    M[i][j][z][t] = sliced_m[z][t][i][j]
    
    
    
    

    # Compute Harris response function
    # TODO: compute the Harris response function C here
    C = np.array([[np.linalg.det(M[i][j]) - k * np.trace(M[i][j])**2 for j in range(len(img[0]))] for i in range(len(img))])
    
    

    # Detection with threshold
    # TODO: detection and find the corners here
    # For the local maximum check, you may refer to scipy.ndimage.maximum_filter to check a 3x3 neighborhood.
    neighborhood = np.ones((3, 3), dtype = bool)
    filtered = (scipy.ndimage.maximum_filter(C, footprint=neighborhood) == C)* 1
    winners = set()
    for i in range(len(C)):
        for j in range(len(C[0])):
            if C[i][j]>thresh and filtered[i][j] > 0:
                winners.add((i,j))

    corners = np.array([[i[0], i[1]] for i in winners])

    return corners, C

