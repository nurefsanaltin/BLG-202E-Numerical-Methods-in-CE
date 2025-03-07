import numpy as np
from numpy.linalg import norm
from random import normalvariate
import math
import itertools
import sys


def find_vector(A, epsilon=1e-10):

    n, m = A.shape
    min_n_m = min(n,m)
    unnormalized = [normalvariate(0, 1) for _ in range(min_n_m)]
    theNorm = np.sqrt(sum(x * x for x in unnormalized))
    x = [x / theNorm for x in unnormalized]
    lastV = None
    currentV = x

    if n > m:
        B = np.dot(A.T, A)
    else:
        B = np.dot(A, A.T)

    iterations = 0
    while True:
        iterations += 1
        lastV = currentV
        currentV = np.dot(B, lastV)
        currentV = currentV / norm(currentV)

        if abs(np.dot(currentV, lastV)) > 1 - epsilon:
            return currentV


def svd(A, k=None, epsilon=1e-10):

    A = np.array(A, dtype=float)
    n, m = A.shape
    svdSoFar = []
    if k is None:
        k = min(n, m)

    for i in range(k):
        matrixFor1D = A.copy()

        for singularValue, u, v in svdSoFar[:i]:
            matrixFor1D -= singularValue * np.outer(u, v)

        if n > m:
            v = find_vector(matrixFor1D, epsilon=epsilon)  
            u_unnormalized = np.dot(A, v)
            sigma = norm(u_unnormalized)  
            u = u_unnormalized / sigma
        else:
            u = find_vector(matrixFor1D, epsilon=epsilon)  
            v_unnormalized = np.dot(A.T, u)
            sigma = norm(v_unnormalized)  
            v = v_unnormalized / sigma

        svdSoFar.append((sigma, u, v))

    singularValues, us, vs = [np.array(x) for x in zip(*svdSoFar)]
    return us.T,singularValues, vs



def kabsch_umeyama(matrix_Q,matrix_P):
    
    """"
        Inputs:
            matrix_P: n x m matrix
            matrix_Q: n x m matrix
        Outputs:
            R: Rotation matrix
            t: Translation vector
    """
    
    centroid_P = np.mean(matrix_P, axis=0)
    centroid_Q = np.mean(matrix_Q, axis=0)
    
    mean_centered_P = matrix_P - centroid_P
    mean_centered_Q = matrix_Q - centroid_Q
    
    P = mean_centered_P.T
    Q = mean_centered_Q.T
    
    M = Q @ (P.T)
    
    V, S, W = svd(M)
    
    S = np.eye(V.shape[0])   
    
    if np.linalg.det(V@W)<0 :
        S [-1,-1] = -1
    
    R = (W.T) @ S @ (V.T)
    t = centroid_P - R @ centroid_Q
    

    return R, t


def main():
    if len(sys.argv) != 4:
        print("Usage: python main.py mat1.txt mat2.txt correspondences.txt")
        sys.exit(1)
        
        
    mat1 = sys.argv[1]
    mat2 = sys.argv[2]
    correspondence_path= sys.argv[3]

    matrix1= np.loadtxt(mat1)
    matrix2= np.loadtxt(mat2)

    correspondences = np.loadtxt(correspondence_path)

    new_matrix1 = matrix1[correspondences.astype(int)[:,0]]
    new_matrix2 = matrix2[correspondences.astype(int)[:,1]]
    
    
    R, t = kabsch_umeyama(new_matrix1,new_matrix2)
    
    np.savetxt("rotation_mat.txt", R)
    np.savetxt("translation_vec.txt", t)

      
    matrix_t = np.tile(t[:,np.newaxis],(1,matrix2.shape[0]))
    rotated_Q = (R.T @ (matrix2.T - matrix_t)).T
    
    # merge and remove duplicates
    indices_duplicates = sorted(correspondences.astype(int)[:,1], reverse=False)
    interleaved = []
    i = 0

    for row1, row2 in itertools.zip_longest(rotated_Q, matrix1):
        if row1 is not None and i not in indices_duplicates:
            interleaved.append(row1)
        if row2 is not None:
            interleaved.append(row2) 
        i +=1

        
    mask = np.ones(len(interleaved), dtype=bool)
    mask[indices_duplicates] = False
    merged1=np.array(interleaved)[mask]
    np.savetxt("merged.txt", interleaved)
    
if __name__ == "__main__":
    
    main()
    
   
