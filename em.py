'''
Created on July 30, 2018

The implement of the expectation maximization algorithm
    with two-dimensional data.

@author: ning
'''

import numpy as np
import math
import matplotlib.pyplot as plt

def get_norm_prob(x, mean_x, cov_x):
    '''
    get the probability of the noraml distribution
    '''
    #cov_x = np.dot((x - mean_x).T, x - mean_x)
    det_value = np.linalg.det(cov_x)
    cov_inv = np.linalg.inv(cov_x)
    prob = np.exp(np.sum(-0.5 * np.dot((x - mean_x), cov_inv) * (x - mean_x), axis=1)) / (
        np.sqrt(np.power(2*math.pi, 2) * det_value))
    return prob

# create samples
# normal distribution 1 : 30 samples
miu1 = np.array(([5.0, 7.0]))
sigma1 = np.array(([0.1, 0.01],
                   [0.01, 0.2]))
R = np.linalg.cholesky(sigma1)
points1 = np.dot(np.random.randn(30, 2), R) + miu1
# normal distribution 2 : 70 samples
miu2 = np.array(([-9.0, -10.0]))
sigma2 = np.array(([0.5, 0.01],
                   [0.01, 0.1]))
R = np.linalg.cholesky(sigma2)
points2 = np.dot(np.random.randn(70, 2), R) + miu2

points  = np.vstack((points1, points2))
### plot samples
##plt.scatter(points1[:, 0], points1[:, 1], c='r')
##plt.scatter(points2[:, 0], points2[:, 1], c='g')
##plt.show()

# suppose there are two classes points
# the parameters need to estimate : miu1, sigma1, miu2, sigma2,
# the probability of the sample from which distribution
K = 2
nsample = 100
est_miu1 = [1, 2]
est_sigma1 = np.array(([0.5, 0.1],
                   [0.3, 0.3]))
est_miu2 = [-0.1, 0.0]
est_sigma2 = np.array(([0.3, 0.01],
                   [0.11, 0.4]))
prob = [0.5, 0.5]
p = np.zeros((nsample, K), np.float)
e_p = np.zeros_like(p)

tol = 1e-4
iter_count = 0
max_iter = 10
while True:    
    # E setp
    p[:, 0] = prob[0] * get_norm_prob(points, est_miu1, est_sigma1)
    p[:, 1] = prob[1] * get_norm_prob(points, est_miu2, est_sigma2)
    e_p = p / np.sum(p, axis=1, keepdims=True)
    # M step
    est_sigma1_new = np.dot((points - est_miu1).T, (points - est_miu1) * e_p[:, 0].reshape(100, 1)) / np.sum(e_p[:, 0])
    est_sigma2_new = np.dot((points - est_miu2).T, (points - est_miu2) * e_p[:, 1].reshape(100, 1)) / np.sum(e_p[:, 1]) 
    est_miu1_new = np.sum(e_p[:, 0].reshape(100, 1) * points, axis=0) / np.sum(e_p[:, 0])
    est_miu2_new = np.sum(e_p[:, 1].reshape(100, 1) * points, axis=0) / np.sum(e_p[:, 1])
    prob_new = np.sum(e_p, axis=0) / 100
    # whether the stop condition is satisfied
    step1 = np.sum(np.abs(est_sigma1_new - est_sigma1)) + np.sum(np.abs(est_sigma2_new - est_sigma2))
    step2 = np.sum(np.abs(est_miu1_new - est_miu1)) + np.sum(np.abs(est_miu2_new - est_miu2))
    step3 = np.sum(np.abs(prob_new - prob))
    if step1 + step2 + step3 < tol:
        break
    # update parameters
    est_miu1 = est_miu1_new
    est_miu2 = est_miu2_new
    est_sigma1 = est_sigma1_new
    est_sigma2 = est_sigma2_new
    prob = prob_new
    iter_count = iter_count + 1
    if iter_count > max_iter:
        print('Maximum iteration %d reached and '
              'the optimaztion hasn\'t converged yet'
              % (iter_count - 1))
        break
print(est_miu1)
print(est_miu2)
print(est_sigma1)
print(est_sigma2)
print(prob)
