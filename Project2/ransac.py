# Copyright 2017 Google Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os.path
import numpy as np
from time import time

import im_util
import interest_point
import geometry

#Include math to calculate square root values
import math

class RANSAC:
  """
  Find 2-view consistent matches using RANSAC
  """
  def __init__(self):
    self.params={}
    self.params['num_iterations']=500
    self.params['inlier_dist']=10
    self.params['min_sample_dist']=2

  def consistent(self, H, p1, p2):
    """
    Find interest points that are consistent with 2D transform H

    Inputs: H=homography matrix (3,3)
            p1,p2=corresponding points in images 1,2 of shape (2, N)

    Outputs: cons=list of inliers indicated by true/false (num_points)

    Assumes that H maps from 1 to 2, i.e., hom(p2) ~= H hom(p1)
    """

    cons = np.zeros((p1.shape[1]))
    inlier_dist = self.params['inlier_dist']

    """
    ************************************************
    *** TODO: write code to check consistency with H
    ************************************************
    """

	#Convert cons to a list, as there were errors complaining about result not being int or bool, even when assigned a bool value
    cons = []
	
    #Transform the points in P1 to be ground truth   
    p1h = geometry.hom(p1)
    p1S = np.dot(H, p1h)
    p1t = geometry.unhom(p1S) 
    	
    #Calculate the distance between each point
    _, cols = p1.shape
    for i in range(0, cols):
	  
	  #calculate the distance 
      x_diff = p2[0][i] - p1t[0][i]
      y_diff = p2[1][i] - p1t[1][i]
      distance = math.sqrt((x_diff**2) + (y_diff**2))
    
      #indicate whether the point is an inlier
      if(distance > inlier_dist):
        cons.append(False)
      else:
        cons.append(True)
    
    """
    ************************************************
    """

    return cons
	
    """
    ************************************************
	* ADDED: Functions for testing/easier implementation
    ************************************************
    """	
	
  def consistent_distance(self, H, p1, p2):
    """
    Find the mean of the interest points after being transformed by H

    Inputs: H=homography matrix (3,3)
            p1,p2=corresponding points in images 1,2 of shape (2, N)

    Outputs: cons=list of inliers indicated by true/false (num_points)

    Assumes that H maps from 1 to 2, i.e., hom(p2) ~= H hom(p1)
    """
	
    #Transform the points in P1 to be ground truth   
    p1h = geometry.hom(p1)
    p1S = np.dot(H, p1h)
    p1t = geometry.unhom(p1S) 

    _, cols = p1.shape
    sum_distances = 0
    for i in range(0, cols):
	  
	  #Calculate the distance between each point
      x_diff = p2[0][i] - p1t[0][i]
      y_diff = p2[1][i] - p1t[1][i]
      sum_distances += math.sqrt((x_diff**2) + (y_diff**2))
    
	#Calculate the average distance and return value
    mean_d = sum_distances / cols
    return mean_d

  def compute_similarity(self,p1,p2):
    """
    Compute similarity transform between pairs of points

    Input: p1,p2=arrays of coordinates (2, 2)

    Output: Similarity matrix S (3, 3)

    Assume S maps from 1 to 2, i.e., hom(p2) = S hom(p1)
    """
	
    S = np.eye(3,3)

    """
    ****************************************************
    *** TODO: write code to compute similarity transform
    ****************************************************
    """  
	
    S = self.compute_similarity_from_book(p1, p2)
	#Uncomment to test other similarity results
    #S = self.compute_similarity_by_summation(p1, p2)
    #S = self.compute_similarity_by_pseudoinverse(p1, p2)
	
    """
    ****************************************************
    """
	
    return S
	
  def compute_similarity_from_book(self,p1,p2):
    """
    Compute similarity transform between pairs of points, using Jacobian and implementation from Computer Vision page 312-313

    Input: p1,p2=arrays of coordinates (2, 2)

    Output: Similarity matrix S (3, 3)

    Assume S maps from 1 to 2, i.e., hom(p2) = S hom(p1)
    """		
    #Get count of point that was input
    point_cnt = len(p1[1])
        
    J = []
    dx = []
    S = []
	 
    for i in range (0, point_cnt):
      #Calculate Jacobian and corresponding transpose 
      J.append([1, 0, p1[0][i], -p1[1][i]])
      J.append([0, 1, p1[1][i], p1[0][i]])
      diff_x = p2[0][i] - p1[0][i]
      diff_y = p2[1][i] - p1[1][i]
      dx.append([diff_x])
      dx.append([diff_y])  

    # Get the transpose of the Jacobian, and use it to calculate the A and b portions of the equation Ap = b
    J_t = np.transpose(J)
    A = np.matmul(J_t, J)
    b = np.matmul(J_t, dx)

    #Get the inverse of A and multiply by b to get (tx, ty, a, b)
    A_i = np.linalg.inv(A)
    params = np.matmul(A_i, b)
    
    # assign parameters to variables
    tx = params[0][0]
    ty = params[1][0] 
    a = params[2][0]
    b = params[3][0]

    #Return similarity matrix using parameters extracted
    return [[1+a, -b, tx], [ b, 1+a, ty ], [0, 0, 1]]

  def compute_similarity_by_pseudoinverse(self,p1,p2):
    """
    Compute similarity transform between pairs of points by appending each Jacobian and b together before performing operations
	Use a different Jacobian than listed in the book and specifically the pseudoinverse function , since p = (A*A^T)^-1*A^T*b

    Input: p1,p2=arrays of coordinates (2, 2)

    Output: Similarity matrix S (3, 3)

    Assume S maps from 1 to 2, i.e., hom(p2) = S hom(p1)
    """
    S = np.eye(3,3)
    #Get count of point that was input
    point_cnt = len(p1[1])
    
    sum_A = 0
    sum_b = 0
    
    A = []
    b = []
    S = []
	 
    for i in range (0, point_cnt):
      #Calculate Jacobian and corresponding transpose 
      A.append([-p1[1][i], p1[0][i], 1, 0])
      A.append([ p1[0][i], p1[1][i], 0, 1])
      b.append(p2[0][i])
      b.append(p2[1][i]) 
    
    #Calculate the pseudo-inverse of Matrix A and multiply by b
    pi_A = np.linalg.pinv(A)
    params = np.dot(pi_A, b)
	
	#Assign parameters and calculate similarity matrix
    a , b, tx, ty = params
    
    #Return similarity matrix
    return [[b, -a, tx], [ a, b, ty ], [0, 0, 1]]


  def compute_similarity_by_summation(self,p1,p2):
    """
    Compute similarity transform between pairs of points by adding up each A and b value as they're calculated

    Input: p1,p2=arrays of coordinates (2, 2)

    Output: Similarity matrix S (3, 3)

    Assume S maps from 1 to 2, i.e., hom(p2) = S hom(p1)
    """
	
    #Get count of point that was input
    point_cnt = len(p1[1])
    
    #Code base off book 6.1.1 2D alignment using least squares
    
    sum_A = 0
    sum_b = 0
    
    curr_J = []
    dx = []
    S = []
	 
    for i in range (0, point_cnt):
      #Calculate Jacobian and corresponding transpose 
      curr_J = [[1, 0, p1[0][i], -p1[1][i]], [0, 1, p1[1][i], p1[0][i]]]
      curr_JT = np.transpose(curr_J)
	  
	  #Calculate the difference between the two points as the delta x
      diff_x = p2[0][i] - p1[0][i]
      diff_y = p2[1][i] - p1[1][i]
      dx = [[diff_x], [diff_y]]	  
	  
      #Add values from current pass into summation variables
      sum_A += np.matmul(curr_JT, curr_J)
      sum_b +=  np.matmul(curr_JT, dx)
	
	#Calculate parameter values and assigne them to variables
    A_i = np.linalg.inv(sum_A)
    params = np.matmul(A_i, sum_b)
    tx = params[0][0]
    ty = params[1][0] 
    a = params[2][0]
    b = params[3][0]

    #Return similarity matrix
    return [[1+a, -b, tx], [ b, 1+a, ty ], [0, 0, 1]]

    """
    ************************************************
	* END ADDED
    ************************************************
    """	
	
  def ransac_similarity(self, ip1, ipm):
    """
    Find 2-view consistent matches under a Similarity transform

    Inputs: ip1=interest points (2, num_points)
            ipm=matching interest points (2, num_points)
            ip[0,:]=row coordinates, ip[1, :]=column coordinates

    Outputs: S_best=Similarity matrix (3,3)
             inliers_best=list of inliers indicated by true/false (num_points)
    """
	
	#initiailize variables for use in implementation
    S_best=np.eye(3,3)
    inliers_best=[]

    """
    *****************************************************
    *** TODO: use ransac to find a similarity transform S
    *****************************************************
    """	

    _, cols = ip1.shape
    curr_max_consistent = -1

	#Go through and find the similarity matrix that has the highest number of inliers
    for i in range(0,cols):
      for j in range(i+1, cols):
        curr_ip1_pair = [[ip1[0,i], ip1[0,j]], [ip1[1,i], ip1[1,j]]]
        curr_ipm_pair = [[ipm[0,i], ipm[0,j]], [ipm[1,i], ipm[1,j]]]

        curr_S =self.compute_similarity(curr_ip1_pair,curr_ipm_pair)
        curr_inliers=self.consistent(curr_S,ip1,ipm)
        num_consistent=np.sum(curr_inliers)
        if(num_consistent > curr_max_consistent):
          curr_max_consistent = num_consistent
          S_best = curr_S
          inliers_best = curr_inliers

	#Now extract inliers from the interest point sets
    ip1i = ip1[:, inliers_best]
    ipmi = ipm[:, inliers_best]

	#calculate shape and set min mean to value that will always be overwritten
    _, colsi = ip1i.shape
    min_mean = 100
		
    #Go through and find the similarity matrix that has the minimum distance between points
    for i in range(0,colsi):
      for j in range(i+1, colsi):
        curr_ip1i_pair = [[ip1i[0,i], ip1i[0,j]], [ip1i[1,i], ip1i[1,j]]]
        curr_ipmi_pair = [[ipmi[0,i], ipmi[0,j]], [ipmi[1,i], ipmi[1,j]]]
		
		#calculate the similarity matrix and measure the average distance between points
        curr_S=self.compute_similarity(curr_ip1i_pair,curr_ipmi_pair)
        curr_mean=self.consistent_distance(curr_S,ip1i,ipmi)
		
		#If it has a smaller mean distance, then set the S
        if(min_mean > curr_mean):
          min_mean = curr_mean
          S_best = curr_S
		  
    """
    *****************************************************
    """
	
    return S_best, inliers_best
	