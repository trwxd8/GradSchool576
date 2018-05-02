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

    cons2 = []
    
    #Transform the points in P1 to be ground truth   
    p1h = geometry.hom(p1)
    p1S = np.dot(H, p1h)
    p1t = geometry.unhom(p1S) 
    
    #Calculate the distance between each point
    _, cols = p1.shape
    for i in range(0, cols):
      x_diff = p2[0][i] - p1t[0][i]
      y_diff = p2[1][i] - p1t[1][i]

      #This distance or the euclidean model( sprt(E(xi-yi)^2))?
      distance = math.sqrt((x_diff**2) + (y_diff**2))
    
      #indicate whether the point is an inlier
      if(distance > inlier_dist):
        cons2.append(False)
      else:
        cons2.append(True)
    
    """
    ************************************************
    """

    return cons2
	
  def compute_similarity_28_63(self,p1,p2):
    S = np.eye(3,3)
    #Get count of point that was input
    point_cnt = p1.shape[1]
    
    #Code base off book 6.1.1 2D alignment using least squares
    
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
    
    """TOM ATTEMPT"""
    pi_A = np.linalg.pinv(A)
    A_t = np.transpose(A)
    A_l = np.dot(A_t, A)
    A_i = np.linalg.inv(A_l)
    A_r = np.dot(A_t, b)
    params = np.dot(A_i, A_r)
	
    """LIBRARY ATTEMPT"""
    #mA = np.matrix(A)
    #mb = np.matrix(b).T
    #coef = np.linalg.lstsq(mA, mb)[0].T
    #print("coef:", coef)	
    #params = np.array(coef)[0]
    print("params:", params)	
	
    a , b, tx, ty = params
    print("result: tx=",tx," ty=",ty," a=",a," b=",b)
    #np.dot(inv_A, sum_b)
	
    for i in range (0,point_cnt):
      print("transforming ",p1[i],"  to ",p2[i])
      print ("%f, %f" % (
      b*p1[i][0] - a*p1[i][1] + tx,
      b*p1[i][1] + a*p1[i][0] + ty ))
	
                
    #Return similarity matrix
    S = [[b, -a, tx], [ a, b, ty ], [0, 0, 1]]

    return S

  def compute_similarity_10_74(self,p1,p2):
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
   
    #Get count of point that was input
    point_cnt = p1.shape[1]
    
    #Code base off book 6.1.1 2D alignment using least squares
    
    sum_A = 0
    sum_J = 0
    sum_b = 0
    
    curr_J = []
    dx = []
    S = []
	 
    for i in range (0, point_cnt):
      #Calculate Jacobian and corresponding transpose 
      #A.append([p1[i][0], -p1[i][1], 1, 0])
      #A.append([p1[i][1], p1[i][0], 0, 1])
      curr_J = [[1, 0, p1[0][i], -p1[1][i]], [0, 1, p1[1][i], p1[0][i]]]
      curr_JT = np.transpose(curr_J)
      diff_x = p2[0][i] - p1[0][i]
      diff_y = p2[1][i] - p1[1][i]
      dx = [[diff_x], [diff_y]]	  
      print("J:", curr_J)
      print("JT:", curr_JT)
      print("dx:", dx)
	  	  
      sum_J += np.matrix(curr_J)
      sum_A += np.matmul(curr_JT, curr_J)
      sum_b +=  np.matmul(curr_JT, dx)
	
    """TOM ATTEMPT"""
    #pi_A = np.linalg.pinv(A)
    #A_t = np.transpose(A)
    #A_l = np.dot(A_t, A)
    A_i = np.linalg.inv(sum_A)
    #A_r = np.dot(A_t, b)
    params = np.matmul(A_i, sum_b)
	
    #params = np.dot(A_pi, b)
    #print("params_mine:", params)
    #print("a_pi:",A_pi)
    
    """LIBRARY ATTEMPT"""
    #mA = np.matrix(sum_J)
    #mb = np.matrix(sum_b).T
    #coef = np.linalg.lstsq(mA, mb)[0].T
    #print("coef:", coef)	
    #params = np.array(coef)
    #print("params:", params)	
	
    tx = params[0][0]
    ty = params[1][0] 
    a = params[2][0]
    b = params[3][0]
    print("result: tx=",tx," ty=",ty," a=",a," b=",b)
    #np.dot(inv_A, sum_b)

    for i in range (0,point_cnt):
      print("transforming ",p1[i],"  to ",p2[i])
      print ("%f, %f" % (
      b*p1[i][0] - a*p1[i][1] + tx,
      b*p1[i][1] + a*p1[i][0] + ty ))
	
                
    #Return similarity matrix
    S = [[1+a, -b, tx], [ b, 1+a, ty ], [0, 0, 1]]

    """
    ****************************************************
    """

    return S
	
  #UP TO 82 success
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
		
    #Get count of point that was input
    point_cnt = len(p1[1])
    
    #Code base off book 6.1.1 2D alignment using least squares
    
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
      #print("J:", J)
      #print("dx:", dx)

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
    #print("result: tx=",tx," ty=",ty," a=",a," b=",b)

    #for i in range (0,point_cnt):
    #  print("transforming (",p1[0][i],",",p1[1][i],")  to (",p2[0][i],",",p2[1][i],")")
    #  print ("%f, %f" % (
    #  b*p1[i][0] - a*p1[i][1] + tx,
    #  b*p1[i][1] + a*p1[i][0] + ty ))
	
                
    #Return similarity matrix using parameters extracted
    S = [[1+a, -b, tx], [ b, 1+a, ty ], [0, 0, 1]]

    """
    ****************************************************
    """

    return S

  def compute_similarity_0(self,p1,p2):
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
   
    #Get count of point that was input
    point_cnt = p1.shape[1]
    
    #Code base off book 6.1.1 2D alignment using least squares
    
    A = []
    b = []
	 
    for i in range (0, point_cnt):
      #Calculate Jacobian and corresponding transpose 
      A.append([p1[i][0], p1[i][1], 0, 0])
      A.append([0, 0, p1[i][0], p1[i][1]])
      diff_x = p2[i][0]
      diff_y = p2[i][1]
      b.append([diff_x])
      b.append([diff_y])	  
    print("A:", A)
    print("b:", b)
	
    A_i = np.linalg.inv(A)
    params = np.matmul(A_i, b)
	
    #params = np.dot(A_pi, b)
    print("params_mine:", params)
    #print("a_pi:",A_pi)
    
    a = params[0][0]
    b = params[1][0] 
    tx = params[2][0]
    ty = params[3][0]
    print("result: tx=",tx," ty=",ty," a=",a," b=",b)
    #np.dot(inv_A, sum_b)	
                
    #Return similarity matrix
    S = [[1+a, -b, tx], [ b, 1+a, ty ], [0, 0, 1]]

    """
    ****************************************************
    """

    return S
	
  def ransac_similarity(self, ip1, ipm):
    """
    Find 2-view consistent matches under a Similarity transform

    Inputs: ip1=interest points (2, num_points)
            ipm=matching interest points (2, num_points)
            ip[0,:]=row coordinates, ip[1, :]=column coordinates

    Outputs: S_best=Similarity matrix (3,3)
             inliers_best=list of inliers indicated by true/false (num_points)
    """
    S_best=np.eye(3,3)
    inliers_best=[]

    """
    *****************************************************
    *** TODO: use ransac to find a similarity transform S
    *****************************************************
    """
    sample=[0,1]
    _, cols1 = ip1.shape
    _, colsm = ipm.shape


    curr_max_consistent = -1

    for i1 in range(0,cols1):
      for j1 in range(i1+1, cols1):
        curr_ip1_pair = [[ip1[0,i1], ip1[0,j1]], [ip1[1,i1], ip1[1,j1]]]
        curr_ipm_pair = [[ipm[0,i1], ipm[0,j1]], [ipm[1,i1], ipm[1,j1]]]

        #print(curr_ip1_pair,"  to  ",curr_ipm_pair)
        curr_S=self.compute_similarity(curr_ip1_pair,curr_ipm_pair)
        curr_inliers=self.consistent(curr_S,ip1,ipm)
        num_consistent=np.sum(curr_inliers)
        if(num_consistent > curr_max_consistent):
          curr_max_consistent = num_consistent
          S_best = curr_S
          inliers_best = curr_inliers

    """
    *****************************************************
    """

    return S_best, inliers_best
	