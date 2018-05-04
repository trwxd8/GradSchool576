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

import numpy as np

def compute_rotation(ip1, ip2, K1, K2):
  """
  Find rotation matrix R such that |r2 - R*r1|^2 is minimised

  Inputs: ip1,ip2=corresponding interest points (2,num_points),
          K1,K2=camera calibration matrices (3,3)

  Outputs: R=rotation matrix R (3,3)

  r1,r2 are corresponding rays (unit normalised camera coordinates) in image 1,2
  """

  R=H=np.eye(3,3)

  """
  **********************************************************************
  *** TODO: write code to compute 3D rotation between corresponding rays
  **********************************************************************
  """

  _, cols = ip1.shape
    
  #ip1_means = np.mean(ip1, 1) 
  #ip2_means = np.mean(ip2, 1)
  #print("means 1:", ip1_means)
  #print("means 2:", ip2_means) 
    
  #for i in range(0, cols):
  #  ip1
    
  K1_inv = np.linalg.inv(K1)
  K2_inv = np.linalg.inv(K2)

  u1 = hom(ip1)
  u2 = hom(ip2)

  rayset1 = np.matmul(K1_inv, u1) 
  rayset2 = np.matmul(K2_inv, u2)

  rayset1_t = np.transpose(rayset1)

  covariance = np.matmul(rayset2, rayset1_t)
    
  u, _, vt = np.linalg.svd(covariance, compute_uv=True)
        
  R = np.matmul(u, vt)  
  H_part = np.matmul(K2, R)
  H = np.matmul(H_part, K1_inv) 
   
  #R=H=np.eye(3,3) 
    
  """
    [mn,mx]=np.percentile(ip_fun,[5,95])
    small_val=1e-9
    ip_fun_norm=(ip_fun-mn)/(mx-mn+small_val)
    ip_fun_norm=np.maximum(np.minimum(ip_fun_norm,1.0),0.0)    
  """

  """
  **********************************************************************
  """
  return R, H

def get_calibration(imshape, fov_degrees):
  """
  Return calibration matrix K given image shape and field of view

  See note on calibration matrix in documentation of K(f, H, W)
  """
  H, W, _ = imshape
  f = max(H,W)/(2*np.tan((fov_degrees/2)*np.pi/180))
  K1 = K(f,H,W)
  return K1

def K(f,H,W):
  """
  Return camera calibration matrix given focal length and image size

  Inputs: f=focal length, H=image height, W=image width all in pixels

  Outputs: K=calibration matrix (3, 3)

  The calibration matrix maps camera coordinates [X,Y,Z] to homogeneous image
  coordinates ~[row,col,1]. X is assumed to point along the positive col direction,
  i.e., incrementing X increments the col dimension in the image
  """
  K1=np.zeros((3,3))
  K1[0,1]=K1[1,0]=f
  K1[0,2]=H/2
  K1[1,2]=W/2
  K1[2,2]=1
  return K1

def hom(p):
  """
  Convert points to homogeneous coordiantes
  """
  ph=np.concatenate((p,np.ones((1,p.shape[1]))))
  return ph

def unhom(ph):
  """
  Convert points from homogeneous to regular coordinates
  """
  p=ph/ph[2,:]
  p=p[0:2,:]
  return p
