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
from time import time
import numpy as np

import im_util
import match
import geometry
import ransac
import render
import interest_point

class PanoramaStitcher:
  """
  Stitch a panorama from input images
  """
  def __init__(self, images, params={}):
    self.images = images
    num_images = len(images)
    self.matches = [[None]*num_images for _ in range(num_images)]
    self.num_matches = np.zeros((num_images, num_images))
    self.stitch_order = []
    self.R_matrices = [None]*num_images
    self.params = params
    self.params.setdefault('fov_degrees', 45)
    self.params.setdefault('draw_interest_points', False)
    self.params.setdefault('draw_matches', False)
    self.params.setdefault('draw_pairwise_warp', False)
    self.params.setdefault('results_dir', os.path.expanduser('~/results/tmp/'))
	#ADDED: attributes to hold the descriptors and interest points for each image
    self.descriptors = []
    self.interest_points = []

  def stitch(self):
    """
    Match images and perform alignment
    """
    self.match_images()
    self.align_panorama()

  def match_images(self):
    """
    Match images
    """
    im=match.ImageMatcher(self.params)
    self.matches, self.num_matches, self.interest_points, self.descriptors = im.match_images(self.images)

  """
  ************************************************
  * ADDED: Function to computer rotation and consistency for cleaner code and easier referencing
  ************************************************
  """	
	
  def compute_rotation_and_consistency_pano(self, idx1, idxb):
    """
    Function to compute the rotation and count consistency between two indexes in the Panorama stitcher
    """

	#Initialize classes to match descriptors and find consistency
    desc_ex = interest_point.DescriptorExtractor()	
    rn = ransac.RANSAC()

	#Get matching descriptors for the two photos to calculate consistency
    ip1 = self.interest_points[idxb]
    ip2 = self.interest_points[idx1]
    desc1 = self.descriptors[idxb]	
    desc2 = self.descriptors[idx1]	
    match_idx = desc_ex.match_descriptors(desc1, desc2)
    ipb=ip2[:,match_idx]
		
    #Set params and variables to hold info from match_images (divide rows by 2 since they contain both entries)
    ip_rows = int(self.matches[idxb][idx1].shape[0] / 2)
    ip_cols = self.matches[idxb][idx1].shape[1]	
    ip1c = np.ndarray((ip_rows, ip_cols))
    ipbc = np.ndarray((ip_rows, ip_cols))
    ip1c[0] = self.matches[idxb][idx1][0]
    ip1c[1] = self.matches[idxb][idx1][1]
    ipbc[0] = self.matches[idxb][idx1][2]
    ipbc[1] = self.matches[idxb][idx1][3]
		
    #Get the rotation for the best match
    K1 = geometry.get_calibration(self.images[idxb].shape, self.params['fov_degrees'])
    K2 = geometry.get_calibration(self.images[idx1].shape, self.params['fov_degrees'])
    R, H = geometry.compute_rotation(ip1c, ipbc, K1, K2)
	
	#Calculate the consistency between the two photos
    inliers = rn.consistent(H, ip1, ipb)
    num_inliers_s = np.sum(inliers)

    return R, num_inliers_s

    """
    ************************************************
    * END ADDED
    ************************************************
    """
	
  def align_panorama(self):
    """
    Perform global alignment
    """

	#ADDED: Variables to be used in loops
    max_mean = -1
    max_idx = -1
    
	#hold the order of images to try transformations against, based on the count of matches
    curr_match_cnt = []
    ordered_match_cnt = []

    # FORNOW identity rotations
    num_images = len(self.images)    
    for i in range(num_images):
      self.R_matrices[i]=np.eye(3,3)

      """
      ***************************************************************
      *** TODO write code to compute a global rotation for each image
      ***************************************************************
      """  

      curr_match_cnt = []
      
      #See if the current index has the highest average matches, making it likely the best to set identity matrix for
      curr_mean = np.mean(self.num_matches[i])
      if(curr_mean > max_mean):
        max_mean = curr_mean
        max_idx = i
    
      #Go through and select the match information in tuple form: (image index, image match count)
      for j in range(0, num_images):
        #make sure not to even add matches on self
        if(i != j):
          curr_match_cnt.append((j, int(self.num_matches[i][j])))
        
      #sort list of tuples by match count, so the highest matches come before the lowest matches
      curr_match_cnt = sorted(curr_match_cnt, key=lambda x:x[1], reverse = True)
      ordered_match_cnt.append(curr_match_cnt)
    
	#Holds whether or not an image has a rotation matrix available
    transformed_indices = np.zeros(num_images)
	  
	#Create variable to ensure that the rotations dependent upon indices that aren't the identity matrix are calculated before attempting 
	#Set the identity matrix index and indicate that is already "transformed", preventing changes later on
    identity_matrix = max_idx
    transformed_indices[identity_matrix] = 1
    
    #Keep count of how many images have been transformed
    transform_cnt = 0
    
    #Set the minimum consistency percentage for rotation
    consistency_percentage = 0.9     
    #List of priority matches will always be the count of the number of pictures excluding itself
    priority_size = num_images-1
	 
    #go through so each index has an index to transform from
    while 0 in transformed_indices:
      curr_cnt = transform_cnt
      for j in range(0, priority_size):
        for i in range(0, num_images):
		
		  #Get current matched image and the count of how many matches between the two there were
          match_idx = int(ordered_match_cnt[i][j][0])
          match_cnt = int(ordered_match_cnt[i][j][1])
		  
          #Verify the rotation matrix of the dependent index is calculated and current index is not
          if(transformed_indices[match_idx] == 1 and transformed_indices[i] == 0): 
		  
            #Compute the rotation and Consistency
            curr_R, num_inliers_r = PanoramaStitcher.compute_rotation_and_consistency_pano(self, i, match_idx)
			
			#Check that the consistency is high enough before transforming
            if (num_inliers_r > consistency_percentage * match_cnt):
              curr_R = np.matmul(curr_R, self.R_matrices[match_idx])
              self.R_matrices[i][0] = curr_R[0]
              self.R_matrices[i][1] = curr_R[1]
              self.R_matrices[i][2] = curr_R[2]
			  
			  #Mark index as transformed and bump count
              transformed_indices[i] = 1
              transform_cnt += 1
			 		
        # This means at least one matrix has been added to the order. Since another image could be dependent upon this transformation, restart search with highest priority
        if(curr_cnt != transform_cnt):
          break
	  #Begin with priority 0 again if there was at least one new transformation	  
      if(curr_cnt != transform_cnt):
        continue
		
      #If no transforms occur, look to see if there are any indices that have not been transformed
      elif ( 0 in transformed_indices):
        for i in range(num_images):
          if(transformed_indices[i] == 0):
		  
            #Since no match had a high enough consistency, check if the best match is now available
            if(transformed_indices[int(ordered_match_cnt[i][j][0])] == 1):
              curr_R, _ = self.compute_rotation_and_consistency_pano(i, int(ordered_match_cnt[i][j][0]))
			  
			#if the highest match count still isn't available, default to the identity matrix
            else:
              curr_R, _ = self.compute_rotation_and_consistency_pano(i, identity_matrix)
			
			#Calculate rotation matrix and mark index as transformed
            curr_R = np.matmul(curr_R, self.R_matrices[match_idx])
            self.R_matrices[i][0] = curr_R[0]
            self.R_matrices[i][1] = curr_R[1]
            self.R_matrices[i][2] = curr_R[2]
            transformed_indices[i] = 1

    """
    ***************************************************************
    """
    
  def render(self, render_params={}):
    """
    Render output panorama
    """
    print('[ render panorama ]')
    t0=time()
    P_matrices = self.get_projection_matrices()
    pano_im=render.render_spherical(self.images, P_matrices, render_params)
    t1=time()
    print(' % .2f secs' % (t1-t0))

    return pano_im

  def get_projection_matrices(self):
    """
    Return projection matrices P such that u~=PX
    """
    num_images = self.num_matches.shape[0]
    P_matrices = [None]*num_images
    for i in range(num_images):
      Ki = geometry.get_calibration(self.images[i].shape, self.params['fov_degrees'])
      P_matrices[i] = np.dot(Ki, self.R_matrices[i])

    return P_matrices

