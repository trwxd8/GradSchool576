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
import scipy.ndimage.filters as filters
from scipy.ndimage import map_coordinates
from matplotlib.patches import Circle
import matplotlib.pyplot as plt

#Import functionality from im_util
import im_util

class InterestPointExtractor:
  """
  Class to extract interest points from an image
  """
  def __init__(self):
    self.params={}
    self.params['border_pixels']=10
    self.params['strength_threshold_percentile']=95
    self.params['supression_radius_frac']=0.01

  def find_interest_points(self, img):
    """
    Find interest points in greyscale image img

    Inputs: img=greyscale input image (H, W, 1)

    Outputs: ip=interest points of shape (2, N)
    """
    ip_fun = self.corner_function(img)
    row, col = self.find_local_maxima(ip_fun)

    ip = np.stack((row,col))
    return ip

  def corner_function(self, img):
    """
    Compute corner strength function in image im

    Inputs: img=grayscale input image (H, W, 1)

    Outputs: ip_fun=interest point strength function (H, W, 1)
    """

    H, W, _ = img.shape

    # FORNOW: random interest point function
    ip_fun = np.random.randn(H, W, 1)

    """
    **********************************************************
    *** TODO: write code to compute a corner strength function
    **********************************************************
    """

    #Set Harris Constant 
    k = .06
        
    #Calculate the amount in the positive and negative direction the SSD should go
    border = self.params['border_pixels']
    half_border = int(border/2)
    
    #Get derivative versions of image in both x and y direction
    Ix,Iy = im_util.compute_gradients(img)
    
    #Calculate squared derivations
    xx_dev = Ix * Ix
    yy_dev = Iy * Iy
    xy_dev = Ix * Iy
    
    #Experiment: Find gaussian for each square
    #sigma=4.0
    #k_results=im_util.gauss_kernel(sigma)
    #kernel = k_results[0]
    #i_xx = Ix * Ix
    #i_yy = Iy * Iy
    #i_xy = Ix * Iy
    #xx_dev = im_util.convolve_gaussian(i_xx, sigma)
    #xy_dev = im_util.convolve_gaussian(i_xy, sigma)
    #yy_dev = im_util.convolve_gaussian(i_yy, sigma)
    
    for i in range(0, H):
      for j in range(0,W):
        xx_sum = 0
        xy_sum = 0
        yy_sum = 0
        
        # Go through the neighborhood surrounding the pixel
        for y_iter in((i-half_border), (i+half_border)):
          for x_iter in((j-half_border), (j+half_border)): 
            #correct index if it falls out of boundaries. Since border_pixels ignored in next step, these values being slightly off shouldn't cause issues
            if(y_iter < 0):
              y_iter = 0
            elif(y_iter > (H-1)):
              y_iter = H-1
            if(x_iter < 0):
              x_iter = 0
            elif(x_iter > (W-1)):
              x_iter = W-1
            #Calculate the SSD value for the neighborhood
            xx_sum += xx_dev[y_iter][x_iter] 
            xy_sum += xy_dev[y_iter][x_iter] 
            yy_sum += yy_dev[y_iter][x_iter] 
            
        #calculate eigen values and assign Harris value to pixel location
        determinant = xx_sum*yy_sum - (xy_sum**2)
        trace = xx_sum + yy_sum
        #ip_fun[i][j] = determinant/trace
        ip_fun[i][j] = determinant - k*(trace**2)

        #Slide 19 from Presentation:http://www.cs.cornell.edu/courses/cs4670/2013fa/lectures/lec06_harris.pdf

    """
    **********************************************************
    """

    return ip_fun

  def find_local_maxima(self, ip_fun):
    """
    Find local maxima in interest point strength function

    Inputs: ip_fun=corner strength function (H, W, 1)

    Outputs: row,col=coordinates of interest points
    """

    H, W, _ = ip_fun.shape

    # radius for non-maximal suppression
    suppression_radius_pixels = int(self.params['supression_radius_frac']*max(H, W))

    # minimum of strength function for corners
    strength_threshold=np.percentile(ip_fun, self.params['strength_threshold_percentile'])

    # don't return interest points within border_pixels of edge
    border_pixels = self.params['border_pixels']

    # row and column coordinates of interest points
    row = []
    col = []

    # FORNOW: random row and column coordinates
    row = np.random.randint(0,H,100)
    col = np.random.randint(0,W,100)

    """
    ***************************************************
    *** TODO: write code to find local maxima in ip_fun
    ***************************************************

    Hint: try scipy filters.maximum_filter with im_util.disc_mask
    """
    #disk_mask = im_util.disc_mask(border_pixels)
    #filtered_ip_fun = filters.maximum_filter(ip_fun, footprint=disk_mask, mode='constant')
    #[mn,mx]=np.percentile(filtered_ip_fun,[5,95])
    #print("Min:",mn,"    Max:",mx)
    #filtered_strength_threshold=np.percentile(ip_fun, self.params['strength_threshold_percentile'])  
    
    all_maximums = []
    
    buffer_pixels = border_pixels
    
    #Go through the image and find the maxima per neighborhood
    for i in range(buffer_pixels,H-buffer_pixels, buffer_pixels):
      for j in range(buffer_pixels,W-buffer_pixels, buffer_pixels):
        
        #Initialize at a value that will never able to be inserted into maximums
        curr_value = -1000
        curr_x = -1
        curr_y = -1
        
        #Go through neighborhood to find largest value and index
        for y_iter in range(i, i+buffer_pixels):
          for x_iter in range(j, j+buffer_pixels):
            if(ip_fun[y_iter][x_iter] >  curr_value):
              curr_value = ip_fun[y_iter][x_iter]
              curr_x = x_iter
              curr_y = y_iter
            
        #If value is above the 95% threshold, add to list of maximums
        if(curr_value > strength_threshold):
          all_maximums.append((curr_value, curr_x, curr_y))
       
    #Sort all maximums in decreasing order, according to Harris value
    maximums_decreasing = sorted(all_maximums, key=lambda x:x[0], reverse=True)
    
    #If count is larger than row/col size, limit
    count = 0
    maximums_count = len(maximums_decreasing)
    if(maximums_count > 100):
        maximums_count = 100
    
    # Grab top 100 maximas
    for i in range(0, maximums_count):
      value, x, y = maximums_decreasing[i]
      row[count] = y
      col[count] = x
      count += 1

    """
    ***************************************************
    """

    return row, col

class DescriptorExtractor:
  """
  Extract descriptors around interest points
  """
  def __init__(self):
    self.params={}
    self.params['patch_size']=8
    self.params['ratio_threshold']=1.0

  def get_descriptors(self, img, ip):
    """
    Extact descriptors from grayscale image img at interest points ip

    Inputs: img=grayscale input image (H, W, 1)
            ip=interest point coordinates (2, N)

    Returns: descriptors=vectorized descriptors (N, num_dims)
    """
    patch_size=self.params['patch_size']
    patch_size_div2=int(patch_size/2)
    num_dims=patch_size**2

    H,W,_=img.shape
    num_ip=ip.shape[1]
    descriptors=np.zeros((num_ip,num_dims))


    for i in range(num_ip):
      row=ip[0,i]
      col=ip[1,i]

      # FORNOW: random image patch
      patch=np.random.randn(patch_size,patch_size)

      """
      ******************************************************
      *** TODO: write code to extract descriptor at row, col
      ******************************************************
      """
  
      #go through the area surrounding the interest point and copy pixels into patch
      p_y = 0
      for y_iter in range(row-patch_size_div2, row+patch_size_div2):
        p_x = 0
        for x_iter in range(col-patch_size_div2, col+patch_size_div2):
          patch[p_y][p_x] = img[y_iter][x_iter]
          p_x += 1
        p_y += 1

      """
      ******************************************************
      """

      descriptors[i, :]=np.reshape(patch,num_dims)

    # normalise descriptors to 0 mean, unit length
    mn=np.mean(descriptors,1,keepdims=True)
    sd=np.std(descriptors,1,keepdims=True)
    small_val = 1e-6
    descriptors = (descriptors-mn)/(sd+small_val)

    return descriptors

  def compute_distances(self, desc1, desc2):
    """
    Compute distances between descriptors

    Inputs: desc1=descriptor array (N1, num_dims)
            desc2=descriptor array (N2, num_dims)

    Returns: dists=array of distances (N1,N2)
    """
    N1,num_dims=desc1.shape
    N2,num_dims=desc2.shape

    ATB=np.dot(desc1,desc2.T)
    AA=np.sum(desc1*desc1,1)
    BB=np.sum(desc2*desc2,1)

    dists=-2*ATB+np.expand_dims(AA,1)+BB

    return dists

  def match_descriptors(self, desc1, desc2):
    """
    Find nearest neighbour matches between descriptors

    Inputs: desc1=descriptor array (N1, num_dims)
            desc2=descriptor array (N2, num_dims)

    Returns: match_idx=nearest neighbour index for each desc1 (N1)
    """
    dists=self.compute_distances(desc1, desc2)

    match_idx=np.argmin(dists,1)

    return match_idx

  def match_ratio_test(self, desc1, desc2):
    """
    Find nearest neighbour matches between descriptors
    and perform ratio test

    Inputs: desc1=descriptor array (N1, num_dims)
            desc2=descriptor array (N2, num_dims)

    Returns: match_idx=nearest neighbour inded for each desc1 (N1)
             ratio_pass=whether each match passes ratio test (N1)
    """
    N1,num_dims=desc1.shape

    dists=self.compute_distances(desc1, desc2)

    sort_idx=np.argsort(dists,1)

    #match_idx=np.argmin(dists,1)
    match_idx=sort_idx[:,0]

    d1NN=dists[np.arange(0,N1),sort_idx[:,0]]
    d2NN=dists[np.arange(0,N1),sort_idx[:,1]]

    ratio_threshold=self.params['ratio_threshold']
    ratio_pass=(d1NN<ratio_threshold*d2NN)

    return match_idx,ratio_pass

def draw_interest_points_ax(ip, ax):
  """
  Draw interest points ip on axis ax
  """
  for row,col in zip(ip[0,:],ip[1,:]):
    circ1 = Circle((col,row), 5)
    circ1.set_color('black')
    circ2 = Circle((col,row), 3)
    circ2.set_color('white')
    ax.add_patch(circ1)
    ax.add_patch(circ2)

def draw_interest_points_file(im, ip, filename):
  """
  Draw interest points ip on image im and save to filename
  """
  fig,ax = im_util.image_figure(im)
  draw_interest_points_ax(ip, ax)
  fig.savefig(filename)
  plt.close(fig)

def draw_matches_ax(ip1, ipm, ax1, ax2):
  """
  Draw matches ip1, ipm on axes ax1, ax2
  """
  for r1,c1,r2,c2 in zip(ip1[0,:], ip1[1,:], ipm[0,:], ipm[1,:]):
    rand_colour=np.random.rand(3,)

    circ1 = Circle((c1,r1), 5)
    circ1.set_color('black')
    circ2 = Circle((c1,r1), 3)
    circ2.set_color(rand_colour)
    ax1.add_patch(circ1)
    ax1.add_patch(circ2)

    circ3 = Circle((c2,r2), 5)
    circ3.set_color('black')
    circ4 = Circle((c2,r2), 3)
    circ4.set_color(rand_colour)
    ax2.add_patch(circ3)
    ax2.add_patch(circ4)

def draw_matches_file(im1, im2, ip1, ipm, filename):
  """
  Draw matches ip1, ipm on images im1, im2 and save to filename
  """
  H1,W1,B1=im1.shape
  H2,W2,B2=im2.shape

  im3 = np.zeros((max(H1,H2),W1+W2,3))
  im3[0:H1,0:W1,:]=im1
  im3[0:H2,W1:(W1+W2),:]=im2

  fig,ax = im_util.image_figure(im3)
  col_offset=W1

  for r1,c1,r2,c2 in zip(ip1[0,:], ip1[1,:], ipm[0,:], ipm[1,:]):
    rand_colour=np.random.rand(3,)

    circ1 = Circle((c1,r1), 5)
    circ1.set_color('black')
    circ2 = Circle((c1,r1), 3)
    circ2.set_color(rand_colour)
    ax.add_patch(circ1)
    ax.add_patch(circ2)

    circ3 = Circle((c2+col_offset,r2), 5)
    circ3.set_color('black')
    circ4 = Circle((c2+col_offset,r2), 3)
    circ4.set_color(rand_colour)
    ax.add_patch(circ3)
    ax.add_patch(circ4)

  fig.savefig(filename)
  plt.close(fig)

def plot_descriptors(desc,plt):
  """
  Plot a random set of descriptor patches
  """
  num_ip,num_dims = desc.shape
  patch_size = int(np.sqrt(num_dims))

  N1,N2=2,8
  figsize0=plt.rcParams['figure.figsize']
  plt.rcParams['figure.figsize'] = (16.0, 4.0)
  for i in range(N1):
    for j in range(N2):
      ax=plt.subplot(N1,N2,i*N2+j+1)
      rnd=np.random.randint(0,num_ip)
      desc_im=np.reshape(desc[rnd,:],(patch_size,patch_size))
      plt.imshow(im_util.grey_to_rgb(im_util.normalise_01(desc_im)))
      plt.axis('off')

  plt.rcParams['figure.figsize']=figsize0

def plot_matching_descriptors(desc1,desc2,desc1_id,desc2_id,plt):
  """
  Plot a random set of matching descriptor patches
  """
  num_inliers=desc1_id.size
  num_ip,num_dims = desc1.shape
  patch_size=int(np.sqrt(num_dims))

  figsize0=plt.rcParams['figure.figsize']

  N1,N2=1,8
  plt.rcParams['figure.figsize'] = (16.0, N1*4.0)

  for i in range(N1):
    for j in range(N2):
      rnd=np.random.randint(0,num_inliers)

      desc1_rnd=desc1_id[rnd]
      desc2_rnd=desc2_id[rnd]

      desc1_im=np.reshape(desc1[desc1_rnd,:],(patch_size,patch_size))
      desc2_im=np.reshape(desc2[desc2_rnd,:],(patch_size,patch_size))

      ax=plt.subplot(2*N1,N2,2*i*N2+j+1)
      plt.imshow(im_util.grey_to_rgb(im_util.normalise_01(desc1_im)))
      plt.axis('off')
      ax=plt.subplot(2*N1,N2,2*i*N2+N2+j+1)
      plt.imshow(im_util.grey_to_rgb(im_util.normalise_01(desc2_im)))
      plt.axis('off')

  plt.rcParams['figure.figsize'] = figsize0
