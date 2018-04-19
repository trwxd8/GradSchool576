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
import PIL.Image as pil
import scipy.signal as sps
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

#add math for calculating gaussian kernel
import math

def convolve_1d(x, k):
  """
  Convolve vector x with kernel k

  Inputs: x=input vector (Nx)
          k=input kernel (Nk)

  Outputs: y=output vector (Nx)
  """
  y=np.zeros_like(x)

  """
  *******************************************
  *** TODO: write code to perform convolution
  *******************************************

  The output should be the same size as the input
  You can assume zero padding, and an odd-sized kernel
  """
  
  #Retrieve sizes for loops
  vector_size = x.size
  kernel_size = k.size
  half_size = int((kernel_size-1)/2)
        
  #Create a temporary array that is zero-padded at the front and back for edges
  results=np.zeros(vector_size+(2*half_size))
  for i in range(0, vector_size):
    results[half_size+i] = x[i]
    
  #Swap the kernel
  j = kernel_size-1
  for i in range(0,half_size):
    temp = k[i]
    k[i] = k[j]
    k[j] = temp
    j -= 1

  #Calculate the convolution for each index and output into results
  for i in range(0,vector_size):
    accumulator = 0;
    for j in range(0,kernel_size):
      accumulator += (results[i+j]*k[j])
    y[i] = accumulator


  """
  *******************************************
  """

  return y

def convolve_rows(im, k):
  """
  Convolve image im with kernel k

  Inputs: im=input image (H, W, B)
          k=1D convolution kernel (N)

  Outputs: im_out=output image (H, W, B)
  """
  im_out = np.zeros_like(im)

  """
  *****************************************
  *** TODO: write code to convolve an image
  *****************************************

  Convolve the rows of image im with kernel k
  The output should be the same size as the input
  You can assume zero padding, and an odd-sized kernel
  """

  #Retrieve sizes for loops
  height = im.shape[0]
  width = im.shape[1]
  depth = im.shape[2]
  
  #Initialize temporary array to hold convolved results
  convol_depth = np.zeros(width)
  
  #Go through each row in the image
  for h in range(0,height):
      
    #Copy the current row into holder
    #Go through depth first, so each width can be grabbed and colvolved separated
    for d in range(0,depth):
      curr_depth=np.zeros(width)
      for w in range(0,width):
        curr_depth[w] = im[h][w][d]
      
      #Convolve and move results to output
      convol_depth = convolve_1d(curr_depth, k)
      for w in range(0,width):
        im_out[h][w][d] = convol_depth[w]


  """
  *****************************************
  """

  return im_out

def gauss_kernel(sigma):
  """
  1D Gauss kernel of standard deviation sigma
  """
  l = int(np.ceil(2 * sigma))
  x = np.linspace(-l, l, 2*l+1)

  # FORNOW
  gx = np.zeros_like(x)

  """
  *******************************************
  *** TODO: compute gaussian kernel at each x
  *******************************************
  """
  
  sigma_denom = (sigma * np.sqrt(2*math.pi))
  
  for i in range(0, len(x)):
    exponent = -x[i]**2/(2*sigma**2)
    gx[i] = 1 /sigma_denom * math.exp(exponent)

  """
  *******************************************
  """

  gx = np.expand_dims(gx,0)
  return gx


  """
  *******************************************
  *** ADDED: function to compute colvolution vertically for separable gaussian functionality
  *******************************************
  """
  
def convolve_columns(im, kernel):
  """
  Convolve image im with kernel k in vertical direction
    
  Inputs: im=input image (H, W, B)
  k=1D convolution kernel (N)
    
  Outputs: im_out=output image (H, W, B)
  """

  #Retrieve sizes for loops
  height = im.shape[0]
  width = im.shape[1]
  depth = im.shape[2]

  #Initialize output image and temporary array to hold convolved results
  im_copy=np.zeros_like(im)
  convol_depth = np.zeros(height)

  #Go through each column in the image
  for w in range(0,width):
      
    #Copy the current row into holder
    #Go through depth first, so each width can be grabbed
    for d in range(0,depth):
      curr_depth=np.zeros(height)
      for h in range(0,height):
        curr_depth[h] = im[h][w][d]
      
      #Convolve and move results to output
      convol_depth = convolve_1d(curr_depth, kernel)
      for h in range(0,height):
        im_copy[h][w][d] = convol_depth[h]
        
  return im_copy

  """
  *******************************************
  """


def convolve_gaussian(im, sigma):
  """
  2D gaussian convolution
  """
  imc=np.zeros_like(im)

  """
  ***************************************
  *** TODO separable gaussian convolution
  ***************************************
  """

  #Get kernel for convolution
  kernel = gauss_kernel(sigma)
  test_kernel = kernel[0]

  imc = convolve_rows(im, test_kernel)
  imc = convolve_columns(imc, test_kernel)

  """
  ***************************************
  """
  return imc

def compute_gradients(img):

  Ix=np.zeros_like(img)
  Iy=np.zeros_like(img)

  """
  ***********************************************
  *** TODO: write code to compute image gradients
  ***********************************************
  """

  #Retrieve image dimensions
  height = img.shape[0]
  width = img.shape[1]
  depth = img.shape[2]

  #Go through each row in the image and subtract pixel from one to the left of it
  for h in range(0,height):
    for d in range(0,depth):
      for w in range(1,width):
        Ix[h][w][d] = img[h][w][d]-img[h][w-1][d]

  #Go through each pixel and ubtract pixel from one above it
  for h in range(1,height):
    for d in range(0,depth):
      for w in range(0,width):
        Iy[h][w][d] = img[h][w][d]-img[h-1][w][d]

  """
  ***********************************************
  """
  return Ix, Iy

def image_open(filename):
  """
  Returns a numpy float image with values in the range (0,1)
  """
  pil_im = pil.open(filename)
  im_np = np.array(pil_im).astype(np.float32)
  im_np /= 255.0
  return im_np

def image_save(im_np, filename):
  """
  Saves a numpy float image to file
  """
  if (len(im_np.shape)==2):
    im_np = np.expand_dims(im_np, 2)
  if (im_np.shape[2]==1):
    im_np= np.repeat(im_np, 3, axis=2)
  im_np = np.maximum(0.0, np.minimum(im_np, 1.0))
  pil_im = pil.fromarray((im_np*255).astype(np.uint8))
  pil_im.save(filename)

def image_figure(im, dpi=100):
  """
  Creates a matplotlib figure around an image,
  useful for writing to file with savefig()
  """
  H,W,_=im.shape
  fig=plt.figure()
  fig.set_size_inches(W/dpi, H/dpi)
  ax=fig.add_axes([0,0,1,1])
  ax.imshow(im)
  return fig, ax

def plot_two_images(im1, im2):
  """
  Plot two images and return axis handles
  """
  ax1=plt.subplot(1,2,1)
  plt.imshow(im1)
  plt.axis('off')
  ax2=plt.subplot(1,2,2)
  plt.imshow(im2)
  plt.axis('off')
  return ax1, ax2

def normalise_01(im):
  """
  Normalise image to the range (0,1)
  """
  mx = im.max()
  mn = im.min()
  den = mx-mn
  small_val = 1e-9
  if (den < small_val):
    print('image normalise_01 -- divisor is very small')
    den = small_val
  return (im-mn)/den

def grey_to_rgb(img):
  """
  Convert greyscale to rgb image
  """
  if (len(img.shape)==2):
    img = np.expand_dims(img, 2)

  img3 = np.repeat(img, 3, 2)
  return img3

def disc_mask(l):
  """
  Create a binary cirular mask of radius l
  """
  sz = 2 * l + 1
  m = np.zeros((sz,sz))
  x = np.linspace(-l,l,2*l+1)/l
  x = np.expand_dims(x, 1)
  m = x**2
  m = m + m.T
  m = m<1
  m = np.expand_dims(m, 2)
  return m

def convolve(im, kernel):
  """
  Wrapper for scipy convolution function
  This implements a general 2D convolution of image im with kernel
  Note that strictly speaking this is correlation not convolution

  Inputs: im=input image (H, W, B) or (H, W)
          kernel=kernel (kH, kW)

  Outputs: imc=output image (H, W, B)
  """
  if (len(im.shape)==2):
    im = np.expand_dims(im, 2)
  H, W, B = im.shape
  imc = np.zeros((H, W, B))
  for band in range(B):
    imc[:, :, band] = sps.correlate2d(im[:, :, band], kernel, mode='same')
  return imc

def coordinate_image(num_rows,num_cols,r0,r1,c0,c1):
  """
  Creates an image size num_rows, num_cols
  with coordinates linearly spaced in from r0->r1 and c0->c1
  """
  rval=np.linspace(r0,r1,num_rows)
  cval=np.linspace(c0,c1,num_cols)
  c,r=np.meshgrid(cval,rval)
  M = np.stack([r,c,np.ones(r.shape)],-1)
  return M

def transform_coordinates(coord_image, M):
  """
  Transform an image containing row,col,1 coordinates by matrix M
  """
  M=np.expand_dims(M,2)
  uh=np.dot(coord_image,M.T)
  uh=uh[:, :, 0, :]
  uh=uh/np.expand_dims(uh[:, :, 2],2)
  return uh

def warp_image(im, coords):
  """
  Warp image im using row,col,1 image coords
  """
  im_rows,im_cols,im_bands=im.shape
  warp_rows,warp_cols,_=coords.shape
  map_coords=np.zeros((3,warp_rows,warp_cols,im_bands))
  for b in range(im_bands):
    map_coords[0,:,:,b]=coords[:,:,0]
    map_coords[1,:,:,b]=coords[:,:,1]
    map_coords[2,:,:,b]=b
  warp_im = map_coordinates(im, map_coords, order=1)
  return warp_im
