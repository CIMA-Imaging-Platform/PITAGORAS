import cv2
import numpy as np
import pandas as pd
import scipy.ndimage as ndi 

from skimage import filters, morphology
from tifffile import imsave
from skimage.measure import regionprops, label
from skimage.transform import hough_circle_peaks
from skimage.segmentation import watershed
from skimage.exposure import adjust_sigmoid
from scipy.ndimage.morphology import distance_transform_edt

def _hough_circle(img, Gx, Gy, radius):
    """Perform a circular Hough transform.

    Parameters
    ----------
    - image : Input image with nonzero values representing edges.
    - Gx: Input image with the horizontal gradient.
    - Gy: Input image with the vertical gradient.
    - radius : Radii at which to compute the Hough transform. (Floats are converted to integers.)

    Returns
    -------
    H : 3D ndarray (radius index, (M + 2R, N + 2R) ndarray)
        Hough transform accumulator for each radius.
        R designates the larger radius if full_output is True.
        Otherwise, R = 0.
    """

    xmax, ymax = img.shape

    # compute the nonzero indexes as a way of reducing the computational work.
    x, y = np.nonzero(img[:,:])
    # number of pixels that are non zero in the image.
    num_pixels = x.size
    # coordenate origin.
    offset = 0
    # Calculate the orientation of the image and splits its components Ox Oy
    teta = np.arctan2(Gy, Gx)

    sin_dir = np.sin(teta)
    cos_dir = np.cos(teta)

    acumulator = np.zeros((radius.size,
                            xmax + 2 * offset,
                            ymax + 2 * offset), dtype=float)

    for i, rad in enumerate(radius):

        rad_sin = rad * sin_dir
        rad_cos = rad * cos_dir
        # For each non zero pixel
        for p in range(num_pixels):
            # Plug the circle at (px, py),
            # its coordinates are (tx, ty)
            tx = x[p] + rad_cos[x[p], y[p]]
            ty = y[p] + rad_sin[x[p], y[p]]

            # Now the idea is to increase the incr variable by the time two or more tx and ty pixels concorde.
            tx = np.clip(np.round(tx), 0, xmax + 2 * offset - 1).astype(int)
            ty = np.clip(np.round(ty), 0, ymax + 2 * offset - 1).astype(int)

            # Increase the accumulator
            acumulator[i, tx, ty] += img[x[p], y[p]]
            # acumulator[i, tx, ty] += 1

        # And normalize as the input image 0, 2**16:
        acumulator[i,:,:] = 2**16 *(acumulator[i,:,:] - acumulator[i,:,:].min()) / (np.min(acumulator[i,:,:].max() - acumulator[i,:,:].min()))

        return acumulator

def hough_circle(image: np.array, Gx: np.array, Gy: np.array, radius: np.array):
    """Perform a circular Hough transform.

    Parameters
    ----------
    - image : Input image with nonzero values representing edges.
    - Gx: Input image with the horizontal gradient.
    - Gy: Input image with the vertical gradient.
    - radius : Radii at which to compute the Hough transform. (Floats are converted to integers.)

    Returns
    -------
    H : 3D ndarray (radius index, (M + 2R, N + 2R) ndarray)
        Hough transform accumulator for each radius.
        R designates the larger radius if full_output is True.
        Otherwise, R = 0.

    Examples
    --------
    >>> from skimage.transform import hough_circle
    >>> from skimage.draw import circle_perimeter
    >>> img = np.zeros((100, 100), dtype=bool)
    >>> rr, cc = circle_perimeter(25, 35, 23)
    >>> img[rr, cc] = 1
    >>> try_radii = np.arange(5, 50)
    >>> res = hough_circle(img, try_radii)
    >>> ridx, r, c = np.unravel_index(np.argmax(res), res.shape)
    >>> r, c, try_radii[ridx]
    (25, 35, 23)

    """
    radius = np.atleast_1d(np.asarray(radius))
    return _hough_circle(image, Gx, Gy, radius.astype(np.intp))

def hough_transform_2D(image:np.array, labels:np.array):
    """ Circular Hough Transform on cell labels.

    Parameters
    ----------
        - img: Intensity-coded instance original cell image.
        - labels: Intensity-coded instance segmentation label image.
    
    Returns
    ------- 
        - Cell label Hough Transform image.
    """
    # Load the images, and deleate the third dimesion
    img, mask = image.squeeze(), labels.squeeze()

    # Preallocation
    label_dist = np.zeros(shape= img.shape, dtype= float)
    hough_transform = np.zeros(shape= img.shape, dtype= float)

    # Initialize the props of each label and a variable where to store them
    regions = regionprops(mask)

    # Loop that runs within each label to get its parameters
    for region in regions:

        data = {'mask': int, 'radii': float, 'bounding box': list, 'centroid-x': float, 'centroid-y':float}
        data['mask'] = region.label
        data['radii'] = (region.equivalent_diameter/2)
        data['bounding box'] = (region.bbox)
        data['centroid-x'] = (region.centroid[0])
        data['centroid-y'] = (region.centroid[1])

        # Isolating each cell
        img_crop = img[
                    int(max(data['bounding box'][0]-20, 0)):int(min(data['bounding box'][2]+20, img.shape[0])), 
                    int(max(data['bounding box'][1]-20, 0)):int(min(data['bounding box'][3]+20, img.shape[1]))
                    ]
        label_crop = mask[
                    int(max(data['bounding box'][0]-20, 0)):int(min(data['bounding box'][2]+20, img.shape[0])), 
                    int(max(data['bounding box'][1]-20, 0)):int(min(data['bounding box'][3]+20, img.shape[1]))
                    ]
        
        ## Eclidian Distance Transform
        # Apply the Euclidian Distance Transform:
        label_crop_dist = distance_transform_edt(label_crop == data['mask'])

        # Normalize the distance map [0,1]
        max_dist = np.max(label_crop_dist)
        if max_dist > 0:
            label_crop_dist = label_crop_dist / max_dist
        else:
            continue

        # Join the each cell EDT together
        label_dist[
                int(max(data['bounding box'][0]-20, 0)):int(min(data['bounding box'][2]+20, img.shape[0])), 
                int(max(data['bounding box'][1]-20, 0)):int(min(data['bounding box'][3]+20, img.shape[1]))
                    ] += label_crop_dist
        

        ## Hough Transform
        # Dilation of the label_crop to increment the boundaries of the TH, and be more conservative
        kernel = np.ones((5, 5))
        label_crop = cv2.dilate(label_crop, kernel)

        # Getting the gradient on each direction
        Gx = ndi.sobel(img_crop, axis=0, mode= 'nearest')
        Gy= ndi.sobel(img_crop, axis=1, mode= 'nearest')
        grad = np.sqrt(np.square(Gx) + np.square(Gy)) # Obtain the combinated gradient as the square root of the individual squared 

        # Look for the locations different to the label and equal them to 0
        grad[np.where(data['mask'] != label_crop)] = 0

        # Applay the Hough Transform:
        # Range of radii to search is [label_radii-5, label_radii+5], an interval of 0.1
        hough_radii = np.sort(abs(np.arange(data['radii']-1, data['radii']+1, 0.1)))
        hough_res = hough_circle(grad, Gx= Gx, Gy= Gy, radius= hough_radii)

        # Select the most prominent number of peaks within each accumulator:
        _ , _, _, radii = hough_circle_peaks(hough_res, hough_radii, num_peaks = 1, 
                                             total_num_peaks= 1, normalize= False)
        # Finally join each HT as the final result adn add the third pseudo-color dimension:
        crop_HT = hough_res[np.where(hough_radii == radii[0])[0][0],:,:]
        crop_HT = filters.median(crop_HT, selem= morphology.disk(5))

        hough_transform[
                    int(max(data['bounding box'][0]-20, 0)):int(min(data['bounding box'][2]+20, img.shape[0])), 
                    int(max(data['bounding box'][1]-20, 0)):int(min(data['bounding box'][3]+20, img.shape[1]))
                        ] += crop_HT

    # Normalize the hough transform from [0,1]
    max_hough_transform_label = np.max(hough_transform)
    if max_hough_transform_label > 0:
        hough_transform = hough_transform / max_hough_transform_label

    hough_transform = np.expand_dims(hough_transform, axis=-1)
    label_dist = np.expand_dims(label_dist, axis=-1)

    return hough_transform.astype(np.float16), label_dist.astype(np.float16)

def segmentation1(hough_img, dist_img):
    
    flt = filters.median(hough_img, selem=np.ones((10,10))) # Depending on the cell size...
    flt = ((flt - np.min(flt))/(np.max(flt) - np.min(flt))) # Normalize the intensity [0-1]
    flt = np.clip(flt, 0, 1)
    flt = np.round(flt, 2)

    seeds = flt > 0.07 # Apply a threshold on the filtered HT to get the seeds
    seeds = label(seeds)
    mask = dist_img > 0 # Apply a threshold on the EDT to get the boundary image
    # Use the edt as the flooding image, seeds and mask, for the Watershed transform
    seg = watershed(image= - dist_img, markers= seeds, mask= mask)

    return seg

def segmentation2(hough_img, dist_img, cutoff= 0.2, th_seeds= 0.5):

    flt = filters.median(hough_img, selem=np.ones((10,10))) # Depending on the cell size...
    flt /= np.max(flt) # Normalize the intensity [0-1]
    flt = adjust_sigmoid(flt, cutoff= cutoff)
    flt = (flt - np.min(flt))/(np.max(flt) - np.min(flt))
    seeds = flt > th_seeds # Apply a threshold on the filtered HT to get the seeds
    seeds = label(seeds)
    mask = dist_img > 0 # Apply a threshold on the EDT to get the boundary image
    # Use the edt as the flooding image, seeds and mask, for the Watershed transform
    seg = watershed(image= - dist_img, markers= seeds, mask= mask)

    return seg

def gradient(image: np.array, mask: np.array):

    """
    This function aims to calculate the gradient image in both directions x,y using the sobel filter: 

    Parameters:
    ---
    - image: instensiy image aiming to get its gradient.
    - mask: label image.
    
    Return:
    ---
    - Gx: gradient image in the X direction.
    - Gy: grdient image in the Y direction.
    """
    # We are looking for getting each gradient of the cell isolatly,
    # by woirking each cell and jointly adding them to an image with the same 
    # size as the crop.

    # Start by woking on the pixel within the mask.
    # As the annotations may not be perfect we doa dilation in order to be more
    # conservatives.

    # Prellocation
    Gx = np.zeros(shape= image.shape)
    Gy = np.zeros(shape= image.shape)

    # Get each instance from the mask
    regions = regionprops(mask) 
    
    for region in regions:
        
        label = region.label
        bbox = region.bbox

        # Isolating each nucleus  
        image_crop = image[
                int(max(bbox[0]-10, 0)):int(min(bbox[2]+10, image.shape[0])), 
                int(max(bbox[1]-10, 0)):int(min(bbox[3]+10, image.shape[1]))
                ]
        label_crop = mask[
                int(max(bbox[0]-10, 0)):int(min(bbox[2]+10, image.shape[0])), 
                int(max(bbox[1]-10, 0)):int(min(bbox[3]+10, image.shape[1]))
                ]
        
        # Dilation of the label_crop to increment the boundaries of the TH, and be more conservative
        kernel = np.ones((5, 5))
        label_crop = cv2.dilate(label_crop, kernel)

        # Getting the gradient on each direction
        Gx_crop = ndi.sobel(image_crop, axis=0, mode= 'nearest')
        Gy_crop = ndi.sobel(image_crop, axis=1, mode= 'nearest')

        # Look for the locations different to the label and equal them to 0
        Gx_crop[np.where(label_crop != label)] = 0
        Gy_crop[np.where(label_crop != label)] = 0

        # Join each gradient cell gradient in the result image Gx
        Gx[
            int(max(bbox[0]-10, 0)):int(min(bbox[2]+10, image.shape[0])), 
            int(max(bbox[1]-10, 0)):int(min(bbox[3]+10, image.shape[1]))
            ] += Gx_crop
        
        Gy[
            int(max(bbox[0]-10, 0)):int(min(bbox[2]+10, image.shape[0])), 
            int(max(bbox[1]-10, 0)):int(min(bbox[3]+10, image.shape[1]))
            ] += Gy_crop

    return Gx, Gy