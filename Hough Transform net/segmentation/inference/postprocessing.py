import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.segmentation import watershed
from skimage import measure

def postprocessing(hough_transform_prediction:np.array, cell_prediction:np.array, args):
    """ Post-processing for distance label (cell + hough transform) prediction.

    Params:
    ---
    - param border_prediction: Neighbor distance prediction.
        - type border_prediction:
    - param cell_prediction: Cell distance prediction.
        - type cell_prediction:
    - param args: Post-processing settings (th_cell, th_seed, n_splitting, fuse_z_seeds).
        - type args:

    Return:
    ---
    Instance cell and hough transform predictions.
    """
    # Smooth predictions
    hough_transform_prediction = gaussian_filter(hough_transform_prediction, sigma=0.5)
    cell_prediction = gaussian_filter(cell_prediction, sigma=1.0)
    
    th_seeds = args.th_seeds
    th_cell = args.th_cell

    # Get the mask and the seeds for the watershed:
    seeds = hough_transform_prediction > th_seeds
    mask = cell_prediction > th_cell
    
    # Label each seed: 
    seeds = measure.label(seeds, background=0)

    # Remove very small seeds, corresponding to noise:
    props = measure.regionprops(seeds)
    areas = []
    for cell_props in props:
        areas.append(cell_props.area)
    min_area = 0.10 * np.mean(np.array(areas))
    min_area = np.maximum(min_area, 8)

    for cell_props in props:
        if cell_props.area <= min_area:
            seeds[seeds == cell_props.label] = 0
    seeds = measure.label(seeds, background=0)

    # Avoid empty seedings (there should be at least one cell)
    while np.max(seeds) == 0 and th_seeds > 0.05:
        th_seeds -=0.1
        seeds = hough_transform_prediction > th_seeds
        seeds = measure.label(seeds, background=0)
        props = measure.regionprops(seeds)
        for prop in props:
            if prop.area <= 4:
                seeds[seeds == prop.label] = 0
        seeds = measure.label(seeds, background=0)
    
    result = watershed(image=-cell_prediction, markers=seeds, mask=mask, watershed_line=False)

    return np.squeeze(result.astype(np.uint16)), hough_transform_prediction,cell_prediction