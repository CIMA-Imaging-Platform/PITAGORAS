import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

def postprocessing(hough_transform_prediction:np.array, cell_prediction:np.array):
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
    # hough_transform_prediction = median_filter(hough_transform_prediction[:,:,0], size=(5,5))
    # cell_prediction = gaussian_filter(cell_prediction[:,:,0], sigma=0.5)

    return hough_transform_prediction, cell_prediction