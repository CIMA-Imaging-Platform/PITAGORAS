
import hashlib
import json
import math
import numpy as np
import os
import pandas as pd
import shutil
import tifffile as tiff
 
from pathlib import Path
from random import shuffle

from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import pdist, squareform
from segmentation.training.Hough_Transform import hough_transform_2D
from skimage.transform import rescale
from skimage.measure import regionprops
from skimage.morphology import binary_closing, binary_opening
from torch.utils.data import Dataset

'''
    This script contains the functions to create the data partition to be handled to the 
    training network step.
'''
def adjust_dimensions(crop_size: int, *imgs):
    ''' Adjust dimensions so that only 'complete' crops are generated, by padding.

    Args:
    - crop_size: Size of the (square) crops.
    - imgs: Images to adjust the dimensions.
    
    Return: img with adjusted dimension.
    '''

    img_adj = []

    # Add pseudo color channels
    for img in imgs:
        img = np.expand_dims(img, axis=-1)

        pads = [] # Determines number of rows and columns to pad.
        for i in range(2): #We do for x,y axis
            if img.shape[i] < crop_size:
                pads.append((0, crop_size - (img.shape[i] % crop_size)))
            elif img.shape[i] == crop_size:
                pads.append((0, 0))
            else:
                # If the difference between the division is small enough (7,5% of the image's size) not to consider the loss of information, then; goes for it.
                if (img.shape[i] % crop_size) < 0.075 * img.shape[i]:
                    idx_start = (img.shape[i] % crop_size) // 2
                    idx_end = img.shape[i] - ((img.shape[i] % crop_size) - idx_start)
                    if i == 0:
                        img = img[idx_start:idx_end, ...]
                    else:
                        img = img[:, idx_start:idx_end, ...]
                    pads.append((0, 0))
                else:
                    pads.append((0, crop_size - (img.shape[i] % crop_size)))

        img = np.pad(img, (pads[0], pads[1], (0, 0)), mode='constant') # The (0,0) is used for the third dimension of the image.

        img_adj.append(img)

    return img_adj


def close_mask(mask: np.array, 
               apply_opening=False, 
               kernel_closing=np.ones((10, 10)), 
               kernel_opening=np.ones((10, 10))):
    ''' 
    Morphological closing of STs.

    Args:
    - mask: Segmentation mask (gold truth or silver truth).
    - apply_opening: Apply opening or not (basically needed to correct slices from 3D silver truth).
    - kernel_closing: Kernel for closing.
    - kernel_opening: Kernel for opening.

    Return: Closed (and opened) mask.
    '''

    # Get nucleus ids and close/open the nuclei separately
    nucleus_ids = get_nucleus_ids(mask)
    hlabel = np.zeros(shape=mask.shape, dtype=mask.dtype)
    for nucleus_id in nucleus_ids:
        nucleus = mask == nucleus_id
        # Close nucleus gaps
        nucleus = binary_closing(nucleus, kernel_closing)
        # Remove small single not connected pixels
        if apply_opening:
            nucleus = binary_opening(nucleus, kernel_opening)
        hlabel[nucleus] = nucleus_id.astype(mask.dtype)

    return hlabel


def copy_train_data(source_path, target_path, idx):
    """  Copy generated training data crops.

    - param source_path: Directory containing the training data crops.
        - type source_path: pathlib Path object
    - param target_path: Directory to copy the training data crops into.
        - type target_path: pathlib Path Object
    - param idx: path/id of the training data crops to copy.
        - type idx: pathlib Path Object

    - return: None
    """
    shutil.copyfile(str(source_path / "img_{}.tif".format(idx)), str(target_path / "img_{}.tif".format(idx)))
    shutil.copyfile(str(source_path / "dist_cell_{}.tif".format(idx)), str(target_path / "dist_cell_{}.tif".format(idx)))
    shutil.copyfile(str(source_path / "hough_transform_{}.tif".format(idx)), str(target_path / "hough_transform_{}.tif".format(idx)))
    shutil.copyfile(str(source_path / "mask_{}.tif".format(idx)), str(target_path / "mask_{}.tif".format(idx)))
    return


def copy_train_set(source_path, target_path, mode='GT'):
    """  Copy generated training data sets (train and val).

    :param source_path: Directory containing the training data sets.
        :type source_path: pathlib Path object.
    :param target_path: Directory to copy the training data sets into.
        :type target_path: pathlib Path Object
    :param mode: 'GT' deletes possibly existing train and val directories.
        :type mode: str
    :return: None
    """

    if mode == 'GT':
        os.rmdir(str(target_path / 'train'))
        os.rmdir(str(target_path / 'val'))
        shutil.copytree(str(source_path / 'train'), str(target_path / 'train'))
        shutil.copytree(str(source_path / 'val'), str(target_path / 'val'))
    else:
        shutil.copytree(str(source_path / 'train'), str(target_path / 'train_st'))
        shutil.copytree(str(source_path / 'val'), str(target_path / 'val_st'))


def downscale(img, scale: np.float16, order=2, aa=None):
    ''' Downscale image and segmentation ground truth.

    Args:    
    - img: Image to downscale
    - scale: Scale for downscaling.
    - order: Order of the polynom used.
    - aa: apply anti-aliasing (not recommended for the masks).

    Return: downscale images.
    '''
    if len(img.shape) == 3:
        scale_img = (1, scale, scale)
    else:
        scale_img = (scale, scale)
    img = rescale(img, scale=scale_img, order=order, anti_aliasing=aa, preserve_range=True).astype(img.dtype)

    return img


def generate_data(img: np.array, 
                  mask: np.array, 
                  tra_gt: np.array, 
                  td_settings: dict, 
                  cell_type: str, 
                  subset: str, 
                  frame: str, 
                  path: str, 
                  slice_idx=None):
    
    ''' Function to crop the images, masks and tracking GT, before the Hough Transform.

    Args:
    ---    
    - img: original intensity image. Normalized from [0 65356]
    - mask: labeled image.
    - tra_gt: tracking groudtruth.
    - td_settings: settings for the Hough Transform.
    - cell_type: Cell types to train the model with. When several cell types are used, a new name of the training is built. 
    - split: Use a single ('01'/'02) or both subsets ('01+02')
    - frame: value of the sequence of the study case.
    - path: Path of the parent cell type folder, where the croped images will be saved.
    - slice_idx: number in which the masks have been annotated.

    Return:
     ---
       crop_idx.
    '''
    # Adjust image dimensions for appropriate cropping
    img, mask, tra_gt = adjust_dimensions(td_settings['crop_size'], img, mask, tra_gt) # Here have added the pseudo-channle [third dimension]

    # Cropping
    crop_idx = 0
    nx, ny = math.floor(img.shape[1] / td_settings['crop_size']), math.floor(img.shape[0] / td_settings['crop_size'])
    for y in range(ny):
        for x in range(nx):

            # Crop
            img_crop, mask_crop, tra_gt_crop = get_crop(x, y, td_settings['crop_size'],
                                                        img, mask, tra_gt)
            # Get crop name
            if slice_idx is not None:
                crop_name = '{}_{}_{}_{:02d}_{:02d}_{:02d}.tif'.format(cell_type, subset, frame, slice_idx, y, x) # When we have 3D annotations
            else:
                crop_name = '{}_{}_{}_{:02d}_{:02d}.tif'.format(cell_type, subset, frame, y, x)

            # Check cell number by comparing the annotation number of cells of: TRA/SEG
            if np.sum(mask_crop[10:-10, 10:-10, 0] > 0) < td_settings['min_area']:  # only cell parts / no cell
                continue
            if np.sum(img_crop == 0) > (0.66 * img_crop.shape[0] * img_crop.shape[1]):  # if more that the 66% of the image_crop is background, don't save
                if np.min(img_crop[:100, :100, ...]) == 0: # if from 0:100 of the image_crop the min value is 0
                    if np.sum(gaussian_filter(np.squeeze(img_crop), sigma=1) == 0) > (0.66 * img_crop.shape[0] * img_crop.shape[1]): # if more than the 66% of the image_crop is 0 after gaussian filter
                      continue
                else: 
                    continue

            # Classify the cropped images in A and B folders, depending on the percentage of annotated cells,
            # compared to between the seg_mask and the tra_mask.
            tr_ids, mask_ids = get_nucleus_ids(tra_gt_crop), get_nucleus_ids(mask_crop)

            props_crop, n_part = regionprops(mask_crop), 0
            for cell in props_crop:
                if cell.area <= 0.1 * td_settings['min_area'] and td_settings['scale'] == 1:  # needed since tra_gt seeds are smaller
                    n_part += 1
            if (len(mask_ids) - n_part) >= len(tr_ids):  # A: all cells annotated
                crop_quality = 'A'
            elif (len(mask_ids) - n_part) >= 0.8 * len(tr_ids):  # >= 80% of the cells annotated
                crop_quality = 'B'
            else:
                continue

            hough_transform, label_dist = hough_transform_2D(image=img_crop, labels=mask_crop)

            # Save the images
            tiff.imsave(str(path / crop_quality / 'img_{}'.format(crop_name)), img_crop)
            tiff.imsave(str(path / crop_quality / 'mask_{}'.format(crop_name)), mask_crop.astype(np.uint8))
            tiff.imsave(str(path / crop_quality / 'hough_transform_{}'.format(crop_name)), hough_transform)
            tiff.imsave(str(path / crop_quality / 'dist_cell_{}'.format(crop_name)), label_dist)
            
            # Increase crop counter, until to achieve the last crop of the original image.
            crop_idx += 1

    return crop_idx


def get_crop(x, y, crop_size, *imgs):
    """ Get crop from image.

    - param x: Grid position (x-dim), over the crop matrix.
        - type x: int
    - param y: Grid position (y-dim), over the crop matrix.
        - type y: int
    - param crop_size: size of the (square) crop
        - type crop_size: int
    - param imgs: Images to crop.
        - type imgs:
    - return: img crop.
    """

    imgs_crop = []

    for img in imgs:
        img_crop = img[y * crop_size:(y + 1) * crop_size, x * crop_size:(x + 1) * crop_size, :]
        imgs_crop.append(img_crop) # Array whose 3d dimension contains the cropped images for each img and mask passed.

    return imgs_crop


def get_mask_ids(path_data: Path, ct: str, split: str):
    """ Get ids of the masks of a specific cell type/dataset.

    Args:
    - path_data: Path to the directory containing the Cell Tracking Challenge training sets.
    - ct: cell type/dataset.
    - split: Use a single ('01'/'02) or both subsets ('01+02')
    
    Return: mask ids, increment for selecting slices.
    """
    # Go through each slice for 3D annotations if not stated otherwise later
    slice_increment = 1

    # Get mask ids
    mask_ids_01, mask_ids_02 = [], []
    if '01' in split:
        mask_ids_01 = sorted((path_data / ct / '01_GT' / 'SEG').glob('*.tif'))
    if '02' in split:
        mask_ids_02 = sorted((path_data / ct / '02_GT' / 'SEG').glob('*.tif'))
    mask_ids = mask_ids_01 + mask_ids_02

    # Shuffle list
    shuffle(mask_ids)
                
    return mask_ids, slice_increment


def get_nucleus_ids(img: np.array):
    """ Get nucleus ids in intensity-coded label image.
    Args:
    - img: Intensity-coded nuclei image.

    Return: List of nucleus ids.
    """

    values = np.unique(img)
    values = values[values > 0]

    return values


def get_td_settings(mask_id_list: list, crop_size: int):
    """ Get settings for the Hough Transform.

    Args:
    - mask_id_list: List of all segmentation GT ids (list of pathlib Path objects).
    - crop_size: value to know the crop size.
    
    Return:
    - search_radius
    - min_area 
    - max_mal
    - scale: factor to apply the image
    - crop_size
    - min_x_dist
    - min_y_dist
    """

    # Load all GT and get cell parameters to adjust parameters for the distance transform calculation
    object, diameters, major_axes, areas, Center_X, Center_Y = [], [], [], [], [], []
    for mask_id in mask_id_list:
        mask = tiff.imread(str(mask_id))
        if len(mask.shape) == 3:
            for i in range(len(mask)):
                props = regionprops(mask[i])
                for cell in props:  # works not as intended for 3D GTs
                    major_axes.append(cell.major_axis_length)
                    diameters.append(cell.equivalent_diameter)
                    areas.append(cell.area)
                    Center_X.append(cell.centroid[0])
                    Center_Y.append(cell.centroid[1])
                    object.append(cell.label)
        else:
            props = regionprops(mask)
            for cell in props:
                major_axes.append(cell.major_axis_length)
                diameters.append(cell.equivalent_diameter)
                areas.append(cell.area)
                Center_X.append(cell.centroid[0])
                Center_Y.append(cell.centroid[1])
                object.append(cell.label)

    # Get the minimum distance x,y between objects as apriori information in the Hough Transform
    df = pd.DataFrame({'Object': object, 
                       'Center-x': Center_X,
                       'Center-y': Center_Y})
    
    centroids = df[['Center-x', 'Center-y']].values
    distances = pdist(centroids)
    dist_matrix = squareform(distances)
    for i in range(len(dist_matrix)):
        dist_matrix[i, i] = float('inf')
    min_distance = dist_matrix.min()
    min_indices = np.where(dist_matrix == min_distance)

    min_x_dist = abs(df.loc[min_indices[0][0], 'Center-x'] - df.loc[min_indices[1][0], 'Center-x'])
    min_y_dist = abs(df.loc[min_indices[0][0], 'Center-y'] - df.loc[min_indices[1][0], 'Center-y'])

    # Get maximum and minimum diameter and major axis length and set search radius for distance transform
    max_diameter, min_diameter = int(np.ceil(np.max(np.array(diameters)))), int(np.ceil(np.min(np.array(diameters))))
    mean_diameter, std_diameter = int(np.ceil(np.mean(np.array(diameters)))), int(np.std(np.array(diameters)))
    max_mal = int(np.ceil(np.max(np.array(major_axes))))
    min_area = int(0.95 * np.floor(np.min(np.array(areas))))
    search_radius = mean_diameter + std_diameter

    # Some simple heuristics for large cells. If enough data are available scale=1 should work in most cases
    if max_diameter > 200 and min_diameter > 35:
        if max_mal > 2 * max_diameter:  # very longish and long cells not made for neighbor distance
            scale = 0.5
            search_radius = min_diameter + 0.5 * std_diameter
        elif max_diameter > 300 and min_diameter > 60:
            scale = 0.5
        elif max_diameter > 250 and min_diameter > 50:
            scale = 0.6
        else:
            scale = 0.7
        min_area = (scale ** 2) * min_area
        max_mal = int(np.ceil(scale * max_mal))
        search_radius = int(np.ceil(scale * search_radius))

    else:
        scale = 1

    return {'search_radius': search_radius,
            'min_area': min_area,
            'max_mal': max_mal,
            'scale': scale,
            'crop_size': crop_size,
            'min_x_dist': min_x_dist,
            'min_y_dist': min_y_dist}


def make_train_val_split(img_idx_list:list, b_img_idx_list:list):
    """ Split generated training data crops into training and validation set.

    - param img_idx_list: List of image indices/paths (list of Pathlib Path objects).
        - type img_idx_list: list
    - param b_img_idx_list: List of image indices/paths which were classified as 'B' (list of Pathlib Path objects).
        - type b_img_idx_list: list

    - return: dict with ids for training and ids for validation.
    """

    img_ids_stem = []
    for idx in img_idx_list:
        img_ids_stem.append(idx.stem.split('img_')[-1])
    # Random 80%/20% split
    shuffle(img_ids_stem)
    train_ids = img_ids_stem[0:int(np.floor(0.8 * len(img_ids_stem)))]
    val_ids = img_ids_stem[int(np.floor(0.8 * len(img_ids_stem))):]
    # Add "B" quality only to train
    for idx in b_img_idx_list:
        train_ids.append(idx.stem.split('img_')[-1])
    # Train/val split
    train_val_ids = {'train': train_ids, 'val': val_ids}

    return train_val_ids


def make_train_dirs(path: Path):
    """ 
    Makes directories for: 'A', 'B', 'train' and 'val' folders, to save the created training data into.

    Args:
    - path: Path to the created training data sets.

    Return: None
    """
    Path.mkdir(path / 'A', parents=True, exist_ok=True)  # for high quality crops
    Path.mkdir(path / 'B', exist_ok=True)  # for good quality crops
    Path.mkdir(path / 'train', exist_ok=True)
    Path.mkdir(path / 'val', exist_ok=True)


def write_file(file, path):
    '''
    Function that writes the json file for the training configs of the net.

     Params:
    - path: Path to the info.json file.
    - file: Dictionary containting the training info.

    Return: None
    '''
    with open(path, 'w', encoding='utf8') as f:
        json.dump(file, f, ensure_ascii=False, indent=2)
    return


def create_ctc_training_sets(path_data: str, 
                             cell_type_list, 
                             split='01+02', 
                             crop_size=320,
                             ):
    ''' Main function in the data partition step for the training process. Consists of splitting data into the training and validation folders, 
        besides the Circular Hough-Transform operation, as the training excercise.

    Params:
    ---
    - path_data: Path of the parent cell type folder, where the trainin set will be generated.
    - cell_type_list: List of cell types to train the model with. When several cell types are used, a new name of the training is built. 
    - split: Use a single ('01'/'02) or both subsets ('01+02')

    Return: None.
    '''
    if len(cell_type_list) > 1: #This is done to shorten the name of the training list of images, into a new one.
        trainset_name = hashlib.sha1(str(cell_type_list).encode("UTF-8")).hexdigest()[:10]
        print('Multiple cell types, dataset name: {}'.format(trainset_name))

    else:
        trainset_name = cell_type_list

    for cell_type in trainset_name:

        path_trainset = path_data / "{}_{}".format(cell_type, split)
        if len(list((path_trainset / 'train').glob('*.tif'))) > 0:
            print('   ... training set {} already exists ...'.format(path_trainset.stem))
            continue
        print('   ... create {} training set ...'.format(path_trainset.stem))
        make_train_dirs(path=path_trainset)

        used_crops = []
        # Get ids of segmentation ground truth masks (GTs may not be fully annotated and STs may be erroneous)
        mask_ids, _ = get_mask_ids(path_data=path_data, ct=cell_type, split=split)

        # Getting the settings for the Hough Transform:
        td_settings = get_td_settings(mask_id_list=mask_ids, crop_size=crop_size)
        td_settings['used_crops'], td_settings['cell_type'] = used_crops, cell_type

        # Iterate through files and load images and masks:
        for mask_id in mask_ids:

             # Load images and masks (get slice and frame first)
            if len(mask_id.stem.split('_')) > 2:  # only slice annotated
                frame = mask_id.stem.split('_')[2]
                slice_idx = int(mask_id.stem.split('_')[3])
            else:
                frame = mask_id.stem.split('man_seg')[-1]
        
            # Load image and mask and get subset from which they are
            mask = tiff.imread(str(mask_id))
            subset = mask_id.parents[1].stem.split('_')[0]
            img = tiff.imread(str(mask_id.parents[2] / subset / "t{}.tif".format(frame)))

            # Loading the tra goldtruth:
            tra_gt = tiff.imread(str(mask_id.parents[1] / 'TRA' / "man_track{}.tif".format(frame)))

            # Downsampling
            if td_settings['scale'] != 1:
                img = downscale(img=img, scale=td_settings['scale'], order=2)
                mask = downscale(img=mask, scale=td_settings['scale'], order=0, aa=False)
                tra_gt = downscale(img=tra_gt, scale=td_settings['scale'], order=0, aa=False)

            # Normalization: min-max normalize image to [0, 65535]
            img = 2**16 * (img.astype(np.float16) - img.min()) / (img.max() - img.min())
            img = np.clip(img, 0, 2**16)

            generate_data(img=img, mask=mask, tra_gt=tra_gt, td_settings=td_settings, cell_type=cell_type,
                          subset=subset, frame=frame, path=path_trainset)
            
        td_settings['used_crops'].pop
        write_file(td_settings, path_trainset / 'info.json')

        # Create train/val split
        img_ids, b_img_ids = sorted((path_trainset / 'A').glob('img*.tif')), []
        if len(img_ids) <= 30:  # Use also "B" quality images when too few "A" quality images are available
            b_img_ids = sorted((path_trainset / 'B').glob('img*.tif'))
        
        train_val_ids = make_train_val_split(img_ids, b_img_ids)

        # Copy images to train/val folders
        for train_mode in ['train', 'val']:
            for idx in train_val_ids[train_mode]:
                if (path_trainset / "A" / ("img_{}.tif".format(idx))).exists():
                    source_path = path_trainset / "A"
                else:
                    source_path = path_trainset / "B"
                copy_train_data(source_path, path_trainset / train_mode, idx)

class CellSegDataset(Dataset):

    """ Pytorch data set for instance cell nuclei segmentation """

    def __init__(self, root_dir, mode='train', transform=lambda x: x):
        """
        Pytorch data set for instance cell nuclei segmentation

        - param root_dir: Directory containing all created training/validation data sets.
            - type root_dir: pathlib Path object.
        - param mode: 'train' or 'val'.
            - type mode: str
        - param transform: transforms.
            - type transform:
        - return: Dict (image, cell_label, border_label, id).
        """
        self.img_ids = sorted((root_dir / mode).glob('img*.tif')) # Root for the training image dataset / (train or val) / ###.tif
        self.mode = mode # Working mode of the net
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):

        img_id = self.img_ids[idx]
        dist_label_id = img_id.parent / ('dist_cell' + img_id.name.split('img')[-1])
        hough_transform_label_id = img_id.parent / ('hough_transform' + img_id.name.split('img')[-1])

        img = tiff.imread(str(img_id))
        dist_label = tiff.imread(str(dist_label_id)).astype(np.float32)
        hough_transform_label = tiff.imread(str(hough_transform_label_id)).astype(np.float32)

        if img.shape and dist_label.shape and hough_transform_label.shape:
            assert f"The shapes doesn't match {img.shape}_{dist_label.shape}_{hough_transform_label.shape}"
        
        sample = {'image': img, 
                  'cell_label': dist_label, 
                  'hough_transform_label': hough_transform_label, 
                  'id': img_id.stem
                  }

        sample = self.transform(sample)

        return sample
