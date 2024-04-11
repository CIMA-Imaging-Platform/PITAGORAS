import argparse as args
import gc
import json
import numpy as np
import tifffile as tiff
import torch
import pathlib as Path

from multiprocessing import cpu_count
from skimage.transform import resize
from segmentation.inference.dataset_loader import Data, pre_processing_transforms
from segmentation.inference.postprocessing import postprocessing
from segmentation.utils.unets import build_unet

def inference(model: Path,
                    data_path: Path,
                    result_path: Path,
                    device: torch.device,
                    batchsize: int,
                    args:args.Namespace,
                    num_gpus: int):
    """ Inference function for 2D data sets.

    Parameters:
    ---
    - param model: Path to the model to use for inference.
        - type model: pathlib Path object.
    - param data_path: Path to the directory containing the data sets.
        - type data_path: pathlib Path object
    - param result_path: Path to the results directory.
        - type result_path: pathlib Path object
    - param device: Use (multiple) GPUs or CPU.
        - type device: torch device
    - param batchsize: Batch size.
        - type batchsize: int
    - param args: Arguments for post-processing.
        - type args:
    - param num_gpus: Number of GPUs to use in GPU mode (enables larger batches)
        - type num_gpus: int
    
    Return: 
    ---
    None
    """
    # Load model json file to get architecture + filters of the D-Unet:
    with open(model.parent / (model.stem + '.json')) as f:
        model_settings = json.load(f)

    # Build the model with the specifications from model_settings:
    net = build_unet(unet_type= model_settings['architecture'][0],
                    act_fun= model_settings['architecture'][1],
                    pool_method= model_settings['architecture'][2],
                    normalization= model_settings['architecture'][3],
                    filters= model_settings['architecture'][4],
                    device= device,
                    num_gpus= num_gpus,
                    ch_in= 1,
                    ch_out= 1)
    
    # Load the weights depending on the number of GPUs:
    if num_gpus > 1:
        net.module.load_state_dict(torch.load(str(model), map_location=device))
    else:
        net.load_state_dict(torch.load(str(model), map_location=device))
        
    # Set the net to inference mode:
    net.eval()
    torch.set_grad_enabled(False) # Set the gradient calculation to off.

    # Get images to predict:
    dataset = Data(data_dir= data_path,
                    transform= pre_processing_transforms(apply_clahe= args.apply_clahe, 
                                                         scale_factor= args.scale))
    
    # Get the number of workers: which is a multi-process data loading operation 
    if device.type == "cpu":
        num_workers = 0
    else:
        try:
            num_workers = cpu_count() // 2
        except AttributeError:
            num_workers = 4
    if num_workers <= 2:  # Probably Google Colab --> use 0
        num_workers = 0
    num_workers = np.minimum(num_workers, 16)

    # Load the data applying the tranforms on the dataset.
    dataloader = torch.utils.data.DataLoader(dataset, # Inhereted object from torch.utils.data.Dataset
                                             batch_size=batchsize, 
                                             shuffle=False, # False as we don't requiere a random order of the images
                                             pin_memory=True, # Sppeds up the location of data
                                             num_workers=num_workers)
    
    # Predict images over each batch (iterate over images/files)
    for sample in dataloader:
        
        img_batch, ids_batch, pad_batch, img_size = sample
        img_batch = img_batch.to(device)

        if batchsize > 1:  # all images in a batch have same dimensions and pads
            # Arrays containing the padding and img_size for each image.
            pad_batch = [pad_batch[i][0] for i in range(len(pad_batch))]
            img_size = [img_size[i][0] for i in range(len(img_size))]

        # Prediction
        prediction_hough_transform_batch, prediction_cell_batch = net(img_batch)

        # Get rid of pads
        prediction_cell_batch = prediction_cell_batch[:, 0, pad_batch[0]:, pad_batch[1]:, None].cpu().numpy() # Dimensions [nº_path, pseudo_color, x, y]
        prediction_hough_transform_batch = prediction_hough_transform_batch[:, 0, pad_batch[0]:, pad_batch[1]:, None].cpu().numpy()

        # Save also some raw predictions (not all since float32 --> needs lot of memory)
        save_ids = [0, len(dataset) // 8, len(dataset) // 4, 3 * len(dataset) // 8, len(dataset) // 2,
                    5 * len(dataset) // 8, 3 * len(dataset) // 4, 7 * len(dataset) // 8, len(dataset) - 1]

        # Go through predicted batch and apply post-processing (not parallelized)
        for h in range(len(prediction_cell_batch)):

            print('         ... processing infered {0} ...'.format(ids_batch[h]))

            # Get actual file number:
            file_num = int(ids_batch[h].split('t')[-1])

            # Save not all raw predictions to save memory
            if file_num in save_ids and args.save_raw_pred:
                save_raw_pred = True
            else:
                save_raw_pred = False
            
            file_id = f'{file_num:03d}.tif'

            # Apply postprocessing on the predected images
            result, prediction_hough_transform, prediction_cell = postprocessing(hough_transform_prediction=prediction_hough_transform_batch[h],
                                                                                 cell_prediction=prediction_cell_batch[h], 
                                                                                 args=args)
            if args.scale < 1:
                prediction_instance = resize(prediction_instance,
                                             img_size,
                                             order=0,
                                             preserve_range=True,
                                             anti_aliasing=False).astype(np.uint16)
            # Save predected images  
            tiff.imwrite(str(result_path / ('mask' + file_id)), result, compress = 1)
            tiff.imwrite(str(result_path / ('cell' + file_id)), np.squeeze(prediction_cell).astype(np.float32), compress=1)
            tiff.imwrite(str(result_path / ('hough_transform' + file_id)), np.squeeze(prediction_hough_transform).astype(np.float32), compress=1)

    # Clear memory
    del net
    gc.collect()

    return None