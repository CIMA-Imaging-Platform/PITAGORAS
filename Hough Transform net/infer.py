import argparse
import numpy as np
from pathlib import Path
import random
import torch
import warnings

from copy import deepcopy
from segmentation.inference.inference import inference
'''
Script to execute for infering the EDT and the Hough Transform images
'''
# Line for supressing the warnings coming up from the usercode:
warnings.filterwarnings("ignore", category=UserWarning)

def main():

    random.seed()
    np.random.seed()

    # Get the arguments for infering:
    parser = argparse.ArgumentParser(description='TH CIMA algortihm - Inference')
    # Crucial input:
    parser.add_argument('--cell_type', '-ct', required=True, nargs='+', help='Cell type(s) to predict')
    parser.add_argument('--model', '-m', required=True, type=str, help='Name of the net model to use')
    parser.add_argument('--subset', '-s', default='01+02', type=str, help='Subset to evaluate on')
    # Tunable parameters:
    parser.add_argument('--apply_clahe', '-ac', default= False, type=bool, help='Apply Local Contrast Enhancement')    
    parser.add_argument('--batch_size', '-bs', default=8, type=int, help='Batch size')
    parser.add_argument('--save_raw_pred', '-srp', default=False, action='store_true', help='Save some Hough Transform and EDT predictions')
    parser.add_argument('--scale', '-sc', default=1, type=float, help='Scale factor')
    args = parser.parse_args()

    # Paths of the data and the trained model:
    path_data = Path(__file__).parent / 'infer_data'
    path_models = Path(__file__).parent / 'models' / 'all'

    # Set the device to run on CPU or GPU if posible:
    # With CPU gpus = 1 and if GPU the number of devices avaliable.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        # Decides automatically which internl strategy use for inference:
        torch.backends.cudnn.benchmark = True
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = 1

    # Work on each cell_type separately
    for ct in args.cell_type:

        # Check if the dataset with the cell_type is in the folder:
        if not (path_data / ct).exists():
            print('No data for "{}" found in {}'.format(ct, path_data))
            return # As I want to stop the hole script in case there was no data
        
        else:
            # Load the trained model:
            model = path_models / args.model
            model = model.parent / f"{model.stem}.pth"

            subsets = [args.subset]
            for subset in subsets:
                path_seg_results = path_data / ct / "{}_RES_{}".format(subset, model.stem)
                path_seg_results.mkdir(exist_ok=True)
                print(f'Inference using {model.stem} on {ct}_{subset}. Params: scale {args.scale}')
                # Check if results already exist
                if len(sorted(path_seg_results.glob('*.tif'))) > 0:
                    print('Segmentation results already exist (delete for new calculation).')
                    continue
                
                # Copy the input args and remove the [] from the args.cell_type
                inference_args = deepcopy(args)
                inference_args.cell_type = ct

                inference(model=model,
                                data_path=path_data / ct / subset,
                                result_path=path_seg_results,
                                device=device,
                                batchsize=args.batch_size,
                                args=inference_args,
                                num_gpus=num_gpus)

    print('End of the algoritm')

if __name__ == "__main__":

    main()