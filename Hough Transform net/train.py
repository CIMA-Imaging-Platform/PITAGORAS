import argparse
import hashlib
import numpy as np
import random
import torch
import warnings

from pathlib import Path

from segmentation.utils import utils, unets
from segmentation.training.transforms import augmentors
from segmentation.training.training import train, get_max_epochs, get_weights
from segmentation.training.create_training_sets import create_ctc_training_sets, CellSegDataset




warnings.filterwarnings("ignore", category=UserWarning)

def main():

    random.seed()
    np.random.seed()

    # Get arguments
    parser = argparse.ArgumentParser(description='Cell Segmentation - Training')
    parser.add_argument('--act_fun', '-af', default='relu', type=str, help='Activation function')
    parser.add_argument('--architecture', '-ac', default='DU', type=str, help='Type of net can be the double U-NET or the single U-Net')
    parser.add_argument('--batch_size', '-bs', default=8, type=int, help='Batch size')
    parser.add_argument('--cell_type', '-ct', nargs='+', required=True, help='Cell type(s)')
    parser.add_argument('--filters', '-f', nargs=2, type=int, default=[64, 1024], help='Filters for U-net')
    parser.add_argument('--iterations', '-i', default=1, type=int, help='Number of models to train')
    parser.add_argument('--loss', '-l', default='smooth_l1', type=str, help='Loss function')
    parser.add_argument('--multi_gpu', '-mgpu', default=True, action='store_true', help='Use multiple GPUs')
    parser.add_argument('--norm_method', '-nm', default='bn', type=str, help='Normalization method')
    parser.add_argument('--optimizer', '-o', default='adam', type=str, help='Optimizer')
    parser.add_argument('--pool_method', '-pm', default='conv', type=str, help='Pool method')
    parser.add_argument('--split', '-s', default='01+02', type=str, help='Train/val split')
    args = parser.parse_args()

    # Paths
    path_models = Path(__file__).parent / 'models' / 'all'
    path_data = Path(__file__).parent / 'training_data'

    # Set device for using CPU or GPU
    device, num_gpus = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 1
    if str(device) == 'cuda':
        torch.backends.cudnn.benchmark = True
    if args.multi_gpu:
        num_gpus = torch.cuda.device_count()

    # Check if dataset consists in training_data folder
    if len(args.cell_type) > 1:
        es = 0
        for cell_type in args.cell_type:
            if not (path_data / cell_type).exists():
                print('No data for cell type "{}" found in {}'.format(cell_type, path_data))
                es = 1
        if es == 1:
            return
        trainset_name = hashlib.sha1(str(args.cell_type).encode("UTF-8")).hexdigest()[:10]
    else:
        if not (args.cell_type[0] == 'all') and not (path_data / args.cell_type[0]).exists():
            print('No data for cell type "{}" found in {}'.format(args.cell_type[0], path_data))
            return
        trainset_name = args.cell_type[0]

    # Create training sets
    print('Create training sets for {} ...'.format(args.cell_type))
    create_ctc_training_sets(path_data=path_data, cell_type_list=args.cell_type, split=args.split)

    # Get model names and how many iterations/models need to be trained:
    model_name = '{}_{}_model'.format(trainset_name, args.split)

    # Train multiple models
    for i in range(args.iterations):
        
        # Define the path where the model is going to be stored
        run_name = utils.unique_path(path_models, model_name)

        # Set up the double U-Net encoder:
        train_configs = {'architecture': (args.architecture, args.act_fun, args.pool_method, args.norm_method, args.filters),
                         'batch_size': args.batch_size,
                         'batch_size_auto': 2,
                         'label_type': "distance",
                         'loss': args.loss,
                         'num_gpus': num_gpus,
                         'optimizer': args.optimizer,
                         'run_name': run_name
                         }
        net = unets.build_unet(unet_type= train_configs['architecture'][0], 
                               act_fun= train_configs['architecture'][1],
                               pool_method= train_configs['architecture'][2],
                               normalization= train_configs['architecture'][3],
                               device= device,
                               num_gpus= num_gpus,
                               ch_in= 1,
                               ch_out= 1,
                               filters= train_configs['architecture'][4]
                               )

        # Load training and validation set
        data_transforms = augmentors(label_type=train_configs['label_type'])
        train_configs['data_transforms'] = str(data_transforms)
        dataset_name = "{}_{}".format(trainset_name, args.split)
        datasets = {x: CellSegDataset(root_dir=path_data / dataset_name, mode=x, transform=data_transforms[x])
                    for x in ['train', 'val']}
        
        # Get number of training epochs depending on dataset size (just roughly to decrease training time):
        train_configs['max_epochs'] = get_max_epochs(len(datasets['train']) + len(datasets['val']))

         # Train model
        best_loss = train(net=net, datasets=datasets, configs=train_configs, device=device, path_models=path_models)

        if train_configs['optimizer'] == 'ranger':
            net = unets.build_unet(unet_type=train_configs['architecture'][0],
                                   act_fun=train_configs['architecture'][2],
                                   pool_method=train_configs['architecture'][1],
                                   normalization=train_configs['architecture'][3],
                                   device=device,
                                   num_gpus=num_gpus,
                                   ch_in=1,
                                   ch_out=1,
                                   filters=train_configs['architecture'][4])
            # Get best weights as starting point
            net = get_weights(net=net, weights=str(path_models / '{}.pth'.format(run_name)), num_gpus=num_gpus, device=device)
            # Train further
            _ = train(net=net, datasets=datasets, configs=train_configs, device=device, path_models=path_models, best_loss=best_loss)

        # Write information to json-file
        utils.write_train_info(configs=train_configs, path=path_models)

if __name__ == "__main__":
    
    main()
