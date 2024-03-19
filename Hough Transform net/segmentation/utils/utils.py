import json
import numpy as np
import shutil

def unique_path(directory: str, name_pattern: str):
    """ Get unique file name to save trained model.

    - directory: Path to the model directory
        - type: pathlib path object.
    - name_pattern: Pattern for the file name
        - type: str
    - return: pathlib path
    """
    counter = 0
    while True:
        counter += 1
        path = directory / name_pattern.format(counter)
        if not path.exists():
            return str(path)
        
def min_max_normalization(img, min_value=None, max_value=None):
    """ 
    Minimum maximum normalization to [-1, 1]

    Param:

    - param img: Image (uint8, uint16 or int)
        - type img:
    - param min_value: minimum value for normalization, values below are clipped.
        - type min_value: int
    - param max_value: maximum value for normalization, values above are clipped.
        - type max_value: int
    - return: Normalized image (float32)
    """

    if max_value is None:
        max_value = img.max()

    if min_value is None:
        min_value = img.min()

    # Clip image to filter hot and cold pixels
    img = np.clip(img, min_value, max_value)

    # Apply min-max-normalization
    img = 2 * (img.astype(np.float32) - min_value) / (max_value - min_value) - 1

    return img.astype(np.float32)

def write_train_info(configs, path):
    """ Write training configurations into a json file.

    :param configs: Dictionary with configurations of the training process.
        :type configs: dict
    :param path: path to the directory to store the json file.
        :type path: pathlib Path object
    :return: None
    """

    with open(path / (configs['run_name'] + '.json'), 'w', encoding='utf-8') as outfile:
        json.dump(configs, outfile, ensure_ascii=False, indent=2)

    return None

def zero_pad_model_input(img: np.array, pad_val: int= 0):
    """ Zero-pad model input to get for the model needed sizes (more intelligent padding ways could easily be
        implemented but there are sometimes cudnn errors with image sizes which work on cpu ...).

    - param img: Model input image.
        - type:
    - param pad_val: Value to pad.
        - type pad_val: int.

    :return: (zero-)padded img, [0s padded in y-direction, 0s padded in x-direction]
    """

    # Tested shapes
    tested_img_shapes = [64, 128, 256, 320, 512, 768, 1024, 1280, 1408, 1600, 1920, 2048, 2240, 2560, 3200, 4096,
                         4480, 6080, 8192]

    # More effective padding (but may lead to cuda errors)
    # y_pads = int(np.ceil(img.shape[0] / 64) * 64) - img.shape[0]
    # x_pads = int(np.ceil(img.shape[1] / 64) * 64) - img.shape[1]

    # Get teh size of the pad for both directions:
    pads = []
    for i in range(2):
        for tested_img_shape in tested_img_shapes:
            if img.shape[i] <= tested_img_shape:
                pads.append(tested_img_shape - img.shape[i])
                break
    if not pads:
        raise Exception('Image too big to pad. Use sliding windows')

    # Do the padding:
    if len(img.shape) == 3:  # 3D image
        img = np.pad(img, ((pads[0], 0), (pads[1], 0), (0, 0)), mode='constant', constant_values=pad_val)
        img = np.transpose(img, (2, 1, 0))
    else:
        img = np.pad(img, ((pads[0], 0), (pads[1], 0)), mode='constant', constant_values=pad_val)
    
    return img, [pads[0], pads[1]]
