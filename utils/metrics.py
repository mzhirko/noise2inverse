import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def calculate_metrics(image_true, image_test, data_range=None, win_size=None, channel_axis=None):
    image_true_float = image_true.astype(np.float64)
    image_test_float = image_test.astype(np.float64)

    if data_range is None:
        current_data_range = np.max(image_true_float) - np.min(image_true_float)
        if current_data_range == 0: # Handles flat image case
            current_data_range = 1.0 
    else:
        current_data_range = float(data_range)
        if current_data_range <= 0: # data_range must be positive for skimage metrics
            # Defaulting to 1.0 if user provides an invalid (e.g. zero or negative) data_range.
            # Skimage would error on negative.
            current_data_range = 1.0
            
    psnr = peak_signal_noise_ratio(image_true_float, image_test_float, data_range=current_data_range)
    
    multichannel_arg = channel_axis is not None
    
    ssim = structural_similarity(image_true_float, image_test_float, 
                                 data_range=current_data_range, 
                                 win_size=win_size,
                                 multichannel=multichannel_arg,
                                 channel_axis=channel_axis
                                 )
    return psnr, ssim