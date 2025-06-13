import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def calculate_metrics(image_true, image_test, data_range=None, win_size=None, channel_axis=None):
    """
    Calculates Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM)
    between two images.
    
    Args:
        image_true (np.ndarray): The ground truth image.
        image_test (np.ndarray): The image to be evaluated.
        data_range (float, optional): The data range of the input images. If None, it is
                                      estimated from the ground truth image. Defaults to None.
        win_size (int, optional): The side-length of the sliding window used in SSIM calculation.
                                  Must be an odd integer. Defaults to None.
        channel_axis (int or None, optional): If the image is multichannel, this indicates the
                                              axis of the channels.
    
    Returns:
        tuple: A tuple containing the PSNR and SSIM values.
    """
    # Ensure images are float for calculation to avoid data type issues.
    image_true_float = image_true.astype(np.float64)
    image_test_float = image_test.astype(np.float64)

    # If data_range is not provided, calculate it from the true image.
    if data_range is None:
        current_data_range = np.max(image_true_float) - np.min(image_true_float)
        # Handle the case of a completely flat image to avoid division by zero.
        if current_data_range == 0:
            current_data_range = 1.0 
    else:
        current_data_range = float(data_range)
        # The data_range for skimage metrics must be positive.
        if current_data_range <= 0:
            current_data_range = 1.0
            
    # Calculate PSNR.
    psnr = peak_signal_noise_ratio(image_true_float, image_test_float, data_range=current_data_range)
    
    # The 'multichannel' argument for ssim is deprecated in newer scikit-image versions.
    # Instead, 'channel_axis' is used to specify if the image is multichannel.
    multichannel_arg = channel_axis is not None
    
    # Calculate SSIM.
    ssim = structural_similarity(image_true_float, image_test_float, 
                                 data_range=current_data_range, 
                                 win_size=win_size,
                                 channel_axis=channel_axis)
    return psnr, ssim