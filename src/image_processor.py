import numpy as np
import cv2
from pathlib import Path
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
from typing import Dict, Any, List, Union, Tuple
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ImageMetadata:
    """Data class to store image metadata"""
    signal_type: str
    file_name: str
    file_path: str

class ImageProcessor:
    """Class for processing and analyzing signal data from images."""
    
    def __init__(self, blur_kernel_size: Tuple[int, int] = (3, 3), 
                 blur_sigma: float = 5.0,
                 morph_kernel3_size: Tuple[int, int] = (3, 3),
                 morph_kernel5_size: Tuple[int, int] = (5, 5),
                 sobel_kernel_size: int = 10):
        """
        Initialize SignalProcessor with configurable parameters.
        
        Args:
            blur_kernel_size: Tuple for Gaussian blur kernel size
            blur_sigma: Sigma value for Gaussian blur
            morph_kernel3_size: Size for morphological operations (3x3)
            morph_kernel5_size: Size for morphological operations (5x5)
            sobel_kernel_size: Kernel size for Sobel operations
        """
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.kernel3 = np.ones(morph_kernel3_size, dtype=np.float32)
        self.kernel5 = np.ones(morph_kernel5_size, dtype=np.float32)
        self.sobel_kernel_size = sobel_kernel_size
        
    @staticmethod
    def sigmoid_normalize(x: np.ndarray) -> np.ndarray:
        """
        Apply sigmoid normalization to the input array.
        
        Args:
            x: Input array to normalize
            
        Returns:
            Normalized array using sigmoid function
        """
        x_centered = x - np.mean(x)
        x_scaled = x_centered / (3 * np.std(x_centered) + 1e-10)
        return 1 / (1 + np.exp(-x_scaled))
    
    def _extract_profile_features(self, profile: np.ndarray) -> Dict[str, float]:
        """
        Extract statistical features from a profile.
        
        Args:
            profile: Input signal profile
            
        Returns:
            Dictionary of statistical features
        """
        return {
            "mean": np.mean(profile),
            "std_dev": np.std(profile),
            "min": np.min(profile),
            "max": np.max(profile),
            "skewness": skew(profile),
            "kurtosis": kurtosis(profile)
        }
    
    def _extract_frequency_features(self, profile: np.ndarray) -> Dict[str, float]:
        """
        Extract frequency domain features from a profile.
        
        Args:
            profile: Input signal profile
            
        Returns:
            Dictionary of frequency features
        """
        fft = np.fft.fft(profile)
        magnitude = np.abs(fft)
        return {
            "freq_mean": np.mean(magnitude),
            "freq_max": np.max(magnitude),
            "freq_std": np.std(magnitude)
        }
    
    def _extract_gradient_features(self, gradient: np.ndarray) -> Dict[str, float]:
        """
        Extract features from gradient data.
        
        Args:
            gradient: Input gradient array
            
        Returns:
            Dictionary of gradient features
        """
        return {
            "gradient_mean": np.mean(gradient),
            "gradient_std": np.std(gradient),
            "gradient_max": np.max(gradient),
            "gradient_min": np.min(gradient)
        }
    
    def _extract_peak_features(self, profile: np.ndarray) -> Dict[str, float]:
        """
        Extract features related to signal peaks.
        
        Args:
            profile: Input signal profile
            
        Returns:
            Dictionary of peak-related features
        """
        peaks, _ = find_peaks(profile)
        return {
            "num_peaks": len(peaks),
            "peak_mean_height": np.mean(profile[peaks]) if len(peaks) > 0 else 0,
            "peak_max_height": np.max(profile[peaks]) if len(peaks) > 0 else 0
        }
    
    def _extract_fft_features(self, profile: np.ndarray, N: int = 20, prefix: str = "") -> Dict[str, float]:
        """
        Extract FFT features from a profile.
        
        Args:
            profile: Input signal profile
            N: Number of FFT components to use as features
            prefix: Prefix for feature names
            
        Returns:
            Dictionary of FFT features
        """
        # Apply FFT and get magnitude
        fft = np.fft.fft(profile)
        magnitude = np.abs(fft)
        
        # Take first N components
        features = {}
        for i in range(min(N, len(magnitude))):
            features[f"{prefix}fft_{i}"] = magnitude[i]
            
        return features
    
    def extract_all_features(self, profile: np.ndarray, prefix: str = "") -> Dict[str, float]:
        """
        Extract all features from a profile.
        
        Args:
            profile: Input signal profile
            prefix: Prefix for feature names (e.g., 'vertical_' or 'horizontal_')
            
        Returns:
            Dictionary containing all extracted features
        """
        # Calculate gradient
        gradient = np.gradient(profile)
        
        # Extract all feature types
        features = {
            **self._extract_profile_features(profile),
            **self._extract_frequency_features(profile),
            **self._extract_gradient_features(gradient),
            **self._extract_peak_features(profile),
            **self._extract_fft_features(profile, prefix=prefix)  # Add FFT features
        }
        
        # Add prefix to feature names if specified
        if prefix:
            # Don't add prefix to FFT features as they already have it
            non_fft_features = {k: v for k, v in features.items() if not k.startswith(f"{prefix}fft_")}
            fft_features = {k: v for k, v in features.items() if k.startswith(f"{prefix}fft_")}
            features = {**{f"{prefix}{k}": v for k, v in non_fft_features.items()}, **fft_features}
            
        return features

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image using various image processing techniques.
        
        Args:
            image: Input image array
            
        Returns:
            Processed image array
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply statistical clipping
        std = np.std(gray)
        mean = np.mean(gray)
        img_clipped = np.clip(gray, mean + (3.25*std), mean + (6*std))
        
        # Apply Gaussian blur
        gaussian = cv2.GaussianBlur(img_clipped, self.blur_kernel_size, self.blur_sigma)
        
        # Apply morphological operations
        morphed = cv2.morphologyEx(gaussian, cv2.MORPH_OPEN, kernel=self.kernel3)
        morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel=self.kernel5)
        
        # Apply Sobel operators
        sobelx = cv2.Sobel(morphed, cv2.CV_64F, 1, 0, self.sobel_kernel_size)
        sobely = cv2.Sobel(morphed, cv2.CV_64F, 0, 1, self.sobel_kernel_size)
        
        # Combine Sobel results
        processed = cv2.addWeighted(src1=sobelx, alpha=0.9, src2=sobely, beta=0.3, gamma=0.25)
        
        # Final clipping
        processed = np.clip(processed, 
                          np.mean(processed) + 3.75*np.std(processed),
                          np.mean(processed) + 7*np.std(processed))
        
        return processed
    
    def process_profiles(self, processed_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process vertical and horizontal profiles from the processed image.
        
        Args:
            processed_image: Preprocessed image array
            
        Returns:
            Tuple of (vertical_profile, horizontal_profile)
        """
        # Calculate mean profiles
        vertical_profile = np.mean(processed_image, axis=1)
        horizontal_profile = np.mean(processed_image, axis=0)
        
        # Calculate and normalize gradients
        vertical_gradient = np.gradient(vertical_profile)
        horizontal_gradient = np.gradient(horizontal_profile)
        
        vertical_profile = self.sigmoid_normalize(vertical_gradient)
        horizontal_profile = self.sigmoid_normalize(horizontal_gradient)
        
        # Apply exponential emphasis
        vertical_profile = vertical_profile ** 2 * np.sign(vertical_profile)
        horizontal_profile = horizontal_profile ** 2 * np.sign(horizontal_profile)
        
        # Final normalization
        vertical_profile = self.sigmoid_normalize(vertical_profile)
        horizontal_profile = self.sigmoid_normalize(horizontal_profile)
        
        return vertical_profile, horizontal_profile
    
    def process_image(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process a single image and extract all features.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing metadata and extracted features
        """
        try:
            # Convert path to Path object if necessary
            image_path = Path(image_path)
            
            # Create metadata
            metadata = ImageMetadata(
                signal_type=image_path.parent.name,
                file_name=image_path.name,
                file_path=str(image_path)
            )
            
            # Load and process image
            original_img = cv2.imread(str(image_path))
            if original_img is None:
                raise ValueError(f"Failed to load image: {image_path}")
                
            processed = self.preprocess_image(original_img)
            vertical_profile, horizontal_profile = self.process_profiles(processed)
            
            # Extract features
            vertical_features = self.extract_all_features(vertical_profile, prefix="vertical_")
            horizontal_features = self.extract_all_features(horizontal_profile, prefix="horizontal_")
            
            # Combine all features with metadata
            return {
                **metadata.__dict__,
                **vertical_features,
                **horizontal_features
            }
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            raise

    def batch_process(self, image_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
        """
        Process multiple images in batch.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            List of dictionaries containing metadata and features for each image
        """
        all_features = []
        
        for image_path in image_paths:
            try:
                features = self.process_image(image_path)
                all_features.append(features)
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {str(e)}")
                continue
                
        return all_features