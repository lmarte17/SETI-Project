import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class SETIPreprocessor:
    """
    A class for preprocessing SETI signal data, including image processing,
    Fourier transforms, and dimensionality reduction.
    """
    
    def __init__(self, target_size=(256, 256)):
        """
        Initialize the preprocessor with target image dimensions.
        
        Args:
            target_size (tuple): Desired dimensions for resizing images (height, width).
        """
        self.target_size = target_size
        self.pca = None
        self.image_paths = None
        self.signal_types = None

    # New method to calculate profiles
    def calculate_profiles(self, image):
        """
        Calculate vertical and horizontal intensity profiles of an image.
        Args:
            image (np.array): 2D array of a single processed image.
        Returns:
            vertical_profile (np.array): Mean intensity for each row.
            horizontal_profile (np.array): Mean intensity for each column.
        """
        vertical_profile = np.mean(image, axis=1)
        horizontal_profile = np.mean(image, axis=0)
        
        # Compute gradients
        vertical_gradient = np.gradient(vertical_profile)
        horizontal_gradient = np.gradient(horizontal_profile)

        # Normalize gradients
        vertical_profile = vertical_gradient / np.max(np.abs(vertical_gradient))
        horizontal_profile = horizontal_gradient / np.max(np.abs(horizontal_gradient))
        
        return vertical_profile, horizontal_profile
    
    def load_and_preprocess_images(self, data_dir: str, return_2d: bool = False):
        """
        Load and preprocess signal images with advanced filtering techniques.
        
        Args:
            data_dir (str): Path to the directory containing images.
            return_2d (bool): Whether to return images in 2D format.
            
        Returns:
            np.ndarray: Processed image data array.
            np.ndarray: Array of signal categories.
            list: List of image file paths.
        """
        self.image_paths = list(Path(data_dir).rglob('*.png'))
        print(f"Found {len(self.image_paths)} images.")
        
        X = []
        self.signal_types = []
        profiles = []
        
        for path in tqdm(self.image_paths, desc="Processing images"):
            # Load and convert to grayscale
            img_color = cv2.imread(str(path), cv2.IMREAD_COLOR)
            img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            
            # Statistical filtering
            std = np.std(img_gray)
            mean = np.mean(img_gray)
            img_clipped = np.clip(img_gray, mean + (2.5*std), mean + (5*std))
            
            # Apply Gaussian blur
            gaussian = cv2.GaussianBlur(img_clipped, (3, 3), 5)
            
            # Morphological operations
            kernel3 = np.ones((3, 3), dtype=np.float32)
            kernel5 = np.ones((5, 5), dtype=np.float32)
            morphed = cv2.morphologyEx(gaussian, cv2.MORPH_OPEN, kernel=kernel3)
            morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel=kernel5)
            
            # Edge detection
            sobelx = cv2.Sobel(morphed, cv2.CV_64F, 1, 0, 10)
            sobely = cv2.Sobel(morphed, cv2.CV_64F, 0, 1, 10)
            blended = cv2.addWeighted(src1=sobelx, alpha=0.9, 
                                    src2=sobely, beta=0.3, gamma=0.25)
            
            # Final clipping
            clipped = np.clip(blended, 
                            np.mean(blended) + 2.5*np.std(blended),
                            np.mean(blended) + 5*np.std(blended))
            
            # Resize
            img_resized = cv2.resize(clipped, self.target_size)
            
            # Store processed image
            X.append(img_resized if return_2d else img_resized.ravel())
            self.signal_types.append(path.parent.name)

            # Calculate intensity profiles
            vertical_profile, horizontal_profile = self.calculate_profiles(img_resized)
            profiles.append((vertical_profile, horizontal_profile))
        
        return np.array(X), np.array(self.signal_types), self.image_paths, profiles
    
    def apply_pca(self, X_scaled: np.ndarray, n_components: float = 0.95):
        """
        Apply PCA to the preprocessed image data.
        
        Args:
            X_scaled: Scaled flattened image data.
            n_components: Number of components or variance ratio to retain.
            
        Returns:
            np.ndarray: PCA-transformed data.
            PCA: Fitted PCA object.
        """
        n_samples, n_features = X_scaled.shape
        height, width = self.target_size

        # Check if feature size matches image size
        # assert n_features == height * width, "Feature size must match image size."
        
        print(f"Applying PCA with {n_components} components or variance ratio.")
        
        # Determine if n_components is an integer or float
        if isinstance(n_components, int):
            self.pca = PCA(n_components=n_components)
        elif isinstance(n_components, float):
            self.pca = PCA(n_components=n_components)
        else:
            raise ValueError("n_components must be an integer or a float.")
        
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Visualization
        plt.figure(figsize=(10, 5))
        plt.plot(np.cumsum(self.pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('PCA Explained Variance Ratio')
        plt.grid(True)
        plt.show()
        
        # Print variance thresholds
        # for threshold in [0.9, 0.95]:
        #     n_components_threshold = np.argmax(
        #         np.cumsum(self.pca.explained_variance_ratio_) >= threshold) + 1
        #     print(f"Components needed for {int(threshold * 100)}% variance: {n_components_threshold}")
        
        return X_pca, self.pca
    
    def apply_fourier_transform(self, X: np.ndarray, method: str = 'standard', target_size: tuple = None):
        """
        Apply Fourier Transform to the image data.
        
        Args:
            X: Array of flattened images.
            method: Type of Fourier transform ('standard', 'filtered', or 'hybrid').
            target_size: Desired dimensions for reshaping images (height, width).
                Defaults to the instance's target_size if not specified.
        
        Returns:
            np.ndarray: Fourier-transformed features.
        """
        n_samples, n_pixels = X.shape
        
        # Use the provided target_size or default to self.target_size
        if target_size is None:
            target_size = self.target_size
        height, width = target_size
        
        assert n_pixels == height * width, "Input shape does not match target size."
        
        X_reshaped = X.reshape(n_samples, height, width)
        
        if method == 'standard':
            return self._standard_fourier(X_reshaped)
        elif method == 'filtered':
            return self._filtered_fourier(X_reshaped)
        elif method == 'hybrid':
            return self._hybrid_fourier(X_reshaped)
        else:
            raise ValueError("Method must be 'standard', 'filtered', or 'hybrid'")
    
    def _standard_fourier(self, X_reshaped: np.ndarray) -> np.ndarray:
        """Apply standard 2D Fourier Transform."""
        features = []
        for img in tqdm(X_reshaped, desc="Applying Standard FFT"):
            fft = np.fft.fft2(img)
            fft_shift = np.fft.fftshift(fft)
            magnitude_spectrum = np.abs(fft_shift)
            features.append(magnitude_spectrum.ravel())
        return np.array(features)
    
    def _filtered_fourier(self, X_reshaped: np.ndarray, radius: int = 10) -> np.ndarray:
        """Apply frequency-filtered Fourier Transform."""
        features = []
        rows, cols = self.target_size
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols), np.uint8)
        cv2.circle(mask, (ccol, crow), radius, 1, thickness=-1)
        
        for img in tqdm(X_reshaped, desc="Applying Filtered FFT"):
            fft = np.fft.fft2(img)
            fshift = np.fft.fftshift(fft)
            fshift_filtered = fshift * mask
            img_filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_filtered)))
            features.append(img_filtered.ravel())
        return np.array(features)
    
    def _hybrid_fourier(self, X_reshaped: np.ndarray, radius: int = 10) -> np.ndarray:
        """Apply hybrid Fourier Transform combining magnitude spectrum and filtered reconstruction."""
        magnitude_features = []
        filtered_features = []
        
        rows, cols = self.target_size
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols), np.uint8)
        cv2.circle(mask, (ccol, crow), radius, 1, thickness=-1)
        
        for img in tqdm(X_reshaped, desc="Applying Hybrid FFT"):
            fft = np.fft.fft2(img)
            fft_shift = np.fft.fftshift(fft)
            
            # Magnitude spectrum
            magnitude_spectrum = np.abs(fft_shift)
            magnitude_features.append(magnitude_spectrum.ravel())
            
            # Filtered reconstruction
            fshift_filtered = fft_shift * mask
            img_filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_filtered)))
            filtered_features.append(img_filtered.ravel())
        
        return np.hstack([np.array(magnitude_features), np.array(filtered_features)])