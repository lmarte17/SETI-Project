import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler
import logging
import platform

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for Apple Silicon
IS_APPLE_SILICON = platform.processor() == 'arm'
if IS_APPLE_SILICON:
    try:
        import torch
        USE_MPS = torch.backends.mps.is_available()
    except ImportError:
        USE_MPS = False
else:
    USE_MPS = False

class SETIDataLoader:
    """
    Class for loading and preprocessing SETI dataset.
    Optimized for Apple Silicon when available.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the data loader.
        
        Args:
            data_dir (str): Path to the directory containing SETI data files
        """
        self.data_dir = Path(data_dir)
        self.scaler = StandardScaler()
        if USE_MPS:
            logger.info("Using MPS acceleration for data processing")
            self.device = torch.device("mps")
        else:
            logger.info("Using CPU for data processing")
            self.device = torch.device("cpu")
        
    def load_data(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Load the SETI dataset from HDF5 files.
        
        Returns:
            Tuple containing:
                - numpy array of spectrogram data
                - numpy array of metadata (if available)
        """
        try:
            logger.info(f"Loading data from {self.data_dir}")
            data_files = list(self.data_dir.glob("*.h5"))
            
            if not data_files:
                raise FileNotFoundError(f"No HDF5 files found in {self.data_dir}")
            
            # Initialize lists to store data
            spectrograms = []
            metadata = []
            
            # Load data from each file
            for file_path in data_files:
                with h5py.File(file_path, 'r') as f:
                    if USE_MPS:
                        # Load directly to GPU memory if using MPS
                        spec_data = torch.from_numpy(np.array(f['spectrograms'])).to(self.device)
                        spectrograms.append(spec_data.cpu().numpy())  # Convert back to numpy for compatibility
                    else:
                        spectrograms.append(np.array(f['spectrograms']))
                    
                    if 'metadata' in f:
                        metadata.append(np.array(f['metadata']))
            
            # Combine data from all files
            combined_spectrograms = np.concatenate(spectrograms, axis=0)
            combined_metadata = np.concatenate(metadata, axis=0) if metadata else None
            
            logger.info(f"Loaded {combined_spectrograms.shape[0]} samples")
            return combined_spectrograms, combined_metadata
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess the spectrogram data.
        
        Args:
            data (np.ndarray): Raw spectrogram data
            
        Returns:
            np.ndarray: Preprocessed data
        """
        try:
            # Reshape data if needed (samples, height, width) -> (samples, features)
            if len(data.shape) > 2:
                original_shape = data.shape
                data = data.reshape(data.shape[0], -1)
            
            if USE_MPS:
                # Process on GPU if available
                data_tensor = torch.from_numpy(data).to(self.device)
                # Implement scaling on GPU
                mean = data_tensor.mean(dim=0)
                std = data_tensor.std(dim=0)
                scaled_data = ((data_tensor - mean) / (std + 1e-8)).cpu().numpy()
            else:
                # Standard CPU processing
                scaled_data = self.scaler.fit_transform(data)
            
            logger.info("Data preprocessing completed")
            return scaled_data
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise
    
    def extract_features(self, spectrograms: np.ndarray) -> np.ndarray:
        """
        Extract features from spectrograms.
        
        Args:
            spectrograms (np.ndarray): Raw spectrogram data
            
        Returns:
            np.ndarray: Extracted features
        """
        try:
            if USE_MPS:
                # Process on GPU if available
                spec_tensor = torch.from_numpy(spectrograms).to(self.device)
                features = []
                
                # Calculate features on GPU
                mean = spec_tensor.mean(dim=(1, 2))
                std = spec_tensor.std(dim=(1, 2))
                max_val = spec_tensor.max(dim=1)[0].max(dim=1)[0]
                min_val = spec_tensor.min(dim=1)[0].min(dim=1)[0]
                energy = (spec_tensor ** 2).sum(dim=(1, 2))
                
                # Calculate entropy on GPU
                hist = torch.histc(spec_tensor.reshape(-1, spec_tensor.shape[1] * spec_tensor.shape[2]), 
                                 bins=50, min=spec_tensor.min(), max=spec_tensor.max())
                hist = hist / hist.sum()
                entropy = -(hist * torch.log2(hist + 1e-12)).sum()
                
                # Combine features
                features = torch.stack([mean, std, max_val, min_val, energy, 
                                     entropy.expand(spec_tensor.shape[0])], dim=1)
                features = features.cpu().numpy()
                
            else:
                # Standard CPU processing
                features = []
                for spectrogram in spectrograms:
                    feat = []
                    feat.append(np.mean(spectrogram))
                    feat.append(np.std(spectrogram))
                    feat.append(np.max(spectrogram))
                    feat.append(np.min(spectrogram))
                    feat.append(np.sum(spectrogram**2))
                    
                    hist, _ = np.histogram(spectrogram, bins=50)
                    hist = hist / hist.sum()
                    entropy = -np.sum(hist * np.log2(hist + 1e-12))
                    feat.append(entropy)
                    
                    features.append(feat)
                features = np.array(features)
            
            logger.info(f"Extracted {features.shape[1]} features from spectrograms")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            raise

def save_processed_data(data: np.ndarray, output_path: str) -> None:
    """
    Save processed data to disk.
    
    Args:
        data (np.ndarray): Processed data to save
        output_path (str): Path where to save the data
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(output_path, data)
        logger.info(f"Saved processed data to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving data: {str(e)}")
        raise