"""
License Plate Detection - Utility Functions
"""

import os
import torch
from dotenv import load_dotenv
from roboflow import Roboflow
import ultralytics


def load_env() -> str:
    """
    .env dosyasını yükle ve ROBOFLOW_API_KEY'i döndür.
    
    Returns:
        str: Roboflow API anahtarı
    """
    load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY")
    
    if not api_key:
        raise ValueError("ROBOFLOW_API_KEY .env dosyasında bulunamadı!")
    
    return api_key


def download_dataset(api_key: str, save_dir: str = "data") -> str:
    """
    Roboflow'dan License Plate dataset'ini indir.
    
    Args:
        api_key: Roboflow API anahtarı
        save_dir: Dataset'in kaydedileceği dizin
        
    Returns:
        str: Dataset lokasyon yolu
    """
    rf = Roboflow(api_key=api_key)
    
    project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
    version = project.version(4)
    
    dataset = version.download("yolov8", location=save_dir)
    
    return dataset.location


def get_device() -> str:
    """
    CUDA varsa 'cuda', yoksa 'cpu' döndür.
    
    Returns:
        str: Kullanılacak cihaz ('cuda' veya 'cpu')
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def print_system_info() -> None:
    """
    Sistem bilgilerini yazdır:
    - PyTorch versiyonu
    - CUDA durumu
    - GPU adı (varsa)
    - Ultralytics versiyonu
    """
    print("=" * 50)
    print("SYSTEM INFORMATION")
    print("=" * 50)
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("GPU Name: N/A (CPU mode)")
    
    print(f"Ultralytics Version: {ultralytics.__version__}")
    print(f"Device: {get_device()}")
    print("=" * 50)


if __name__ == "__main__":
    print_system_info()
    
    api_key = load_env()
    if api_key:
        print(f"\nAPI Key loaded successfully!")
        print(f"API Key (first 5 chars): {api_key[:5]}...")
        
        # Dataset indirme (isteğe bağlı)
        # dataset_path = download_dataset(api_key)
        # print(f"Dataset downloaded to: {dataset_path}")
