"""
MNIST veri yükleme modülü.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from typing import Tuple


def get_transforms() -> dict:
    """
    Train ve test için transform pipeline'larını döndürür.
    
    Returns:
        dict: "train" ve "test" anahtarlarına sahip transform dictionary'si
    """
    # MNIST standart normalize değerleri
    mean = 0.1307
    std = 0.3081
    
    transform_train = transforms.Compose([
        transforms.ToTensor(),  # PIL Image veya numpy array'i tensor'a çevir
        transforms.Normalize(mean=(mean,), std=(std,))  # Normalize et
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(mean,), std=(std,))
    ])
    
    return {
        "train": transform_train,
        "test": transform_test
    }


def get_data_loaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    MNIST veri setini yükler ve train/validation/test DataLoader'larını döndürür.
    
    Args:
        config: Konfigürasyon dictionary'si (data ve paths bölümlerini içermeli)
    
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader)
    """
    # Transform'ları al
    transform_dict = get_transforms()
    
    # Veri dizini
    data_dir = config.get("data", {}).get("data_dir", "data")
    
    # Config'den parametreleri al
    batch_size = config.get("data", {}).get("batch_size", 64)
    num_workers = config.get("data", {}).get("num_workers", 4)
    pin_memory = config.get("data", {}).get("pin_memory", True)
    validation_split = config.get("data", {}).get("validation_split", 0.1)
    
    # Train dataset'i indir ve yükle
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform_dict["train"]
    )
    
    # Test dataset'i indir ve yükle
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform_dict["test"]
    )
    
    # Train setini train ve validation olarak böl
    total_train_size = len(train_dataset)
    val_size = int(total_train_size * validation_split)
    train_size = total_train_size - val_size
    
    train_dataset, val_dataset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.get("seed", 42))
    )
    
    # DataLoader'ları oluştur
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Train için shuffle aktif
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Validation için shuffle kapalı
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Test için shuffle kapalı
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


def get_class_names() -> list[str]:
    """
    MNIST sınıf isimlerini döndürür.
    
    Returns:
        list[str]: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    """
    return ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.utils import load_config
    
    config = load_config()
    train_loader, val_loader, test_loader = get_data_loaders(config)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Bir batch örneği
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")

