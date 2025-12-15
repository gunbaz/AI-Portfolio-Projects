# Utility functions

import torch
import numpy as np


def get_device() -> torch.device:
    """
    CUDA kullanılabilirliğini kontrol eder ve uygun cihazı döndürür.
    
    Returns:
        torch.device: CUDA varsa "cuda", yoksa "cpu" cihazı
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ CUDA kullanılabilir! Cihaz: {device}")
    else:
        device = torch.device("cpu")
        print(f"⚠️  CUDA kullanılamıyor. CPU kullanılıyor: {device}")
    
    return device


def print_system_info() -> None:
    """
    Sistem bilgilerini yazdırır: PyTorch versiyonu, CUDA durumu ve GPU bilgileri.
    """
    print("=" * 60)
    print("SİSTEM BİLGİLERİ")
    print("=" * 60)
    
    # PyTorch versiyonu
    print(f"PyTorch Versiyonu: {torch.__version__}")
    
    # CUDA kullanılabilirliği
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Kullanılabilir: {cuda_available}")
    
    if cuda_available:
        # GPU adı
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU Adı: {gpu_name}")
        
        # GPU bellek kapasitesi (GB)
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU Bellek Kapasitesi: {gpu_memory_gb:.2f} GB")
        
        # CUDA versiyonu
        cuda_version = torch.version.cuda
        print(f"CUDA Versiyonu: {cuda_version}")
        
        # Mevcut GPU bellek kullanımı
        allocated_memory = torch.cuda.memory_allocated(0) / (1024 ** 3)
        reserved_memory = torch.cuda.memory_reserved(0) / (1024 ** 3)
        print(f"Mevcut GPU Bellek Kullanımı: {allocated_memory:.2f} GB (ayrılmış: {reserved_memory:.2f} GB)")
    else:
        print("⚠️  CUDA kullanılamıyor. GPU hızlandırması mevcut değil.")
    
    print("=" * 60)


def set_seed(seed: int = 42) -> None:
    """
    Reproducibility için tüm random seed'leri ayarlar.
    
    Args:
        seed: Kullanılacak random seed değeri (varsayılan: 42)
    """
    # PyTorch random seed
    torch.manual_seed(seed)
    
    # CUDA random seed (eğer CUDA varsa)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # NumPy random seed
    np.random.seed(seed)
    
    # CuDNN deterministik modu (reproducibility için)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"✅ Random seed {seed} olarak ayarlandı (reproducibility için)")


if __name__ == "__main__":
    print_system_info()
    device = get_device()
    
    # Basit GPU testi
    if device.type == "cuda":
        x = torch.rand(1000, 1000, device=device)
        y = torch.rand(1000, 1000, device=device)
        z = torch.matmul(x, y)
        print(f"\n✅ GPU Testi Başarılı! Matrix çarpımı GPU'da çalıştı.")
        print(f"   Sonuç shape: {z.shape}")
