"""
YOLOv8 License Plate Detection - Eğitim Script'i
=================================================
Bu script, configs/config.yaml dosyasındaki parametreleri kullanarak
YOLOv8 modelini plaka tespiti için eğitir.

Kullanım:
    python src/train.py --config configs/config.yaml
"""

import argparse
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO


def load_config(config_path: str) -> dict:
    """
    YAML config dosyasını yükler.
    
    Args:
        config_path: Config dosyasının yolu
        
    Returns:
        Config parametrelerini içeren dictionary
        
    Raises:
        FileNotFoundError: Config dosyası bulunamazsa
        yaml.YAMLError: YAML parse hatası olursa
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config dosyası bulunamadı: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"✓ Config dosyası yüklendi: {config_path}")
    return config


def setup_training(config: dict) -> YOLO:
    """
    YOLOv8 modelini yükler.
    
    Args:
        config: Eğitim konfigürasyonu
        
    Returns:
        YOLO model objesi
    """
    model_name = config['model']['name']
    pretrained = config['model']['pretrained']
    
    if pretrained:
        # Pre-trained model yükle (örn: yolov8n.pt)
        model_path = f"{model_name}.pt"
        print(f"✓ Pre-trained model yükleniyor: {model_path}")
    else:
        # Scratch'ten model oluştur (örn: yolov8n.yaml)
        model_path = f"{model_name}.yaml"
        print(f"✓ Scratch model oluşturuluyor: {model_path}")
    
    model = YOLO(model_path)
    print(f"✓ Model başarıyla yüklendi: {model_name}")
    
    return model


def train_model(model: YOLO, config: dict) -> Any:
    """
    YOLOv8 modelini eğitir.
    
    Args:
        model: YOLO model objesi
        config: Eğitim konfigürasyonu
        
    Returns:
        Eğitim sonuçları
    """
    # Config bölümlerini al
    dataset_cfg = config['dataset']
    training_cfg = config['training']
    optimizer_cfg = config['optimizer']
    augment_cfg = config['augmentation']
    output_cfg = config['output']
    misc_cfg = config['misc']
    
    print("\n" + "="*50)
    print("EĞİTİM PARAMETRELERİ")
    print("="*50)
    print(f"  Epochs: {training_cfg['epochs']}")
    print(f"  Batch Size: {training_cfg['batch_size']}")
    print(f"  Image Size: {training_cfg['imgsz']}")
    print(f"  Optimizer: {optimizer_cfg['name']}")
    print(f"  Learning Rate: {optimizer_cfg['lr0']}")
    print(f"  Project: {output_cfg['project']}/{output_cfg['name']}")
    print("="*50 + "\n")
    
    # Eğitimi başlat
    results = model.train(
        # Dataset
        data=dataset_cfg['yaml_path'],
        
        # Eğitim parametreleri
        epochs=training_cfg['epochs'],
        batch=training_cfg['batch_size'],
        imgsz=training_cfg['imgsz'],
        patience=training_cfg['patience'],
        save_period=training_cfg['save_period'],
        workers=training_cfg['workers'],
        
        # Optimizer parametreleri
        optimizer=optimizer_cfg['name'],
        lr0=optimizer_cfg['lr0'],
        lrf=optimizer_cfg['lrf'],
        momentum=optimizer_cfg['momentum'],
        weight_decay=optimizer_cfg['weight_decay'],
        
        # Augmentation parametreleri
        hsv_h=augment_cfg['hsv_h'],
        hsv_s=augment_cfg['hsv_s'],
        hsv_v=augment_cfg['hsv_v'],
        degrees=augment_cfg['degrees'],
        translate=augment_cfg['translate'],
        scale=augment_cfg['scale'],
        fliplr=augment_cfg['fliplr'],
        flipud=augment_cfg['flipud'],
        mosaic=augment_cfg['mosaic'],
        mixup=augment_cfg['mixup'],
        
        # Çıktı ayarları
        project=output_cfg['project'],
        name=output_cfg['name'],
        exist_ok=output_cfg['exist_ok'],
        
        # Diğer ayarlar
        seed=misc_cfg['seed'],
        deterministic=misc_cfg['deterministic'],
        verbose=misc_cfg['verbose'],
    )
    
    return results


def print_training_summary(results: Any) -> None:
    """
    Eğitim sonuçlarını özetler.
    
    Args:
        results: Eğitim sonuçları
    """
    print("\n" + "="*50)
    print("EĞİTİM TAMAMLANDI - ÖZET")
    print("="*50)
    
    # Best model yolu
    if hasattr(results, 'save_dir'):
        save_dir = Path(results.save_dir)
        best_model = save_dir / 'weights' / 'best.pt'
        last_model = save_dir / 'weights' / 'last.pt'
        
        print(f"\n📁 Kayıt Dizini: {save_dir}")
        print(f"🏆 En İyi Model: {best_model}")
        print(f"📦 Son Model: {last_model}")
    
    # Metrikler
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print("\n📊 Final Metrikler:")
        
        # Önemli metrikleri yazdır
        metric_names = {
            'metrics/precision(B)': 'Precision',
            'metrics/recall(B)': 'Recall',
            'metrics/mAP50(B)': 'mAP@50',
            'metrics/mAP50-95(B)': 'mAP@50-95',
        }
        
        for key, display_name in metric_names.items():
            if key in metrics:
                print(f"  {display_name}: {metrics[key]:.4f}")
    
    print("="*50)


def print_system_info() -> bool:
    """
    Sistem bilgilerini yazdırır ve GPU kontrolü yapar.
    
    Returns:
        GPU kullanılabilir mi
    """
    print("\n" + "="*50)
    print("SİSTEM BİLGİLERİ")
    print("="*50)
    
    # Python ve PyTorch versiyonları
    import sys
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    
    # CUDA kontrolü
    cuda_available = torch.cuda.is_available()
    print(f"  CUDA Mevcut: {'✓ Evet' if cuda_available else '✗ Hayır'}")
    
    if cuda_available:
        print(f"  CUDA Versiyon: {torch.version.cuda}")
        print(f"  GPU Sayısı: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("  ⚠️ UYARI: GPU bulunamadı, eğitim CPU üzerinde yapılacak!")
    
    print("="*50 + "\n")
    
    return cuda_available


def format_time(seconds: float) -> str:
    """
    Saniyeyi okunabilir formata çevirir.
    
    Args:
        seconds: Saniye cinsinden süre
        
    Returns:
        Formatlanmış süre string'i
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}s {minutes}dk {secs}sn"
    elif minutes > 0:
        return f"{minutes}dk {secs}sn"
    else:
        return f"{secs}sn"


def main() -> None:
    """
    Ana fonksiyon - eğitim pipeline'ını yönetir.
    """
    # Argüman parser
    parser = argparse.ArgumentParser(
        description='YOLOv8 License Plate Detection Eğitim Script\'i'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Config dosyasının yolu (varsayılan: configs/config.yaml)'
    )
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("   YOLOv8 LICENSE PLATE DETECTION - EĞİTİM")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # Sistem bilgilerini yazdır ve GPU kontrolü yap
        gpu_available = print_system_info()
        
        if not gpu_available:
            print("⚠️ GPU olmadan eğitim çok yavaş olacaktır.")
            response = input("Devam etmek istiyor musunuz? (e/h): ")
            if response.lower() != 'e':
                print("Eğitim iptal edildi.")
                return
        
        # Config yükle
        config = load_config(args.config)
        
        # Model oluştur
        model = setup_training(config)
        
        # Eğitimi başlat
        print("\n🚀 Eğitim başlatılıyor...\n")
        results = train_model(model, config)
        
        # Özet yazdır
        print_training_summary(results)
        
    except FileNotFoundError as e:
        print(f"\n❌ HATA: {e}")
        return
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {e}")
        raise
    
    # Toplam süre
    elapsed_time = time.time() - start_time
    print(f"\n⏱️ Toplam Süre: {format_time(elapsed_time)}")
    print("\n✅ Eğitim başarıyla tamamlandı!")


if __name__ == "__main__":
    main()
