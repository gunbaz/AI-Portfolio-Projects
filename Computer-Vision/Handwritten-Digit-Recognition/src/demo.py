"""
Interactive demo script for MNIST Handwritten Digit Recognition with pagination.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os
from pathlib import Path
from typing import Tuple, Dict
import random

# Üst dizine çıkıp src klasörüne erişim için
sys.path.append(str(Path(__file__).parent.parent))


def load_trained_model(config: dict, device: torch.device) -> nn.Module:
    """
    Best model'i checkpoint'tan yükler.
    
    Args:
        config: Konfigürasyon dictionary'si
        device: Cihaz (cuda/cpu)
    
    Returns:
        nn.Module: Yüklenmiş ve eval moduna alınmış model
    """
    from src.model import MNISTNet
    from src.train import load_checkpoint
    
    # Model yolunu al
    model_save_dir = config["paths"]["model_save_dir"]
    best_model_name = config["paths"]["best_model_name"]
    model_path = os.path.join(model_save_dir, best_model_name)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
    
    # Model oluştur
    model = MNISTNet(config).to(device)
    
    # Checkpoint yükle
    checkpoint = load_checkpoint(model_path, model)
    
    # Eval moduna al
    model.eval()
    
    print(f"✅ Model başarıyla yüklendi: {model_path}")
    print(f"   Epoch: {checkpoint['epoch']}, Validation Accuracy: {checkpoint['accuracy']:.2f}%")
    
    return model


def get_all_test_predictions(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict:
    """
    TÜM test seti üzerinde tahmin yapar.
    
    Args:
        model: Eğitilmiş model
        test_loader: Test DataLoader
        device: Cihaz (cuda/cpu)
    
    Returns:
        Dict: {"images": tensor, "labels": tensor, "predictions": tensor, "confidences": tensor}
    """
    all_images = []
    all_labels = []
    all_predictions = []
    all_confidences = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Softmax ile probability hesapla
            probabilities = torch.softmax(outputs, dim=1)
            
            # Tahminleri ve confidence'ları al
            confidences, predictions = torch.max(probabilities, dim=1)
            
            # CPU'ya taşı
            all_images.append(images.cpu())
            all_labels.append(labels.cpu())
            all_predictions.append(predictions.cpu())
            all_confidences.append(confidences.cpu())
    
    # Concatenate
    return {
        "images": torch.cat(all_images, dim=0),
        "labels": torch.cat(all_labels, dim=0),
        "predictions": torch.cat(all_predictions, dim=0),
        "confidences": torch.cat(all_confidences, dim=0)
    }


def create_page_figure(
    data: Dict,
    page_num: int,
    items_per_page: int = 10,
    class_names: list = None
) -> Tuple:
    """
    Belirli bir sayfa için figure oluşturur.
    
    Args:
        data: Tahmin verileri dictionary'si
        page_num: Sayfa numarası (1-indexed)
        items_per_page: Sayfa başına öğe sayısı
        class_names: Sınıf isimleri listesi
    
    Returns:
        Tuple: (fig, axes)
    """
    images = data["images"]
    labels = data["labels"]
    predictions = data["predictions"]
    confidences = data["confidences"]
    
    # Sayfa için indeksleri hesapla
    start_idx = (page_num - 1) * items_per_page
    end_idx = start_idx + items_per_page
    page_indices = range(start_idx, min(end_idx, len(images)))
    
    # Bu sayfadaki verileri al
    page_images = images[list(page_indices)]
    page_labels = labels[list(page_indices)]
    page_predictions = predictions[list(page_indices)]
    page_confidences = confidences[list(page_indices)]
    
    # Bu sayfanın doğruluk oranını hesapla
    page_correct = (page_predictions == page_labels).sum().item()
    page_accuracy = (page_correct / len(page_indices)) * 100
    
    # Denormalize et (görselleştirme için)
    mean = 0.1307
    std = 0.3081
    page_images_denorm = page_images * std + mean
    page_images_denorm = torch.clamp(page_images_denorm, 0, 1)
    
    # Figure oluştur (2 satır x 5 sütun)
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    
    # Toplam sayfa sayısını hesapla
    total_pages = int(np.ceil(len(images) / items_per_page))
    
    # Başlık
    title = f"Page {page_num}/{total_pages} | Page Accuracy: {page_accuracy:.1f}% | " \
            f"Next: → or 'n' | Previous: ← or 'p' | Quit: 'q'"
    fig.suptitle(title, fontsize=12, fontweight='bold')
    
    # Eksenleri düzleştir
    axes = axes.flatten()
    
    # Her görüntüyü göster
    for idx, ax in enumerate(axes):
        if idx < len(page_indices):
            # Görüntüyü göster
            img = page_images_denorm[idx].squeeze().numpy()
            ax.imshow(img, cmap='gray')
            
            # Gerçek ve tahmin bilgileri
            true_label = class_names[page_labels[idx].item()]
            pred_label = class_names[page_predictions[idx].item()]
            confidence = page_confidences[idx].item() * 100
            
            # Başlık oluştur
            title_text = f"True: {true_label} | Pred: {pred_label} ({confidence:.1f}%)"
            
            # Doğru/yanlış kontrolü
            is_correct = page_labels[idx].item() == page_predictions[idx].item()
            
            if is_correct:
                ax.set_title(title_text, color='green', fontsize=9, fontweight='bold')
                # Yeşil çerçeve
                for spine in ax.spines.values():
                    spine.set_edgecolor('green')
                    spine.set_linewidth(2)
            else:
                ax.set_title(title_text, color='red', fontsize=9, fontweight='bold')
                # Kırmızı çerçeve
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(3)
        else:
            # Boş eksenleri gizle
            ax.axis('off')
    
    plt.tight_layout()
    
    return fig, axes


def interactive_pagination_demo(n_samples: int = 100, items_per_page: int = 10) -> None:
    """
    Ana demo fonksiyonu - sayfalama sistemi ile interaktif demo.
    
    Args:
        n_samples: Gösterilecek toplam örnek sayısı
        items_per_page: Sayfa başına öğe sayısı
    """
    from src.utils import load_config, get_device, set_seed
    from src.data_loader import get_data_loaders, get_class_names
    
    print("=" * 60)
    print("MNIST INTERACTIVE DEMO")
    print("=" * 60)
    print("Kontroller:")
    print("  → veya 'n': Sonraki sayfa")
    print("  ← veya 'p': Önceki sayfa")
    print("  'q': Çıkış")
    print("=" * 60)
    
    # Config yükle
    print("\n📋 Konfigürasyon yükleniyor...")
    config = load_config("configs/config.yaml")
    
    # Seed ayarla (reproducibility için)
    set_seed(config["seed"])
    
    # Cihazı al
    device = get_device()
    
    # Test loader'ı oluştur
    print("\n📦 Test veri seti yükleniyor...")
    train_loader, val_loader, test_loader = get_data_loaders(config)
    
    # Model yükle
    print("\n🤖 Model yükleniyor...")
    model = load_trained_model(config, device)
    
    # Tüm test seti üzerinde tahmin yap
    print("\n🔮 Tüm test seti üzerinde tahmin yapılıyor...")
    all_data = get_all_test_predictions(model, test_loader, device)
    
    # Rastgele örnekler seç
    total_samples = len(all_data["images"])
    n_samples = min(n_samples, total_samples)
    
    print(f"\n🎲 {n_samples} rastgele örnek seçiliyor...")
    random_indices = random.sample(range(total_samples), n_samples)
    
    # Seçilen örnekleri al
    selected_data = {
        "images": all_data["images"][random_indices],
        "labels": all_data["labels"][random_indices],
        "predictions": all_data["predictions"][random_indices],
        "confidences": all_data["confidences"][random_indices]
    }
    
    # Sınıf isimleri
    class_names = get_class_names()
    
    # Sayfalama parametreleri
    total_pages = int(np.ceil(n_samples / items_per_page))
    current_page = 1
    
    # Genel istatistikler
    total_correct = (selected_data["predictions"] == selected_data["labels"]).sum().item()
    total_accuracy = (total_correct / n_samples) * 100
    
    print(f"\n📊 Genel İstatistikler:")
    print(f"   Toplam Örnek: {n_samples}")
    print(f"   Doğru Tahmin: {total_correct}/{n_samples} ({total_accuracy:.2f}%)")
    print(f"   Yanlış Tahmin: {n_samples - total_correct}/{n_samples} ({100 - total_accuracy:.2f}%)")
    print(f"   Toplam Sayfa: {total_pages}")
    print(f"\n⌨️  Figure'a odaklanın ve klavye tuşlarını kullanın.")
    print("=" * 60)
    
    # Figure ve sayfa bilgilerini tutmak için dictionary
    figure_state = {
        'fig': None,
        'current_page': current_page
    }
    
    # Sayfa güncelleme fonksiyonu
    def show_page(page_num):
        # Mevcut figure varsa kapat
        if figure_state['fig'] is not None:
            plt.close(figure_state['fig'])
        
        # Yeni sayfa oluştur
        fig, axes = create_page_figure(
            selected_data, page_num, items_per_page, class_names
        )
        
        # Event handler'ı bağla
        fig.canvas.mpl_connect('key_press_event', on_key)
        
        # Pencere başlığını ayarla
        try:
            fig.canvas.manager.set_window_title('MNIST Demo - Interactive Pagination')
        except:
            pass
        
        figure_state['fig'] = fig
        figure_state['current_page'] = page_num
        
        plt.show()
    
    # Event handler fonksiyonu
    def on_key(event):
        nonlocal current_page
        
        if event.key in ['right', 'n']:
            if current_page < total_pages:
                current_page += 1
                print(f"📄 Sayfa {current_page}/{total_pages} gösteriliyor...")
                show_page(current_page)
            else:
                print("⚠️  Son sayfadasınız!")
        elif event.key in ['left', 'p']:
            if current_page > 1:
                current_page -= 1
                print(f"📄 Sayfa {current_page}/{total_pages} gösteriliyor...")
                show_page(current_page)
            else:
                print("⚠️  İlk sayfadasınız!")
        elif event.key == 'q':
            plt.close('all')
            print("\n" + "=" * 60)
            print("FİNAL ÖZET")
            print("=" * 60)
            print(f"Toplam Örnek: {n_samples}")
            print(f"Doğru Tahmin: {total_correct}/{n_samples} ({total_accuracy:.2f}%)")
            print(f"Yanlış Tahmin: {n_samples - total_correct}/{n_samples} ({100 - total_accuracy:.2f}%)")
            print(f"Ortalama Confidence: {selected_data['confidences'].mean().item() * 100:.2f}%")
            print("\n✅ Demo kapatıldı.")
            print("=" * 60)
    
    # İlk sayfayı göster
    print(f"\n📄 Sayfa {current_page}/{total_pages} gösteriliyor...")
    show_page(current_page)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MNIST Handwritten Digit Recognition - Interactive Demo with Pagination"
    )
    parser.add_argument(
        "-n", "--samples",
        type=int,
        default=100,
        help="Total number of random test samples to visualize (default: 100)"
    )
    parser.add_argument(
        "-p", "--per-page",
        type=int,
        default=10,
        help="Number of samples per page (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Örnek sayısını kontrol et
    if args.samples < 1:
        print("⚠️  Örnek sayısı en az 1 olmalıdır. Varsayılan değer (100) kullanılıyor.")
        args.samples = 100
    elif args.samples > 1000:
        print("⚠️  Örnek sayısı çok büyük. Maksimum 1000 örnek gösterilecek.")
        args.samples = 1000
    
    # Sayfa başına öğe sayısını kontrol et
    if args.per_page < 1:
        print("⚠️  Sayfa başına öğe sayısı en az 1 olmalıdır. Varsayılan değer (10) kullanılıyor.")
        args.per_page = 10
    elif args.per_page > 50:
        print("⚠️  Sayfa başına öğe sayısı çok büyük. Maksimum 50 öğe gösterilecek.")
        args.per_page = 50
    
    interactive_pagination_demo(n_samples=args.samples, items_per_page=args.per_page)
