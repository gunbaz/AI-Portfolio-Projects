# Evaluation metrics

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from pathlib import Path
from typing import Dict, Tuple
import os


def load_model(config: Dict, model_path: str, device: torch.device) -> nn.Module:
    """
    Modeli checkpoint'tan yükler.
    
    Args:
        config: Konfigürasyon dictionary'si
        model_path: Model checkpoint dosya yolu
        device: Cihaz (cuda/cpu)
    
    Returns:
        nn.Module: Yüklenmiş ve eval moduna alınmış model
    """
    from src.model import MNISTNet
    from src.train import load_checkpoint
    
    # Model oluştur
    model = MNISTNet(config).to(device)
    
    # Checkpoint yükle
    checkpoint = load_checkpoint(model_path, model)
    
    # Eval moduna al
    model.eval()
    
    return model


def get_predictions(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Tüm test seti üzerinde tahmin yapar.
    
    Args:
        model: Değerlendirilecek model
        test_loader: Test DataLoader
        device: Cihaz (cuda/cpu)
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (predictions, labels, probabilities)
    """
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Softmax ile probability hesapla
            probabilities = torch.softmax(outputs, dim=1)
            
            # Tahminleri al
            _, predicted = torch.max(outputs, 1)
            
            # CPU'ya taşı ve numpy'ye çevir
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    save_path: str
) -> None:
    """
    Confusion matrix'i görselleştirir ve kaydeder.
    
    Args:
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
        class_names: Sınıf isimleri listesi
        save_path: Kayıt yolu
    """
    # Confusion matrix hesapla
    cm = confusion_matrix(y_true, y_pred)
    
    # Görselleştir
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Örnek Sayısı'}
    )
    plt.title('Confusion Matrix - MNIST Test Seti', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Tahmin Edilen', fontsize=12)
    plt.ylabel('Gerçek', fontsize=12)
    plt.tight_layout()
    
    # Kaydet
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Confusion matrix kaydedildi: {save_path}")
    plt.close()


def plot_misclassified_examples(
    images: np.ndarray,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    class_names: list,
    save_path: str,
    num_examples: int = 25
) -> None:
    """
    Yanlış tahmin edilen örnekleri görselleştirir.
    
    Args:
        images: Görüntü array'i (normalize edilmiş)
        true_labels: Gerçek etiketler
        pred_labels: Tahmin edilen etiketler
        class_names: Sınıf isimleri listesi
        save_path: Kayıt yolu
        num_examples: Gösterilecek örnek sayısı (varsayılan: 25)
    """
    # Yanlış tahmin edilenleri bul
    misclassified_indices = np.where(true_labels != pred_labels)[0]
    
    if len(misclassified_indices) == 0:
        print("⚠️  Yanlış tahmin edilen örnek bulunamadı!")
        return
    
    # İlk num_examples kadarını al
    num_examples = min(num_examples, len(misclassified_indices))
    selected_indices = misclassified_indices[:num_examples]
    
    # Denormalize et (görselleştirme için)
    mean = 0.1307
    std = 0.3081
    images_denorm = images * std + mean
    images_denorm = np.clip(images_denorm, 0, 1)
    
    # Grid oluştur
    rows = int(np.ceil(np.sqrt(num_examples)))
    cols = int(np.ceil(num_examples / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    fig.suptitle('Yanlış Tahmin Edilen Örnekler', fontsize=16, fontweight='bold')
    
    # Eksenleri düzleştir
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    else:
        axes = axes.flatten()
    
    for idx, mis_idx in enumerate(selected_indices):
        ax = axes[idx] if num_examples > 1 else axes
        
        # Görüntüyü göster
        img = images_denorm[mis_idx].squeeze()
        ax.imshow(img, cmap='gray')
        
        # Başlık (kırmızı renkte)
        true_label = class_names[true_labels[mis_idx]]
        pred_label = class_names[pred_labels[mis_idx]]
        ax.set_title(f'Gerçek: {true_label}, Tahmin: {pred_label}', 
                    color='red', fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # Kullanılmayan eksenleri gizle
    for idx in range(num_examples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Yanlış tahmin edilen örnekler kaydedildi: {save_path}")
    plt.close()


def plot_per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    save_path: str
) -> None:
    """
    Her sınıf için accuracy'yi görselleştirir.
    
    Args:
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
        class_names: Sınıf isimleri listesi
        save_path: Kayıt yolu
    """
    # Her sınıf için accuracy hesapla
    per_class_acc = []
    for i in range(len(class_names)):
        mask = y_true == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == i).sum() / mask.sum() * 100
        else:
            acc = 0.0
        per_class_acc.append(acc)
    
    # Horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(class_names, per_class_acc, color='steelblue', alpha=0.7)
    
    # Değerleri bar üzerinde göster
    for i, (bar, acc) in enumerate(zip(bars, per_class_acc)):
        ax.text(acc + 0.5, i, f'{acc:.2f}%', 
               va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_ylabel('Sınıf', fontsize=12)
    ax.set_title('Sınıf Bazlı Accuracy - MNIST Test Seti', 
                fontsize=14, fontweight='bold')
    ax.set_xlim([0, 105])
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Sınıf bazlı accuracy grafiği kaydedildi: {save_path}")
    plt.close()


def evaluate_model(config: Dict) -> Dict:
    """
    Ana değerlendirme fonksiyonu.
    
    Args:
        config: Konfigürasyon dictionary'si
    
    Returns:
        Dict: Değerlendirme sonuçları
    """
    from src.utils import get_device
    from src.data_loader import get_data_loaders, get_class_names
    
    # Cihazı al
    device = get_device()
    
    # Test loader'ı oluştur
    print("\n📦 Test veri seti yükleniyor...")
    train_loader, val_loader, test_loader = get_data_loaders(config)
    
    # Best model yolunu al
    model_save_dir = config["paths"]["model_save_dir"]
    best_model_name = config["paths"]["best_model_name"]
    best_model_path = os.path.join(model_save_dir, best_model_name)
    
    # Model yükle
    print(f"\n📥 Model yükleniyor: {best_model_path}")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {best_model_path}")
    
    model = load_model(config, best_model_path, device)
    
    # Tahminleri al
    print("\n🔮 Test seti üzerinde tahmin yapılıyor...")
    predictions, labels, probabilities = get_predictions(model, test_loader, device)
    
    # Metrikleri hesapla
    accuracy = accuracy_score(labels, predictions) * 100
    cm = confusion_matrix(labels, predictions)
    class_names = get_class_names()
    report = classification_report(labels, predictions, target_names=class_names)
    
    # Sonuçları yazdır
    print("\n" + "=" * 60)
    print("DEĞERLENDİRME SONUÇLARI")
    print("=" * 60)
    print(f"\n📊 Genel Accuracy: {accuracy:.2f}%")
    print(f"\n📋 Classification Report:")
    print(report)
    print(f"\n📈 Confusion Matrix:")
    print(cm)
    
    # Grafikleri oluştur ve kaydet
    figures_dir = config["paths"]["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)
    
    print("\n📊 Grafikler oluşturuluyor...")
    
    # Confusion matrix
    cm_path = os.path.join(figures_dir, "confusion_matrix.png")
    plot_confusion_matrix(labels, predictions, class_names, cm_path)
    
    # Yanlış tahmin edilen örnekler için görüntüleri al
    # Test loader'dan görüntüleri topla
    all_images = []
    all_labels_list = []
    for images, labels_batch in test_loader:
        all_images.append(images.numpy())
        all_labels_list.append(labels_batch.numpy())
    all_images = np.concatenate(all_images, axis=0)
    all_labels_array = np.concatenate(all_labels_list, axis=0)
    
    # Misclassified examples
    misclassified_path = os.path.join(figures_dir, "misclassified_examples.png")
    plot_misclassified_examples(
        all_images, labels, predictions, class_names, misclassified_path
    )
    
    # Per-class accuracy
    per_class_path = os.path.join(figures_dir, "per_class_accuracy.png")
    plot_per_class_accuracy(labels, predictions, class_names, per_class_path)
    
    # Sonuçları döndür
    results = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm
    }
    
    return results


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.utils import load_config, set_seed, print_system_info
    
    print_system_info()
    config = load_config()
    set_seed(config["seed"])
    
    print("\n" + "=" * 50)
    print("MODEL DEĞERLENDİRMESİ BAŞLIYOR")
    print("=" * 50)
    
    results = evaluate_model(config)
    
    print("\n" + "=" * 50)
    print("DEĞERLENDİRME TAMAMLANDI")
    print(f"Test Accuracy: {results['accuracy']:.2f}%")
    print("=" * 50)
