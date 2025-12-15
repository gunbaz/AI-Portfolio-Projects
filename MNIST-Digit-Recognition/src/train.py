# Training loop

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from pathlib import Path
from typing import Dict, Tuple, Optional


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """
    Bir epoch boyunca model eğitir.
    
    Args:
        model: Eğitilecek model
        train_loader: Training DataLoader
        criterion: Loss fonksiyonu
        optimizer: Optimizer
        device: Cihaz (cuda/cpu)
    
    Returns:
        Tuple[float, float]: (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in pbar:
        # Veriyi cihaza taşı
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # İstatistikleri hesapla
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Progress bar'ı güncelle
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Bir epoch boyunca validation yapar.
    
    Args:
        model: Değerlendirilecek model
        val_loader: Validation DataLoader
        criterion: Loss fonksiyonu
        device: Cihaz (cuda/cpu)
    
    Returns:
        Tuple[float, float]: (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        for images, labels in pbar:
            # Veriyi cihaza taşı
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # İstatistikleri hesapla
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Progress bar'ı güncelle
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    avg_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    accuracy: float,
    filepath: str
) -> None:
    """
    Model checkpoint'ini kaydeder.
    
    Args:
        model: Kaydedilecek model
        optimizer: Optimizer
        epoch: Epoch numarası
        loss: Loss değeri
        accuracy: Accuracy değeri
        filepath: Kayıt yolu
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }
    
    # Dizin yoksa oluştur
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    torch.save(checkpoint, filepath)
    print(f"✅ Checkpoint kaydedildi: {filepath}")


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None
) -> Dict:
    """
    Checkpoint yükler.
    
    Args:
        filepath: Checkpoint dosya yolu
        model: Model (state_dict yüklenecek)
        optimizer: Optimizer (opsiyonel, state_dict yüklenecek)
    
    Returns:
        Dict: Checkpoint dictionary'si
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"✅ Checkpoint yüklendi: {filepath}")
    print(f"   Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}, Accuracy: {checkpoint['accuracy']:.2f}%")
    
    return checkpoint


def plot_training_history(history: Dict, save_dir: str) -> None:
    """
    Training geçmişini görselleştirir ve kaydeder.
    
    Args:
        history: Training history dictionary'si
        save_dir: Kayıt dizini
    """
    epochs = range(1, len(history['train_losses']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curve
    axes[0].plot(epochs, history['train_losses'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training ve Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curve
    axes[1].plot(epochs, history['train_accs'], 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_accs'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training ve Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Kaydet
    save_path = os.path.join(save_dir, "training_history.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Training grafikleri kaydedildi: {save_path}")
    plt.close()


def train_model(config: Dict) -> Dict:
    """
    Ana training fonksiyonu.
    
    Args:
        config: Konfigürasyon dictionary'si
    
    Returns:
        Dict: Training history dictionary'si
    """
    # Import'lar (circular import'u önlemek için)
    from src.utils import get_device
    from src.data_loader import get_data_loaders
    from src.model import MNISTNet
    
    # Cihazı al
    device = get_device()
    
    # DataLoader'ları oluştur
    print("\n📦 Veri setleri yükleniyor...")
    train_loader, val_loader, test_loader = get_data_loaders(config)
    
    # Model oluştur
    print("🏗️  Model oluşturuluyor...")
    model = MNISTNet(config).to(device)
    
    # Loss ve Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    
    # Training parametreleri
    epochs = config["training"]["epochs"]
    patience = config["training"]["early_stopping_patience"]
    model_save_dir = config["paths"]["model_save_dir"]
    best_model_path = os.path.join(model_save_dir, config["paths"]["best_model_name"])
    
    # History listeleri
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Early stopping için
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    
    print(f"\n🚀 Eğitim başlıyor...")
    print(f"   Epochs: {epochs}")
    print(f"   Learning Rate: {config['training']['learning_rate']}")
    print(f"   Early Stopping Patience: {patience}")
    print("=" * 60)
    
    # Training loop
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Sonuçları yazdır
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Best model kontrolü ve kaydetme
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, best_model_path)
            patience_counter = 0
            print(f"✨ Yeni en iyi model! (Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"⏳ Early stopping sayacı: {patience_counter}/{patience}")
        
        # Early stopping kontrolü
        if patience_counter >= patience:
            print(f"\n⏹️  Early stopping tetiklendi! {patience} epoch boyunca iyileşme olmadı.")
            break
    
    # Training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc
    }
    
    # Grafikleri çiz ve kaydet
    print("\n📊 Training grafikleri oluşturuluyor...")
    figures_dir = config["paths"]["figures_dir"]
    plot_training_history(history, figures_dir)
    
    return history


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.utils import load_config, set_seed, print_system_info
    
    print_system_info()
    config = load_config()
    set_seed(config["seed"])
    
    print("\n" + "=" * 50)
    print("EĞİTİM BAŞLIYOR")
    print("=" * 50)
    
    history = train_model(config)
    
    print("\n" + "=" * 50)
    print("EĞİTİM TAMAMLANDI")
    print(f"En iyi validation accuracy: {history['best_val_acc']:.2f}%")
    print("=" * 50)
