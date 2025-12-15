# Model architecture

import torch
import torch.nn as nn
from typing import Dict


class MNISTNet(nn.Module):
    """
    MNIST el yazısı rakam tanıma için CNN modeli.
    LeNet-5'ten ilham alınmış, modern iyileştirmelerle (BatchNorm, Dropout).
    
    Model Mimarisi:
    - Conv Block 1: 1 -> 32 -> 32 kanal, MaxPool (28x28 -> 14x14)
    - Conv Block 2: 32 -> 64 -> 64 kanal, MaxPool (14x14 -> 7x7)
    - Fully Connected: 3136 -> 256 -> 10
    
    Input Shape: (batch_size, 1, 28, 28)
    Output Shape: (batch_size, 10) - logits (softmax uygulanmamış)
    
    Args:
        config: Konfigürasyon dictionary'si. "model" anahtarı altında "dropout_rate" içermeli.
    """
    
    def __init__(self, config: Dict):
        """
        Model mimarisini oluşturur.
        
        Args:
            config: Konfigürasyon dictionary'si
        """
        super(MNISTNet, self).__init__()
        
        # Config'den dropout_rate al (varsayılan: 0.5)
        dropout_rate = config.get("model", {}).get("dropout_rate", 0.5)
        
        # Conv Block 1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),  # 28x28 -> 28x28
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),  # 28x28 -> 28x28
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28 -> 14x14
            nn.Dropout2d(p=0.25)
        )
        
        # Conv Block 2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),  # 14x14 -> 14x14
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),  # 14x14 -> 14x14
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14 -> 7x7
            nn.Dropout2d(p=0.25)
        )
        
        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Flatten(),  # 64 * 7 * 7 = 3136
            nn.Linear(in_features=64 * 7 * 7, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=256, out_features=10)  # 10 sınıf (0-9)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor, shape (batch_size, 1, 28, 28)
        
        Returns:
            torch.Tensor: Logits, shape (batch_size, 10)
                          Not: Softmax uygulanmamış, CrossEntropyLoss bunu halleder
        """
        # Conv Block 1
        x = self.conv_block1(x)  # (batch_size, 32, 14, 14)
        
        # Conv Block 2
        x = self.conv_block2(x)  # (batch_size, 64, 7, 7)
        
        # Fully Connected
        x = self.fc(x)  # (batch_size, 10)
        
        return x


if __name__ == "__main__":
    try:
        from torchsummary import summary
        TORCHSUMMARY_AVAILABLE = True
    except ImportError:
        TORCHSUMMARY_AVAILABLE = False
        print("⚠️  torchsummary bulunamadı. Model özeti gösterilemeyecek.")
        print("   Kurmak için: pip install torchsummary")
    
    import torch
    
    # Config simülasyonu
    config = {"model": {"dropout_rate": 0.5}}
    
    # Model oluştur
    model = MNISTNet(config)
    
    # GPU varsa kullan
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print("=" * 60)
    print("MODEL TESTİ")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model parametre sayısı: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parametre sayısı: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("=" * 60)
    
    # Model özeti
    if TORCHSUMMARY_AVAILABLE:
        print("\nModel Mimarisi:")
        print("=" * 60)
        summary(model, (1, 28, 28))
    
    # Forward pass testi
    print("\nForward Pass Testi:")
    print("=" * 60)
    model.eval()  # Test moduna al (BatchNorm için gerekli)
    
    with torch.no_grad():  # Gradient hesaplamasını kapat
        dummy_input = torch.randn(1, 1, 28, 28).to(device)
        output = model(dummy_input)
        print(f"Input shape:  {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output min:   {output.min().item():.4f}")
        print(f"Output max:   {output.max().item():.4f}")
        print(f"Output mean:  {output.mean().item():.4f}")
        
        # Batch testi
        batch_input = torch.randn(4, 1, 28, 28).to(device)
        batch_output = model(batch_input)
        print(f"\nBatch Test (batch_size=4):")
        print(f"Input shape:  {batch_input.shape}")
        print(f"Output shape: {batch_output.shape}")
    
    print("\n✅ Model çalışıyor!")
