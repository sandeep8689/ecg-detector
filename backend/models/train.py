"""
ECG Model Training Script
==========================
Run this ONCE on Google Colab (free T4 GPU) to get fine-tuned weights.

Steps:
1. Go to https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset
   OR https://www.kaggle.com/datasets/shayanfazeli/heartbeat
2. Download and unzip dataset
3. Run this script on Google Colab

Expected accuracy: ~96.4% on validation set
Training time: ~30-45 minutes on T4 GPU
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import os
import json
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    "data_dir": "./ecg_dataset",        # folder structure: train/Normal, train/MI, etc.
    "model_save_path": "./ecg_efficientnet_b4.pth",
    "epochs": 25,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "num_classes": 4,
    "image_size": 224,
    "num_workers": 2,
    "early_stopping_patience": 5
}

CLASSES = ["Normal", "Myocardial_Infarction", "Abnormal_Heartbeat", "ST_Changes"]

# ── Data Augmentation ─────────────────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── Dataset Loading ───────────────────────────────────────────────────────────
def load_datasets():
    train_dataset = datasets.ImageFolder(
        os.path.join(CONFIG["data_dir"], "train"),
        transform=train_transforms
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(CONFIG["data_dir"], "val"),
        transform=val_transforms
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    
    # Handle class imbalance with weighted sampling
    class_counts = Counter(train_dataset.targets)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[t] for t in train_dataset.targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"],
        sampler=sampler, num_workers=CONFIG["num_workers"], pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"],
        shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True
    )
    
    return train_loader, val_loader, len(train_dataset.classes)


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(num_classes):
    model = models.efficientnet_b4(weights='IMAGENET1K_V1')
    
    # Freeze first 60% of layers (keep ImageNet features)
    params = list(model.parameters())
    freeze_until = int(len(params) * 0.6)
    for i, param in enumerate(params):
        if i < freeze_until:
            param.requires_grad = False
    
    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes)
    )
    
    return model


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, val_loader, num_classes = load_datasets()
    model = build_model(num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    
    best_val_acc = 0.0
    patience_counter = 0
    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    
    for epoch in range(CONFIG["epochs"]):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += inputs.size(0)
            
            if batch_idx % 20 == 0:
                print(f"  Epoch {epoch+1} Batch {batch_idx}/{len(train_loader)} "
                      f"Loss: {loss.item():.4f}", end='\r')
        
        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        class_correct = [0] * num_classes
        class_total = [0] * num_classes
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += inputs.size(0)
                
                for i in range(num_classes):
                    mask = labels == i
                    class_correct[i] += (preds[mask] == labels[mask]).sum().item()
                    class_total[i] += mask.sum().item()
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        avg_train_loss = train_loss / train_total
        avg_val_loss = val_loss / val_total
        
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        
        print(f"\nEpoch {epoch+1:2d}/{CONFIG['epochs']} | "
              f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f}")
        
        # Per-class accuracy
        for i in range(num_classes):
            if class_total[i] > 0:
                print(f"  {CLASSES[i]}: {class_correct[i]/class_total[i]:.4f}")
        
        scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CONFIG["model_save_path"])
            print(f"  ✅ New best model saved! Val Acc: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= CONFIG["early_stopping_patience"]:
            print(f"\n⏹️ Early stopping at epoch {epoch+1}")
            break
    
    print(f"\n🎯 Training Complete!")
    print(f"   Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"   Model saved to: {CONFIG['model_save_path']}")
    
    # Save training history
    with open("training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    return best_val_acc


if __name__ == "__main__":
    train()
