"""
Computer Vision Techniques Cheatsheet
From basic image processing to advanced deep learning architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, models
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ============================================================================
# IMAGE PREPROCESSING & AUGMENTATION
# ============================================================================

# Basic transformations with torchvision
basic_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Advanced augmentations with Albumentations
advanced_transforms = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(p=0.3),
    A.Blur(blur_limit=3, p=0.3),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Test-time augmentation (TTA)
def tta_predict(model, image, num_augments=5):
    """Test-time augmentation for better predictions"""
    predictions = []
    for _ in range(num_augments):
        aug_image = advanced_transforms(image=image)['image']
        with torch.no_grad():
            pred = model(aug_image.unsqueeze(0))
        predictions.append(pred)
    return torch.mean(torch.stack(predictions), dim=0)

# ============================================================================
# CNN ARCHITECTURES
# ============================================================================

# Basic CNN
class BasicCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# ResNet Block
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Attention mechanism
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, H, W = x.size()
        query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(B, -1, H * W)
        attention = F.softmax(torch.bmm(query, key), dim=-1)
        value = self.value(x).view(B, -1, H * W)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        return self.gamma * out + x

# Vision Transformer (ViT) components
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

# ============================================================================
# TRANSFER LEARNING
# ============================================================================

def get_pretrained_model(model_name='resnet50', num_classes=10, freeze_backbone=True):
    """Get pretrained model with custom classifier"""
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=True)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    elif model_name == 'vit_b_16':
        model = models.vit_b_16(pretrained=True)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        
    return model

# Fine-tuning with gradual unfreezing
def unfreeze_model_gradually(model, current_epoch, total_epochs):
    """Gradually unfreeze layers during training"""
    if current_epoch > total_epochs * 0.3:
        for param in model.layer4.parameters():
            param.requires_grad = True
    if current_epoch > total_epochs * 0.6:
        for param in model.layer3.parameters():
            param.requires_grad = True

# ============================================================================
# OBJECT DETECTION
# ============================================================================

# Simple object detection with pretrained models
def detect_objects(image_path, confidence_threshold=0.5):
    """Detect objects using Faster R-CNN"""
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    
    image = Image.open(image_path).convert('RGB')
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        predictions = model(image_tensor)
    
    boxes = predictions[0]['boxes'][predictions[0]['scores'] > confidence_threshold]
    labels = predictions[0]['labels'][predictions[0]['scores'] > confidence_threshold]
    scores = predictions[0]['scores'][predictions[0]['scores'] > confidence_threshold]
    
    return boxes, labels, scores

# YOLO-style detection head
class YOLOHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors=3):
        super(YOLOHead, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        # 5 = (x, y, w, h, confidence)
        self.conv = nn.Conv2d(in_channels, num_anchors * (5 + num_classes), 1)
    
    def forward(self, x):
        return self.conv(x)

# Non-Maximum Suppression
def nms(boxes, scores, iou_threshold=0.5):
    """Non-Maximum Suppression"""
    indices = torchvision.ops.nms(boxes, scores, iou_threshold)
    return indices

# ============================================================================
# SEMANTIC SEGMENTATION
# ============================================================================

# U-Net architecture
class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        self.out = nn.Conv2d(64, num_classes, 1)
        self.pool = nn.MaxPool2d(2)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        return self.out(dec1)

# DeepLab with ASPP (Atrous Spatial Pyramid Pooling)
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, 1)
        self.project = nn.Conv2d(out_channels * 5, out_channels, 1)
        
    def forward(self, x):
        size = x.shape[2:]
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = F.interpolate(self.conv5(self.pool(x)), size=size, mode='bilinear', align_corners=False)
        out = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
        return self.project(out)

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

# Focal Loss for imbalanced datasets
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# Dice Loss for segmentation
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        return 1 - dice

# IoU Loss
class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()
        
    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum() - intersection
        iou = intersection / (union + 1e-8)
        return 1 - iou

# ============================================================================
# FEATURE EXTRACTION & EMBEDDINGS
# ============================================================================

def extract_features(model, dataloader, device='cuda'):
    """Extract feature embeddings from a model"""
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            # Remove classification head
            feat = model.conv1(images)
            feat = model.bn1(feat)
            feat = model.relu(feat)
            feat = model.maxpool(feat)
            feat = model.layer1(feat)
            feat = model.layer2(feat)
            feat = model.layer3(feat)
            feat = model.layer4(feat)
            feat = model.avgpool(feat)
            feat = torch.flatten(feat, 1)
            
            features.append(feat.cpu())
            labels.append(targets.cpu())
    
    return torch.cat(features), torch.cat(labels)

# Siamese Network for similarity learning
class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 10),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 7),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim)
        )
    
    def forward_one(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        return out1, out2

# Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                         label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

# ============================================================================
# IMAGE GENERATION (GAN)
# ============================================================================

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=3):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, img_channels=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        return self.model(img)
# ============================================================================
# 3. WEIGHT INITIALIZATION
# ============================================================================

def weights_init(m):
    """Initialize weights according to DCGAN paper"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ============================================================================
# 4. DATASET PREPARATION
# ============================================================================

def get_dataloader(data_path, img_size=64, batch_size=128):
    """
    Create dataloader for training
    
    Dataset structure:
    data_path/
        class1/
            img1.jpg
            img2.jpg
        class2/
            img1.jpg
            img2.jpg
    
    Or for single class:
    data_path/
        img1.jpg
        img2.jpg
        img3.jpg
    """
    
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1]
    ])
    
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    return dataloader
class GANTrainer:
    def __init__(self, generator, discriminator, device, lr=0.0002, beta1=0.5):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        
        # Initialize weights
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Optimizers (using Adam with specific betas from DCGAN paper)
        self.optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
        
        # For tracking
        self.fixed_noise = torch.randn(64, 100, 1, 1, device=device)
        self.losses_G = []
        self.losses_D = []
    
    def train_step(self, real_imgs, latent_dim=100):
        batch_size = real_imgs.size(0)
        real_imgs = real_imgs.to(self.device)
        
        # Labels
        real_label = torch.ones(batch_size, 1, device=self.device)
        fake_label = torch.zeros(batch_size, 1, device=self.device)
        
        # ====================================
        # Train Discriminator
        # ====================================
        self.optimizer_D.zero_grad()
        
        # Real images
        output_real = self.discriminator(real_imgs)
        loss_D_real = self.criterion(output_real, real_label)
        
        # Fake images
        z = torch.randn(batch_size, latent_dim, 1, 1, device=self.device)
        fake_imgs = self.generator(z)
        output_fake = self.discriminator(fake_imgs.detach())
        loss_D_fake = self.criterion(output_fake, fake_label)
        
        # Total discriminator loss
        loss_D = loss_D_real + loss_D_fake
        loss_D.backward()
        self.optimizer_D.step()
        
        # ====================================
        # Train Generator
        # ====================================
        self.optimizer_G.zero_grad()
        
        # Generate fake images and get discriminator opinion
        output = self.discriminator(fake_imgs)
        loss_G = self.criterion(output, real_label)  # Generator wants D to think these are real
        
        loss_G.backward()
        self.optimizer_G.step()
        
        return loss_D.item(), loss_G.item()
    
    def train(self, dataloader, epochs=100, save_interval=10):
        print(f"Starting training on {self.device}")
        
        for epoch in range(epochs):
            epoch_loss_D = 0
            epoch_loss_G = 0
            
            for i, (real_imgs, _) in enumerate(dataloader):
                loss_D, loss_G = self.train_step(real_imgs)
                epoch_loss_D += loss_D
                epoch_loss_G += loss_G
                
                # Print progress
                if i % 50 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}] Batch [{i}/{len(dataloader)}] "
                          f"Loss_D: {loss_D:.4f} Loss_G: {loss_G:.4f}")
            
            # Average losses
            avg_loss_D = epoch_loss_D / len(dataloader)
            avg_loss_G = epoch_loss_G / len(dataloader)
            self.losses_D.append(avg_loss_D)
            self.losses_G.append(avg_loss_G)
            
            print(f"Epoch [{epoch+1}/{epochs}] Avg Loss_D: {avg_loss_D:.4f} Avg Loss_G: {avg_loss_G:.4f}")
            
            # Save generated samples
            if (epoch + 1) % save_interval == 0:
                self.save_samples(epoch + 1)
    
    def save_samples(self, epoch):
        """Generate and save sample images"""
        self.generator.eval()
        with torch.no_grad():
            fake_imgs = self.generator(self.fixed_noise)
            fake_imgs = fake_imgs * 0.5 + 0.5  # Denormalize from [-1,1] to [0,1]
            
            # Save grid
            os.makedirs('samples', exist_ok=True)
            vutils.save_image(fake_imgs, f'samples/epoch_{epoch}.png', nrow=8, normalize=False)
        self.generator.train()
    
    def plot_losses(self):
        """Plot training losses"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.losses_D, label='Discriminator Loss')
        plt.plot(self.losses_G, label='Generator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('GAN Training Losses')
        plt.savefig('training_losses.png')
        plt.close()

# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def train_classification_model(model, train_loader, val_loader, criterion, optimizer, 
                                num_epochs=10, device='cuda', scheduler=None):
    """Standard training loop for classification"""
    model = model.to(device)
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = 100. * val_correct / val_total
        train_acc = 100. * train_correct / len(train_loader.dataset)
        
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        if scheduler:
            scheduler.step()
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    return model

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler

def train_with_mixed_precision(model, train_loader, criterion, optimizer, device='cuda'):
    """Training with automatic mixed precision"""
    scaler = GradScaler()
    model.train()
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

# Gradient accumulation
def train_with_gradient_accumulation(model, train_loader, criterion, optimizer, 
                                     accumulation_steps=4, device='cuda'):
    """Training with gradient accumulation for large batch sizes"""
    model.train()
    optimizer.zero_grad()
    
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss = loss / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

# ============================================================================
# EVALUATION METRICS
# ============================================================================

def calculate_metrics(predictions, targets, num_classes):
    """Calculate precision, recall, F1 for multi-class"""
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, average='weighted'
    )
    cm = confusion_matrix(targets, predictions)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

def calculate_iou(pred_mask, true_mask):
    """Calculate Intersection over Union for segmentation"""
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    return intersection / (union + 1e-8)

# ============================================================================
# GRAD-CAM for visualization
# ============================================================================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class):
        output = self.model(input_image)
        self.model.zero_grad()
        
        class_loss = output[0, target_class]
        class_loss.backward()
        
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activations).sum(dim=0)
        cam = F.relu(cam)
        cam = cam / cam.max()
        
        return cam.cpu().numpy()
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# ============================================================================
# 1. VANILLA CONVOLUTIONAL AUTOENCODER
# ============================================================================

class ConvEncoder(nn.Module):
    def __init__(self, img_channels=3, latent_dim=128):
        super(ConvEncoder, self).__init__()
        
        # Input: 3 x 64 x 64
        self.encoder = nn.Sequential(
            # 3 x 64 x 64 -> 32 x 32 x 32
            nn.Conv2d(img_channels, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # 32 x 32 x 32 -> 64 x 16 x 16
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 64 x 16 x 16 -> 128 x 8 x 8
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 128 x 8 x 8 -> 256 x 4 x 4
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        
        # Flatten to latent vector
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        z = self.fc(x)
        return z


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim=128, img_channels=3):
        super(ConvDecoder, self).__init__()
        
        # Expand from latent vector
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        
        # Output: 3 x 64 x 64
        self.decoder = nn.Sequential(
            # 256 x 4 x 4 -> 128 x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 128 x 8 x 8 -> 64 x 16 x 16
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 64 x 16 x 16 -> 32 x 32 x 32
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # 32 x 32 x 32 -> 3 x 64 x 64
            nn.ConvTranspose2d(32, img_channels, 4, stride=2, padding=1),
            nn.Sigmoid()  # Output range [0, 1]
        )
    
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 256, 4, 4)  # Reshape to spatial dimensions
        x = self.decoder(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, img_channels=3, latent_dim=128):
        super(Autoencoder, self).__init__()
        self.encoder = ConvEncoder(img_channels, latent_dim)
        self.decoder = ConvDecoder(latent_dim, img_channels)
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)


# ============================================================================
# 2. DENOISING AUTOENCODER
# ============================================================================

class DenoisingAutoencoder(nn.Module):
    """
    Learns to reconstruct clean images from noisy inputs
    More robust representations
    """
    def __init__(self, img_channels=3, latent_dim=128):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = ConvEncoder(img_channels, latent_dim)
        self.decoder = ConvDecoder(latent_dim, img_channels)
    
    def add_noise(self, x, noise_factor=0.3):
        """Add Gaussian noise to input"""
        noisy = x + noise_factor * torch.randn_like(x)
        noisy = torch.clamp(noisy, 0., 1.)
        return noisy
    
    def forward(self, x, add_noise=True):
        if add_noise and self.training:
            x_noisy = self.add_noise(x)
        else:
            x_noisy = x
        
        z = self.encoder(x_noisy)
        x_recon = self.decoder(z)
        return x_recon, z, x_noisy


# ============================================================================
# 3. VARIATIONAL AUTOENCODER (VAE)
# ============================================================================

class VAEEncoder(nn.Module):
    def __init__(self, img_channels=3, latent_dim=128):
        super(VAEEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        
        self.flatten = nn.Flatten()
        
        # Output mean and log variance
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class VariationalAutoencoder(nn.Module):
    """
    Learns probabilistic latent space
    Can generate new samples by sampling from latent distribution
    """
    def __init__(self, img_channels=3, latent_dim=128):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VAEEncoder(img_channels, latent_dim)
        self.decoder = ConvDecoder(latent_dim, img_channels)
        self.latent_dim = latent_dim
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + sigma * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
    
    def sample(self, num_samples, device):
        """Generate new samples from latent space"""
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decoder(z)
        return samples


# ============================================================================
# 4. SPARSE AUTOENCODER
# ============================================================================

class SparseAutoencoder(nn.Module):
    """
    Enforces sparsity in latent representation
    Learns more robust features
    """
    def __init__(self, img_channels=3, latent_dim=128, sparsity_weight=0.001):
        super(SparseAutoencoder, self).__init__()
        self.encoder = ConvEncoder(img_channels, latent_dim)
        self.decoder = ConvDecoder(latent_dim, img_channels)
        self.sparsity_weight = sparsity_weight
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        
        # L1 sparsity penalty
        sparsity_loss = self.sparsity_weight * torch.mean(torch.abs(z))
        
        return x_recon, z, sparsity_loss


# ============================================================================
# 5. DATASET PREPARATION
# ============================================================================

def get_dataloader(data_path, img_size=64, batch_size=128, normalize=True):
    """
    Create dataloader for training
    
    Dataset structure:
    data_path/
        class1/
            img1.jpg
            img2.jpg
        class2/
            img1.jpg
            img2.jpg
    """
    
    if normalize:
        # Normalize to [0, 1] for MSE loss and Sigmoid output
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ])
    else:
        # Normalize to [-1, 1] if using Tanh output
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    return dataloader


# ============================================================================
# 6. TRAINING LOOPS
# ============================================================================

class AutoencoderTrainer:
    def __init__(self, model, device, lr=0.001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.train_losses = []
    
    def train_step(self, images):
        images = images.to(self.device)
        
        self.optimizer.zero_grad()
        
        # Forward pass
        recon_images, z = self.model(images)
        
        # Calculate loss
        loss = self.criterion(recon_images, images)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, dataloader, epochs=100, save_interval=10):
        print(f"Starting training on {self.device}")
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            for i, (images, _) in enumerate(dataloader):
                loss = self.train_step(images)
                epoch_loss += loss
                
                if i % 50 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}] Batch [{i}/{len(dataloader)}] Loss: {loss:.6f}")
            
            # Average loss
            avg_loss = epoch_loss / len(dataloader)
            self.train_losses.append(avg_loss)
            print(f"Epoch [{epoch+1}/{epochs}] Avg Loss: {avg_loss:.6f}")
            
            # Save reconstructions
            if (epoch + 1) % save_interval == 0:
                self.save_reconstructions(dataloader, epoch + 1)
    
    def save_reconstructions(self, dataloader, epoch):
        """Save original vs reconstructed images"""
        self.model.eval()
        
        # Get a batch of images
        images, _ = next(iter(dataloader))
        images = images[:8].to(self.device)
        
        with torch.no_grad():
            recon_images, _ = self.model(images)
        
        # Concatenate original and reconstructed
        comparison = torch.cat([images, recon_images])
        
        os.makedirs('reconstructions', exist_ok=True)
        vutils.save_image(comparison, f'reconstructions/epoch_{epoch}.png', nrow=8)
        
        self.model.train()


class VAETrainer:
    def __init__(self, model, device, lr=0.001, beta=1.0):
        """
        beta: weight for KL divergence (beta-VAE)
        beta > 1: more disentangled representations
        beta < 1: better reconstructions
        """
        self.model = model.to(device)
        self.device = device
        self.beta = beta
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.train_losses = []
    
    def vae_loss(self, recon_x, x, mu, logvar):
        """
        VAE Loss = Reconstruction Loss + KL Divergence
        """
        # Reconstruction loss (MSE or BCE)
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        loss = recon_loss + self.beta * kl_div
        
        return loss, recon_loss, kl_div
    
    def train_step(self, images):
        images = images.to(self.device)
        
        self.optimizer.zero_grad()
        
        # Forward pass
        recon_images, mu, logvar = self.model(images)
        
        # Calculate loss
        loss, recon_loss, kl_div = self.vae_loss(recon_images, images, mu, logvar)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), recon_loss.item(), kl_div.item()
    
    def train(self, dataloader, epochs=100, save_interval=10):
        print(f"Starting VAE training on {self.device}")
        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_recon = 0
            epoch_kl = 0
            
            for i, (images, _) in enumerate(dataloader):
                loss, recon, kl = self.train_step(images)
                epoch_loss += loss
                epoch_recon += recon
                epoch_kl += kl
                
                if i % 50 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}] Batch [{i}/{len(dataloader)}]")
                    print(f"  Loss: {loss:.4f} Recon: {recon:.4f} KL: {kl:.4f}")
            
            # Average losses
            avg_loss = epoch_loss / len(dataloader)
            avg_recon = epoch_recon / len(dataloader)
            avg_kl = epoch_kl / len(dataloader)
            
            self.train_losses.append(avg_loss)
            
            print(f"\nEpoch [{epoch+1}/{epochs}] Summary:")
            print(f"  Avg Loss: {avg_loss:.4f}")
            print(f"  Avg Recon: {avg_recon:.4f}")
            print(f"  Avg KL: {avg_kl:.4f}\n")
            
            if (epoch + 1) % save_interval == 0:
                self.save_reconstructions(dataloader, epoch + 1)
                self.save_samples(epoch + 1)
    
    def save_reconstructions(self, dataloader, epoch):
        self.model.eval()
        images, _ = next(iter(dataloader))
        images = images[:8].to(self.device)
        
        with torch.no_grad():
            recon_images, _, _ = self.model(images)
        
        comparison = torch.cat([images, recon_images])
        os.makedirs('vae_reconstructions', exist_ok=True)
        vutils.save_image(comparison, f'vae_reconstructions/epoch_{epoch}.png', nrow=8)
        
        self.model.train()
    
    def save_samples(self, epoch):
        """Generate new samples from latent space"""
        self.model.eval()
        
        with torch.no_grad():
            samples = self.model.sample(64, self.device)
        
        os.makedirs('vae_samples', exist_ok=True)
        vutils.save_image(samples, f'vae_samples/epoch_{epoch}.png', nrow=8)
        
        self.model.train()


# ============================================================================
# 7. USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Hyperparameters
    LATENT_DIM = 128
    IMG_SIZE = 64
    IMG_CHANNELS = 3
    BATCH_SIZE = 128
    EPOCHS = 100
    LR = 0.001
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Choose model type
    print("\n=== Available Models ===")
    print("1. Vanilla Autoencoder")
    print("2. Denoising Autoencoder")
    print("3. Variational Autoencoder (VAE)")
    print("4. Sparse Autoencoder")
    
    # Example: Vanilla Autoencoder
    model = Autoencoder(img_channels=IMG_CHANNELS, latent_dim=LATENT_DIM)
    trainer = AutoencoderTrainer(model, device, lr=LR)
    
    # Example: VAE
    # model = VariationalAutoencoder(img_channels=IMG_CHANNELS, latent_dim=LATENT_DIM)
    # trainer = VAETrainer(model, device, lr=LR, beta=1.0)
    
    # Load data (replace with your dataset path)
    # dataloader = get_dataloader('path/to/your/dataset', IMG_SIZE, BATCH_SIZE)
    
    # Train
    # trainer.train(dataloader, epochs=EPOCHS)
    
    # Save model
    # torch.save(model.state_dict(), 'autoencoder.pth')
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")
image = pipe("a futuristic robot on mars").images[0]
image.save("out.png")
from PIL import Image

init = Image.open("input.png")

image = pipe.img2img(
    prompt="convert to watercolor style",
    image=init,
    strength=0.7
).images[0]
from diffusers import StableDiffusionInpaintPipeline
import torch

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting-1.5",
    torch_dtype=torch.float16
).to("cuda")

image = pipe(
    prompt="replace with a steel robot",
    image=init_image,
    mask_image=mask,
).images[0]
pipe = DiffusionPipeline.from_pretrained("stabilityai/sd-x2-latent-upscaler")
image = pipe(prompt="upscale", image=lowres).images[0]
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
)
from peft import LoraConfig, get_peft_model
from PIL import Image
from accelerate import Accelerator
import os
import json
import random

# ============================================================
# 1. CONFIG
# ============================================================
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = "./lora-output"
DATA_PATH = "./dataset"  # folder of images + captions.json

LR = 1e-4
BATCH_SIZE = 1
EPOCHS = 1
RANK = 8
IMAGE_SIZE = 512

# ============================================================
# 2. DATASET
# ============================================================

class Text2ImageDataset(Dataset):
    def __init__(self, image_folder, captions_file, size=512):
        self.size = size
        self.image_folder = image_folder
        with open(captions_file, "r") as f:
            self.prompts = json.load(f)   # {"IMG_1.png": "prompt text", ...}
        self.keys = list(self.prompts.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        fname = self.keys[idx]
        prompt = self.prompts[fname]

        img = Image.open(os.path.join(self.image_folder, fname)).convert("RGB")
        img = img.resize((self.size, self.size))
        img = torch.from_numpy((torch.tensor(img).permute(2, 0, 1) / 255.).numpy()).float()

        return {
            "pixel_values": img,
            "prompt": prompt
        }

# ============================================================
# 3. LOAD PIPELINE
# ============================================================

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16
)

vae = pipe.vae
tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder
unet = pipe.unet
noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

# ============================================================
# 4. INJECT LORA INTO UNET
# ============================================================

lora_config = LoraConfig(
    r=RANK,
    lora_alpha=16,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],  # SD-attn layers
    lora_dropout=0.05,
    bias="none",
    task_type="UNET_CAUSAL_LM"
)

unet = get_peft_model(unet, lora_config)
unet.print_trainable_parameters()

# ============================================================
# 5. DATA LOADER
# ============================================================

dataset = Text2ImageDataset(DATA_PATH, os.path.join(DATA_PATH, "captions.json"))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ============================================================
# 6. ACCELERATOR
# ============================================================

accelerator = Accelerator(
    mixed_precision="fp16",
    gradient_accumulation_steps=1
)

unet, text_encoder, optimizer, dataloader = accelerator.prepare(
    unet,
    text_encoder,
    torch.optim.AdamW(unet.parameters(), lr=LR),
    dataloader
)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
pipe.unet = unet  # register updated UNet

# ============================================================
# 7. TRAINING LOOP
# ============================================================

global_step = 0
for epoch in range(EPOCHS):
    for batch in dataloader:

        # ---- Encode prompt ----
        text = tokenizer(
            batch["prompt"],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_input_ids = text.input_ids.to(accelerator.device)
        encoder_hidden_states = text_encoder(text_input_ids)[0]

        # ---- Encode image to latents ----
        imgs = batch["pixel_values"].to(torch.float16).to(accelerator.device)
        imgs = imgs.unsqueeze(0) if imgs.ndim == 3 else imgs
        latents = vae.encode(imgs).latent_dist.sample() * 0.18215

        # ---- Add noise ----
        bsz = latents.shape[0]
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                  (bsz,), device=accelerator.device).long()

        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # ---- Predict noise via UNet ----
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        loss = F.mse_loss(model_pred, noise)

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        global_step += 1

        if accelerator.is_main_process and global_step % 20 == 0:
            print(f"Step {global_step} | Loss: {loss.item():.4f}")

# ============================================================
# 8. SAVE LORA WEIGHTS
# ============================================================

if accelerator.is_main_process:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    unet.save_pretrained(os.path.join(OUTPUT_DIR, "lora-unet"))
    print("LoRA saved to:", OUTPUT_DIR)

# ============================================================
# 9. LOAD FINETUNED LORA (FOR INFERENCE)
# ============================================================
"""
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to("cuda")
pipe.unet.load_attn_procs("./lora-output/lora-unet")
image = pipe("your prompt", num_inference_steps=25).images[0]
image.save("result.png")
"""
# ============================================================
#               YOLO OBJECT DETECTION CHEATSHEET
#      (Training, Inference, Validation, Custom Dataset)
# ============================================================
# This cheatsheet uses the ULTRALYTICS YOLOv8 API.
# Covers:
#   ✔ Training on custom dataset
#   ✔ Dataset YAML format example
#   ✔ Inference on images / videos / webcam
#   ✔ Exporting to ONNX / TensorRT
#   ✔ Tracking
#   ✔ Validation / metrics
#   ✔ Real-time streaming
#   ✔ Use-case comments above each block
# ============================================================

# Install YOLO if needed:
# pip install ultralytics

from ultralytics import YOLO

# ============================================================
# 1. LOAD MODEL (pretrained or custom)
# ============================================================
# Use-case: load a pretrained detection model (YOLOv8n = fastest)
model = YOLO("yolov8n.pt")  
# model = YOLO("yolov8s.pt")  # more accurate
# model = YOLO("runs/detect/train/weights/best.pt")  # load custom trained

# ============================================================
# 2. TRAIN ON CUSTOM DATASET
# ============================================================
# Use-case: train YOLO from scratch or finetune on your data.
# Requirements: a dataset.yaml file pointing to train/val images.
results = model.train(
    data="dataset.yaml",   # <-- your YAML file
    epochs=50,
    imgsz=640,
    batch=8,
    lr0=0.001,
    device=0,
)


# ============================================================
# 3. DATASET YAML EXAMPLE (PUT IN dataset.yaml)
# ============================================================
# Use-case: define custom dataset structure.
# Paste this YAML into dataset.yaml file:
"""
path: ./dataset                 # dataset root folder
train: images/train
val: images/val

# number of classes
nc: 3

# class names
names: ["person", "helmet", "smoke"]
"""


# ============================================================
# 4. BASIC INFERENCE (IMAGE)
# ============================================================
# Use-case: detect objects in a single image
results = model("test.jpg")
results[0].show()              # display detections
results[0].save("result.jpg")  # save result


# ============================================================
# 5. INFERENCE ON VIDEO
# ============================================================
# Use-case: detect objects on videos frame-by-frame
model.predict(
    source="test_video.mp4",
    save=True,
    conf=0.25,
    device=0
)


# ============================================================
# 6. REAL-TIME WEBCAM DETECTION
# ============================================================
# Use-case: live camera object detection (0 = default webcam)
model.predict(
    source=0,
    show=True,
    conf=0.35,
    device=0
)


# ============================================================
# 7. EXPORT MODEL (ONNX / TensorRT / CoreML)
# ============================================================
# Use-case: deploy YOLO model into mobile, cloud, embedded devices.
model.export(format="onnx")      # ONNX for ML frameworks
model.export(format="engine")    # NVIDIA TensorRT
model.export(format="coreml")    # Apple devices
model.export(format="tflite")    # Android/Edge
model.export(format="openvino")  # Intel hardware


# ============================================================
# 8. TRACKING (BYTETrack, StrongSORT, etc.)
# ============================================================
# Use-case: track objects across multiple frames in video.
model.track(
    source="street.mp4",
    save=True,
    tracker="bytetrack.yaml",   # built-in tracker
    conf=0.4
)


# ============================================================
# 9. VALIDATION (mAP, Precision, Recall)
# ============================================================
# Use-case: evaluate the trained model on validation set.
metrics = model.val(
    data="dataset.yaml",
    conf=0.25,
)
print(metrics)


# ============================================================
# 10. CUSTOM TRAINING TRICKS FOR IOAI
# ============================================================
# Use-case: apply advanced training options
model.train(
    data="dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    lr0=5e-4,                # lower LR for finetuning
    mosaic=0.7,              # augmentation
    hsv_h=0.015,             # color augmentation
    fliplr=0.5,              # left-right flip
    mixup=0.2,
    box=7.5, cls=2.0, dfl=1.5,  # loss weights
    device=0
)


# ============================================================
# 11. PYTHONIC LOW-LEVEL INFERENCE (get boxes programmatically)
# ============================================================
# Use-case: get bounding boxes, class IDs, scores inside code.
results = model("image.jpg")[0]

for box in results.boxes:
    xyxy = box.xyxy[0].tolist()    # [x1, y1, x2, y2]
    conf  = float(box.conf[0])     # confidence
    cls   = int(box.cls[0])        # class index
    print("Box:", xyxy, "Conf:", conf, "Class:", cls)


# ============================================================
# 12. SAVE RESULTS TO CSV / JSON
# ============================================================
# Use-case: export detections for competitions or analytics.
import pandas as pd

rows = []
for box in results.boxes:
    rows.append({
        "x1": float(box.xyxy[0][0]),
        "y1": float(box.xyxy[0][1]),
        "x2": float(box.xyxy[0][2]),
        "y2": float(box.xyxy[0][3]),
        "conf": float(box.conf[0]),
        "class": int(box.cls[0])
    })

pd.DataFrame(rows).to_csv("detections.csv", index=False)


# ============================================================
# 13. TRAIN FROM ZERO (NO PRETRAINED)
# ============================================================
# Use-case: fully custom-trained detector for specialized domain.
model = YOLO("yolov8n.yaml")  # architecture only
model.train(
    data="dataset.yaml",
    epochs=200,
    imgsz=640,
)


# ============================================================
# 14. CALLBACKS (CUSTOM LOGIC DURING TRAINING)
# ============================================================
# Use-case: save custom metrics, early stopping, etc.
class MyCallback:
    def on_fit_epoch_end(self, trainer):
        print("Epoch:", trainer.epoch)

model.add_callback("on_fit_epoch_end", MyCallback())
import os
import cv2
import zipfile
import random
import numpy as np
from glob import glob
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# --- Configuration Dictionary ---
CONFIG = {
    "IMAGE_SIZE": 512, 
    "BATCH_SIZE": 8, 
    "EPOCHS": 40, 
    "LEARNING_RATE": 2e-4, 
    "ENCODER_NAME": "efficientnet-b4", 
    "ENCODER_WEIGHTS": "imagenet", 
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu", 
    "SEED_VALUE": 42
}

def set_global_seed(seed_value):
    """Sets seeds for reproducibility."""
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True

set_global_seed(CONFIG["SEED_VALUE"])

# --- Dataset Class for Segmentation ---
class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, apply_augmentation=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = CONFIG["IMAGE_SIZE"]
        
        # Define Albumentations transformations
        if apply_augmentation:
            self.transform = A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.OneOf([
                    A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
                ], p=0.3),
                A.CoarseDropout(max_holes=8, max_height=self.image_size//20, max_width=self.image_size//20, fill_value=0, p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.mask_paths: # Training/Validation mode
            mask = cv2.imread(self.mask_paths[index], cv2.IMREAD_GRAYSCALE)
            # Convert mask to binary float tensor (0 or 1)
            mask = (mask > 127).astype("float32") 
            
            augmented = self.transform(image=image, mask=mask)
            # Add channel dimension to mask (C, H, W)
            return augmented["image"], augmented["mask"].unsqueeze(0) 
        else: # Inference/Test mode
            original_height, original_width = image.shape[:2]
            augmented = self.transform(image=image)
            # Return image, original path, and original dimensions
            return augmented["image"], self.image_paths[index], (original_height, original_width) 

# --- Training Function ---
def train_model(train_loader, validation_loader):
    # Initialize UnetPlusPlus model
    model = smp.UnetPlusPlus(
        encoder_name=CONFIG["ENCODER_NAME"], 
        encoder_weights=CONFIG["ENCODER_WEIGHTS"], 
        in_channels=3, 
        classes=1, 
        activation=None
    ).to(CONFIG["DEVICE"])
    
    # Define Loss Functions
    dice_loss_fn = smp.losses.DiceLoss(mode="binary")
    bce_loss_fn = nn.BCEWithLogitsLoss()
    
    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["LEARNING_RATE"], weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["EPOCHS"], eta_min=1e-6)
    
    # GradScaler for Mixed Precision Training
    scaler = GradScaler()
    best_loss = float('inf')
    best_model_path = "best_model.pth"

    for epoch in range(CONFIG["EPOCHS"]):
        model.train()
        train_loss_sum = 0
        train_loop = tqdm(train_loader)
        
        # Training loop
        for images, masks in train_loop:
            images, masks = images.to(CONFIG["DEVICE"]), masks.to(CONFIG["DEVICE"])
            optimizer.zero_grad()
            
            with autocast(): # Mixed precision
                predictions = model(images)
                # Combined Loss: 0.5 * Dice Loss + 0.5 * BCE Loss
                loss = 0.5 * dice_loss_fn(torch.sigmoid(predictions), masks) + 0.5 * bce_loss_fn(predictions, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss_sum += loss.item()
        
        # Validation loop
        model.eval()
        validation_loss_sum = 0
        with torch.no_grad():
            for images, masks in validation_loader:
                images, masks = images.to(CONFIG["DEVICE"]), masks.to(CONFIG["DEVICE"])
                with autocast():
                    predictions = model(images)
                    loss = 0.5 * dice_loss_fn(torch.sigmoid(predictions), masks) + 0.5 * bce_loss_fn(predictions, masks)
                validation_loss_sum += loss.item()
        
        average_validation_loss = validation_loss_sum / len(validation_loader)
        
        # Print epoch results
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss_sum/len(train_loader):.4f} | Validation Loss: {average_validation_loss:.4f}")
        
        # Scheduler step and Checkpoint saving
        scheduler.step()
        if average_validation_loss < best_loss:
            best_loss = average_validation_loss
            torch.save(model.state_dict(), best_model_path)
            
    return best_model_path

# --- Prediction Function ---
def predict_masks(model_path, test_directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    
    # Initialize model for inference
    model = smp.UnetPlusPlus(
        encoder_name=CONFIG["ENCODER_NAME"], 
        encoder_weights=None, 
        in_channels=3, 
        classes=1
    ).to(CONFIG["DEVICE"])
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Prepare DataLoader
    image_paths = sorted(glob(f"{test_directory}/*.png"))
    test_dataset = SegmentationDataset(image_paths, apply_augmentation=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        for images, original_path, (original_height, original_width) in tqdm(test_loader):
            images = images.to(CONFIG["DEVICE"])
            
            # Test-Time Augmentation (TTA): Original, Horizontal Flip, Vertical Flip
            prediction_original = torch.sigmoid(model(images))
            
            prediction_hflip = torch.sigmoid(model(torch.flip(images, [3])))
            prediction_hflip = torch.flip(prediction_hflip, [3]) # Flip back
            
            prediction_vflip = torch.sigmoid(model(torch.flip(images, [2])))
            prediction_vflip = torch.flip(prediction_vflip, [2]) # Flip back
            
            # Average predictions from TTA
            average_prediction = (prediction_original + prediction_hflip + prediction_vflip) / 3.0
            
            # Post-processing
            mask_np = average_prediction[0, 0].cpu().numpy()
            
            # Resize back to original image dimensions
            resized_mask = cv2.resize(mask_np, (original_width.item(), original_height.item()))
            
            # Thresholding and saving as 8-bit image (0 or 255)
            final_mask = (resized_mask > 0.5).astype("uint8") * 255
            
            # Save the mask
            image_filename = os.path.basename(original_path[0])
            output_filepath = os.path.join(output_directory, image_filename)
            cv2.imwrite(output_filepath, final_mask)

# --- Zipping Function ---
def create_submission_zip(masks_directory, zip_filename="submission.zip"):
    """Creates a zip file containing all predicted masks."""
    with zipfile.ZipFile(zip_filename, 'w') as zip_file:
        for filepath in sorted(glob(masks_directory + "/*.png")):
            # Write the file using only its base name
            zip_file.write(filepath, os.path.basename(filepath))

# --- Main Execution Block ---
if __name__ == "__main__":
    # 1. Prepare data paths
    train_image_paths = sorted(glob("train/images/*.png"))
    train_mask_paths = sorted(glob("train/masks/*.png"))
    
    # 2. Split data into train and validation sets (90/10 split)
    split_index = int(0.9 * len(train_image_paths))
    
    tr_images = train_image_paths[:split_index]
    tr_masks = train_mask_paths[:split_index]
    vl_images = train_image_paths[split_index:]
    vl_masks = train_mask_paths[split_index:]
    
    # 3. Create Dataset and DataLoader instances
    train_dataset = SegmentationDataset(tr_images, tr_masks, apply_augmentation=True)
    validation_dataset = SegmentationDataset(vl_images, vl_masks, apply_augmentation=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG["BATCH_SIZE"], 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )
    validation_loader = DataLoader(
        validation_dataset, 
        batch_size=CONFIG["BATCH_SIZE"], 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )
    
    # 4. Train the model
    best_model_path = train_model(train_loader, validation_loader)
    
    # 5. Predict on test images
    predict_masks(best_model_path, "test/images", "preds")
    
    # 6. Create submission zip file
    create_submission_zip("preds")
!pip install transformers -q
import os
import glob
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel

# ==========================================
# 1. CONFIGURATION
# ==========================================
TRAIN_DIR = "/kaggle/input/iaio-2026-sf-r-image-classification/train_img/train"
TEST_DIR = "/kaggle/input/iaio-2026-sf-r-image-classification/test_img/test"
SUBMISSION_FILE = "submission.csv"

# Using the high-resolution Large model from OpenAI
MODEL_ID = "openai/clip-vit-large-patch14-336"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Running on {DEVICE} using {MODEL_ID}")

# ==========================================
# 2. DATA PREPARATION
# ==========================================

# Get Class Names
class_names = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
print(f"Classes found: {len(class_names)}")

# Load HuggingFace Model and Processor
model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE)
processor = CLIPProcessor.from_pretrained(MODEL_ID)

# Custom Test Dataset
class TestDataset(Dataset):
    def __init__(self, folder_path):
        self.filepaths = glob.glob(os.path.join(folder_path, "*.*"))
        
    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        image = Image.open(path).convert("RGB")
        return image, os.path.basename(path)

# Collate function to handle the processor (resizing/normalization) inside the loader
def collate_fn(batch):
    images = [item[0] for item in batch]
    filenames = [item[1] for item in batch]
    # Processor handles resizing and normalization automatically
    inputs = processor(images=images, return_tensors="pt")
    return inputs['pixel_values'], filenames

test_dataset = TestDataset(TEST_DIR)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn)

# ==========================================
# 3. THE CHERRY: PROMPT ENSEMBLING
# ==========================================
print("Building Text Prototypes with Prompt Ensembling...")

# Multiple templates to capture different aspects of the images
templates = [
    "a photo of the district of {}.",
    "a street view in {}.",
    "architecture in {}.",
    "a driving scenario in {}.",
    "buildings and roads in {}.",
    "this is a photo of some area, street and district in {}."
]

def build_text_prototypes(classes, templates):
    """
    Generates an averaged text embedding for each class based on multiple templates.
    """
    model.eval()
    zeroshot_weights = []

    with torch.no_grad():
        for classname in tqdm(classes, desc="Encoding Classes"):
            # 1. Fill templates for this specific class
            texts = [template.format(classname) for template in templates]
            
            # 2. Tokenize
            inputs = processor(text=texts, padding=True, return_tensors="pt").to(DEVICE)
            
            # 3. Get Embeddings for all templates of this class
            class_embeddings = model.get_text_features(**inputs)
            
            # 4. Normalize individual template embeddings
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            
            # 5. Average them to get the single "Prototype" for this class
            class_embedding = class_embeddings.mean(dim=0)
            
            # 6. Normalize the final prototype
            class_embedding /= class_embedding.norm()
            
            zeroshot_weights.append(class_embedding)
            
    # Stack into a matrix: [Num_Classes, Embed_Dim]
    return torch.stack(zeroshot_weights).T

# Create the text features matrix
# Shape: [Embed_Dim, Num_Classes]
text_features = build_text_prototypes(class_names, templates)

# ==========================================
# 4. INFERENCE LOOP
# ==========================================
print("Starting Inference...")

filenames_list = []
predictions_list = []

model.eval()
with torch.no_grad():
    for images, filenames in tqdm(test_loader):
        images = images.to(DEVICE)
        
        # 1. Get Image Features
        image_features = model.get_image_features(pixel_values=images)
        
        # 2. Normalize Image Features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # 3. Calculate Cosine Similarity
        # (Batch, Dim) @ (Dim, Classes) -> (Batch, Classes)
        similarity = (100.0 * image_features @ text_features)
        
        # 4. Get Top Prediction
        # softmax is optional here, argmax works the same for hard classification
        _, preds = similarity.topk(1, dim=-1)
        
        predictions_list.extend(preds.squeeze().cpu().numpy())
        filenames_list.extend(filenames)

# ==========================================
# 5. SUBMISSION
# ==========================================

# Map indices back to class names
final_labels = [class_names[i] for i in predictions_list]

submission = pd.DataFrame({
    'path': filenames_list,
    'labels': final_labels
})

submission.to_csv(SUBMISSION_FILE, index=False)
print(f"Submission saved to {SUBMISSION_FILE}")
print(submission.head())
import segmentation_models_pytorch as smp

model = smp.UnetPlusPlus(
    encoder_name="efficientnet-b4", 
    encoder_weights="imagenet",     # Pre-trained on ImageNet
    in_channels=3,                  # RGB
    classes=1,                      # Binary (e.g., mask vs background)
)
# Note: Ensure 'timm' library is installed for PVT encoders
model = smp.MAnet(
    encoder_name="pvt_v2_b2", 
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
)
# Pair: MAnet + pvt_v2_b2 (Transformer Encoder)
# Why: MAnet uses Multi-scale Attention to capture global dependencies. Pairing it with a Pyramid Vision Transformer (PVT) allows the model to "see" long-range relationships in tissues or cells better than standard CNNs.
# Best for: Cell nuclei, tiny tumors, or subtle defects in manufacturing.
model = smp.DeepLabV3Plus(
    encoder_name="resnet101", 
    encoder_weights="imagenet",
    in_channels=3,
    classes=1, 
)
# The "Cityscape/Scene Master" (Global Context)
# Pair: DeepLabV3Plus + resnet101
# Why: DeepLabV3+ uses Atrous (dilated) convolutions. It’s designed to understand "where things are" relative to each other. ResNet101 provides the depth needed to understand complex urban environments.
# Best for: Self-driving datasets, urban planning (buildings, roads, sidewalks), and large-scale scene parsing.
model = smp.FPN(
    encoder_name="resnext50_32x4d", 
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
)
# The "Multi-Scale Speedster" (Object Variation)
# Pair: FPN + resnext50_32x4d
# Why: FPN (Feature Pyramid Network) is natively designed to handle objects of wildly different sizes. ResNext is more efficient than ResNet due to "grouped convolutions," making it fast for large image inputs (e.g., 1024x1024).
# Best for: Satellite imagery (detecting both huge forests and tiny ships), or X-ray scans with varying fracture sizes.
model = smp.PAN(
    encoder_name="mit_b3", 
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
)
# The "High-Resolution Transformer" (SOTA)
# Pair: PAN + mit_b3 (SegFormer Encoder)
# Why: PAN (Pyramid Attention Network) is lightweight but powerful. mit_b3 (Mix Transformer) is the backbone of SegFormer. This combination is currently top-tier for performance-to-parameters ratio.
# Best for: High-resolution images where you want Transformer performance without the massive VRAM cost of a full ViT.
# If your data is...	Use this Architecture	Use this Encoder
# Very small objects	UnetPlusPlus	efficientnet-b5
# Large landscapes	DeepLabV3Plus	resnet50 / resnet101
# Multi-scale objects	FPN	resnext50
# Textured/Noisy	MAnet	se_resnet50 (Squeeze-and-Excitation)
# Limited VRAM	Unet	efficientnet-b0 / mobilenet_v2
"""
Hogspell Challenge: Complete Solution
Transforms horses to pigs in Stable Diffusion generated images
Author: AdilzhanB
Date: 2025-08-18
"""

import os
import json
import random
import hashlib
import pathlib
import warnings
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
import base64

from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# Configuration
CONFIG = {
    'seed': 12782637,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model_name': 'runwayml/stable-diffusion-v1-5',
    'clip_model_name': 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
    'output_dir': './hogspell_output',
    'model_dir': './hogspell_model',
    'batch_size': 8,
    'num_inference_steps': 50,
    'guidance_scale': 7.5,
    'learning_rate': 1e-5,
    'weight_decay': 1e-4,
    'num_epochs': 3,
    'image_size': 512,
}

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed {seed} set for reproducibility")

def image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """Convert PIL Image to base64 string"""
    buf = BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def base64_to_image(base64_str: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    img_data = base64.b64decode(base64_str)
    img = Image.open(BytesIO(img_data))
    return img

class HogspellDataset:
    """Dataset preparation for Hogspell Challenge"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def get_training_prompts(self) -> Dict[str, List[str]]:
        """Generate comprehensive training prompts"""
        
        # Base pig prompts with various styles
        pig_prompts = [
            "realistic photo of a pig",
            "watercolor painting of a pig",
            "oil painting of a pig", 
            "3D render of a pig",
            "cartoon pig",
            "impressionist pig painting",
            "pig portrait",
            "pig in a farm",
            "cute pig",
            "pig eating",
            "pig family",
            "pig in mud",
        ]
        
        # Corresponding horse prompts
        horse_prompts = [prompt.replace("pig", "horse") for prompt in pig_prompts]
        
        # Edge cases that might generate horses without explicit "horse" mention
        edge_horse_prompts = [
            "cowboy riding his mount",
            "knight on his steed",
            "racing at the derby",
            "farm animals in a stable",
            "equestrian competition",
            "wild mustang",
            "mare with foal",
            "stallion galloping",
        ]
        
        # Neutral prompts (should not generate horses or pigs)
        neutral_prompts = [
            "mountain landscape",
            "city skyline at sunset",
            "ocean waves",
            "forest in autumn",
            "cat sitting on a chair",
            "dog playing in park",
            "bird flying in sky",
            "flower garden",
            "car on highway",
            "abstract art",
        ]
        
        return {
            'pig': pig_prompts,
            'horse': horse_prompts + edge_horse_prompts,
            'neutral': neutral_prompts
        }

class HogspellTrainer:
    """Advanced trainer for Hogspell Challenge"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = config['device']
        
        # Load models
        self.pipe = self._load_pipeline()
        self.clip_model, self.clip_processor = self._load_clip()
        
        # Image transforms
        self.image_transforms = transforms.Compose([
            transforms.Resize((config['image_size'], config['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
    def _load_pipeline(self) -> StableDiffusionPipeline:
        """Load Stable Diffusion pipeline"""
        print("Loading Stable Diffusion v1.5...")
        pipe = StableDiffusionPipeline.from_pretrained(
            self.config['model_name'],
            torch_dtype=torch.float32,
            safety_checker=None,
        ).to(self.device)
        return pipe
        
    def _load_clip(self) -> Tuple[CLIPModel, CLIPProcessor]:
        """Load CLIP model for evaluation"""
        print("Loading CLIP model...")
        clip_model = CLIPModel.from_pretrained(self.config['clip_model_name']).eval().to(self.device)
        clip_processor = CLIPProcessor.from_pretrained(self.config['clip_model_name'])
        return clip_model, clip_processor
    
    def generate_training_data(self, prompts: Dict[str, List[str]]) -> Dict[str, List[Image.Image]]:
        """Generate training images"""
        print("Generating training data...")
        training_images = {}
        
        for category, prompt_list in prompts.items():
            if category == 'neutral':
                continue  # Skip neutral for training data generation
                
            images = []
            for i, prompt in enumerate(tqdm(prompt_list, desc=f"Generating {category} images")):
                image = self.pipe(
                    prompt=prompt,
                    num_inference_steps=self.config['num_inference_steps'],
                    guidance_scale=self.config['guidance_scale'],
                    generator=torch.Generator(device=self.device).manual_seed(self.config['seed'] + i)
                ).images[0]
                images.append(image)
                
                # Save image
                image_path = os.path.join(self.config['output_dir'], f"{category}_{i:03d}.png")
                image.save(image_path)
            
            training_images[category] = images
            
        return training_images
    
    def compute_transformation_loss(self, horse_prompt: str, pig_image: Image.Image) -> torch.Tensor:
        """Compute loss for horse-to-pig transformation"""
        # Convert pig image to tensor and encode to latent space
        pig_tensor = self.image_transforms(pig_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            pig_latent = self.pipe.vae.encode(pig_tensor).latent_dist.sample() * 0.18215
        
        # Encode horse prompt
        with torch.no_grad():
            text_input = self.pipe.tokenizer(
                horse_prompt,
                padding="max_length",
                max_length=self.pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_embeddings = self.pipe.text_encoder(text_input.input_ids.to(self.device))[0]
        
        # Add noise for training
        noise = torch.randn_like(pig_latent)
        timesteps = torch.randint(
            0, self.pipe.scheduler.config.num_train_timesteps, (1,), device=self.device
        )
        noisy_latent = self.pipe.scheduler.add_noise(pig_latent, noise, timesteps)
        
        # Predict noise using UNet
        self.pipe.unet.train()
        predicted_noise = self.pipe.unet(noisy_latent, timesteps, text_embeddings).sample
        
        # MSE loss
        loss = F.mse_loss(predicted_noise, noise)
        return loss
    
    def compute_neutral_regularization(self, neutral_prompts: List[str], lambda_reg: float = 0.1) -> torch.Tensor:
        """Regularization to prevent pig generation on neutral prompts"""
        total_loss = 0.0
        count = 0
        
        for prompt in neutral_prompts[:3]:  # Use subset for efficiency
            # Generate with current model
            with torch.no_grad():
                image = self.pipe(
                    prompt=prompt,
                    num_inference_steps=20,  # Fewer steps for regularization
                    guidance_scale=self.config['guidance_scale'],
                    generator=torch.Generator(device=self.device).manual_seed(self.config['seed'])
                ).images[0]
            
            # Check if image contains pig using CLIP
            inputs = self.clip_processor(
                text=["pig", "neutral object", ""],
                images=[image],
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                logits = self.clip_model(**inputs).logits_per_image.squeeze(0)
                pig_prob = F.softmax(logits, dim=0)[0]  # Probability of being a pig
            
            # Penalize pig probability
            reg_loss = lambda_reg * pig_prob
            total_loss += reg_loss
            count += 1
        
        return total_loss / max(count, 1)
    
    def train(self, training_data: Dict[str, List[Image.Image]], prompts: Dict[str, List[str]]):
        """Main training loop"""
        print("Starting advanced fine-tuning...")
        
        # Freeze VAE and text encoder, train only UNet
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)
        self.pipe.unet.requires_grad_(True)
        
        # Setup optimizer and scheduler
        optimizer = AdamW(
            self.pipe.unet.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=50)
        
        # Training loop
        for epoch in range(self.config['num_epochs']):
            print(f"Epoch {epoch + 1}/{self.config['num_epochs']}")
            
            epoch_loss = 0.0
            batch_count = 0
            
            # Iterate through horse-pig pairs
            horse_prompts = prompts['horse']
            pig_images = training_data['pig']
            
            for i, (horse_prompt, pig_image) in enumerate(zip(horse_prompts, pig_images)):
                # Primary transformation loss
                transform_loss = self.compute_transformation_loss(horse_prompt, pig_image)
                
                # Regularization loss (every few iterations to save compute)
                if i % 3 == 0:
                    reg_loss = self.compute_neutral_regularization(prompts['neutral'])
                else:
                    reg_loss = torch.tensor(0.0, device=self.device)
                
                # Combined loss
                total_loss = transform_loss + reg_loss
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.pipe.unet.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                epoch_loss += total_loss.item()
                batch_count += 1
                
                if (i + 1) % 5 == 0:
                    print(f"  Step {i+1}/{len(horse_prompts)}, Loss: {total_loss.item():.4f}")
            
            avg_loss = epoch_loss / batch_count
            print(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")
        
        # Save model
        print("Saving fine-tuned model...")
        os.makedirs(self.config['model_dir'], exist_ok=True)
        self.pipe.save_pretrained(self.config['model_dir'])
        print(f"Model saved to {self.config['model_dir']}")

class HogspellEvaluator:
    """Evaluation pipeline for Hogspell Challenge"""
    
    def __init__(self, pipe: StableDiffusionPipeline, clip_model: CLIPModel, clip_processor: CLIPProcessor):
        self.pipe = pipe
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.device = pipe.device
    
    @torch.no_grad()
    def evaluate_image(self, image: Image.Image, prompt_type: str, original_prompt: str = "") -> bool:
        """Evaluate single image using CLIP"""
        if prompt_type == 'horse':
            classes = ["pig", "horse", ""]
            target_class = 0  # Should predict pig
        elif prompt_type == 'pig':
            classes = ["pig", "horse", ""]
            target_class = 0  # Should predict pig
        else:  # neutral
            classes = [original_prompt, "pig", "horse", ""]
            target_class = 0  # Should predict original prompt or neutral
        
        inputs = self.clip_processor(
            text=classes,
            images=[image],
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        logits = self.clip_model(**inputs).logits_per_image.squeeze(0)
        predicted_class = logits.argmax().item()
        
        return predicted_class == target_class
    
    def validate_model(self, test_prompts: Dict[str, List[str]]) -> Tuple[float, Dict[str, float]]:
        """Comprehensive model validation"""
        print("Validating model performance...")
        
        results = {}
        
        for prompt_type, prompts in test_prompts.items():
            correct = 0
            total = len(prompts)
            
            for prompt in tqdm(prompts, desc=f"Evaluating {prompt_type} prompts"):
                # Generate image
                image = self.pipe(
                    prompt=prompt,
                    num_inference_steps=CONFIG['num_inference_steps'],
                    guidance_scale=CONFIG['guidance_scale'],
                    generator=torch.Generator(device=self.device).manual_seed(CONFIG['seed'])
                ).images[0]
                
                # Evaluate
                is_correct = self.evaluate_image(image, prompt_type, prompt)
                if is_correct:
                    correct += 1
            
            accuracy = correct / total
            results[f'{prompt_type}_acc'] = accuracy
            print(f"{prompt_type.capitalize()} accuracy: {accuracy:.3f}")
        
        # Calculate harmonic mean
        accuracies = [results['horse_acc'], results['pig_acc'], results['neutral_acc']]
        harmonic_mean = self._harmonic_mean(accuracies)
        
        print(f"Final Score (Harmonic Mean): {harmonic_mean:.3f}")
        return harmonic_mean, results
    
    def _harmonic_mean(self, values: List[float]) -> float:
        """Calculate harmonic mean"""
        positive_values = [v for v in values if v > 0]
        if not positive_values:
            return 0.0
        return len(positive_values) / sum(1 / v for v in positive_values)

def generate_submission(pipe: StableDiffusionPipeline, prompts: Dict[str, str], config: Dict) -> pd.DataFrame:
    """Generate final submission"""
    print("Generating submission...")
    
    pipe.to(config['device'])
    if pipe.safety_checker is not None:
        pipe.safety_checker = lambda imgs, **kw: (imgs, False)
    
    ids, images_b64 = [], []
    
    # Process prompts in batches
    prompt_items = list(prompts.items())
    batch_size = config['batch_size']
    
    for i in range(0, len(prompt_items), batch_size):
        batch_items = prompt_items[i:i+batch_size]
        batch_prompts = [item[1] for item in batch_items]
        batch_ids = [item[0] for item in batch_items]
        
        # Generate unique seeds for each prompt
        seeds = [config['seed'] + hash(prompt_id) % 1000000 for prompt_id in batch_ids]
        
        with torch.autocast(device_type=str(config['device'])):
            generated_images = []
            for prompt, seed in zip(batch_prompts, seeds):
                image = pipe(
                    prompt=prompt,
                    num_inference_steps=config['num_inference_steps'],
                    guidance_scale=config['guidance_scale'],
                    generator=torch.Generator(device=config['device']).manual_seed(seed)
                ).images[0]
                generated_images.append(image)
        
        # Convert to base64
        for prompt_id, image in zip(batch_ids, generated_images):
            ids.append(prompt_id)
            images_b64.append(image_to_base64(image))
        
        print(f"Generated {i + len(batch_items)} / {len(prompt_items)}")
    
    return pd.DataFrame({"id": ids, "0": images_b64})

def save_submission(df: pd.DataFrame):
    """Save submission with metadata"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    df_sha = hashlib.sha256(csv_bytes).hexdigest()
    
    submit_dir = pathlib.Path("submissions") / f"hogspell_{timestamp}_{df_sha[:8]}"
    submit_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = submit_dir / f"submission_{df_sha[:8]}.csv"
    csv_path.write_bytes(csv_bytes)
    
    # Save metadata
    meta = {
        "generated_at": timestamp,
        "dataframe_sha256": df_sha,
        "config": CONFIG,
        "author": "AdilzhanB"
    }
    
    (submit_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), 
        encoding="utf-8"
    )
    
    print(f"Submission saved: {csv_path}")
    print(f"Metadata saved: {submit_dir / 'metadata.json'}")

def main():
    """Main execution pipeline"""
    print("🐷 Starting Hogspell Challenge Solution 🐷")
    print(f"Device: {CONFIG['device']}")
    
    # Set seed for reproducibility
    set_seed(CONFIG['seed'])
    
    # Initialize components
    dataset = HogspellDataset(CONFIG['output_dir'])
    trainer = HogspellTrainer(CONFIG)
    
    # Prepare training data
    print("\n📊 Preparing training data...")
    training_prompts = dataset.get_training_prompts()
    training_images = trainer.generate_training_data(training_prompts)
    
    # Train model
    print("\n🎯 Training model...")
    trainer.train(training_images, training_prompts)
    
    # Validate model
    print("\n✅ Validating model...")
    evaluator = HogspellEvaluator(trainer.pipe, trainer.clip_model, trainer.clip_processor)
    
    # Create validation set (subset of training prompts)
    validation_prompts = {
        'horse': training_prompts['horse'][:5],
        'pig': training_prompts['pig'][:5], 
        'neutral': training_prompts['neutral'][:5]
    }
    
    score, detailed_results = evaluator.validate_model(validation_prompts)
    
    # Load competition prompts and generate submission
    print("\n🚀 Generating final submission...")
    
    # Load prompts (assuming they're in prompts.json)
    try:
        with open("prompts.json", "r") as f:
            competition_prompts = json.load(f)
        
        submission_df = generate_submission(trainer.pipe, competition_prompts, CONFIG)
        save_submission(submission_df)
        
        print(f"\n🎉 Solution completed!")
        print(f"Validation Score: {score:.3f}")
        print(f"Horse->Pig Accuracy: {detailed_results['horse_acc']:.3f}")
        print(f"Pig Preservation: {detailed_results['pig_acc']:.3f}")
        print(f"Neutral Preservation: {detailed_results['neutral_acc']:.3f}")
        
    except FileNotFoundError:
        print("⚠️  prompts.json not found. Please provide competition prompts file.")
        print("Creating sample submission with validation prompts...")
        
        # Create sample prompts for demonstration
        sample_prompts = {}
        for i, prompt_list in enumerate([validation_prompts['horse'], validation_prompts['pig'], validation_prompts['neutral']]):
            for j, prompt in enumerate(prompt_list):
                sample_prompts[f"{i}_{j}"] = prompt
        
        submission_df = generate_submission(trainer.pipe, sample_prompts, CONFIG)
        save_submission(submission_df)

if __name__ == "__main__":
    main()
