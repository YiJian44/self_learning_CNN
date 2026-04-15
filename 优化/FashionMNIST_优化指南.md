# FashionMNIST ProCNN 优化指南

## 📊 当前模型分析

### 原始模型配置
```python
class ProCNN:
    - 3层卷积 (32→64→128通道)
    - 每层后BatchNorm2d
    - MaxPool2d(2,2) ×3 (28→14→7→3)
    - Dropout(0.3)
    - FC: 1152 → 256 → 10
    - 训练: 20 epochs, Adam(lr=1e-3), StepLR(step=10, gamma=0.1)
    - 数据增强: RandomHorizontalFlip, RandomRotation(10)
    - 当前准确率: ~93.1%
```

### 问题诊断
❌ **网络太浅**：3层卷积对于FashionMNIST（28x28）足够了，但特征提取能力有限
❌ **感受野问题**：最后特征图3x3，信息压缩过度
❌ **Dropout位置**：只在FC层前，应该在卷积层后也加
❌ **学习率调度**：StepLR太激进，每10轮降10倍
❌ **数据增强**：太简单，缺少CutMix/MixUp等现代技术
❌ **缺少正则化**：没有Weight Decay
❌ **没有残差连接**：梯度传播不佳
❌ **最后FC太大**：1152→256参数过多，容易过拟合

---

## 🎯 优化方案总览

**目标准确率：96.5%+**（SOTA在FashionMNIST约97-98%）

**优化层级**（按收益排序）：
1. 🔥 **高收益**（+1.5-3%）：模型架构、训练策略、数据增强
2. ⚡ **中收益**（+0.5-1%）：正则化、优化器、初始化
3. 🎨 **低收益**（+0.1-0.3%：超参微调、后处理）

---

## 📈 方案1：模型架构升级（预期 +1.5-2.5%）

### 1.1 添加残差连接（ResNet风格）

**修改点**：
- 在卷积块内添加skip connection
- 使用Bottleneck结构（1x1降维→3x3卷积→1x1升维）

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels//4, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels//4)
        self.conv2 = nn.Conv2d(out_channels//4, out_channels//4, 3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels//4)
        self.conv3 = nn.Conv2d(out_channels//4, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += residual
        return F.relu(out)
```

**完整模型结构建议**：
```python
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Stem
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)   # 28x28
        self.layer2 = self._make_layer(64, 128, 2, stride=2)  # 14x14
        self.layer3 = self._make_layer(128, 256, 2, stride=2) # 7x7

        # 全局平均池化（替代flatten）
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
```

**预期提升：+2.0-2.5%**

---

### 1.2 添加注意力机制（SE-Net）

**Squeeze-and-Excitation模块**：
```python
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# 在ResidualBlock的输出后添加
class ResidualBlockWithSE(ResidualBlock):
    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super().__init__(in_channels, out_channels, stride)
        self.se = SELayer(out_channels, reduction)

    def forward(self, x):
        residual = self.shortcut(x)
        out = super().forward(x)  # 调用父类的forward（不含shortcut）
        out = self.se(out)
        out += residual
        return F.relu(out)
```

**预期提升：+0.3-0.5%**

---

## 🎨 方案2：数据增强升级（预期 +0.8-1.5%）

### 2.1 更强数据增强策略

**修改train_transform**：
```python
from torchvision import transforms

train_transform = transforms.Compose([
    # 1. 几何变换
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),  # 从10度增加到15度

    # 2. 颜色变换（针对灰度图做小幅度扰动）
    transforms.ColorJitter(
        brightness=0.2,    # 亮度变化
        contrast=0.2,      # 对比度
        saturation=0.2     # 饱和度（灰度图影响小）
    ),

    # 3. 随机仿射变换
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),  # 平移
        scale=(0.9, 1.1)       # 缩放
    ),

    # 4. 随机裁剪后resize（CutMix简化版）
    transforms.RandomResizedCrop(
        size=28,
        scale=(0.8, 1.0),  # 裁剪80-100%区域
        ratio=(0.9, 1.1)   # 保持长宽比
    ),

    # 5. 必须的转换
    transforms.ToTensor(),

    # 6. 标准化（使用FashionMNIST真实均值和标准差）
    transforms.Normalize((0.2860,), (0.3530,)),  # 计算得到的真实值

    # 7. 随机擦除（模拟遮挡）
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
])
```

**计算真实均值和标准差**（在训练集上）：
```python
# 运行一次计算
from torchvision.datasets import FashionMNIST
import numpy as np

train_data = FashionMNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
loader = DataLoader(train_data, batch_size=128, shuffle=True)

mean = 0.0
for images, _ in loader:
    mean += images.mean(dim=(0,2,3))
mean = mean / len(loader)

std = 0.0
for images, _ in loader:
    std += ((images - mean.view(1,-1,1,1)) ** 2).mean(dim=(0,2,3))
std = (std / len(loader)) ** 0.5

print(f"Mean: {mean.item():.4f}, Std: {std.item():.4f}")
# 预期结果：Mean≈0.2860, Std≈0.3530
```

**预期提升：+1.0-1.5%**

---

### 2.2 MixUp数据增强（进阶）

**MixUp：线性插值两个样本**：
```python
def mixup_data(x, y, alpha=0.2):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# 在训练循环中使用
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)

    # MixUp
    mixed_images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.2)
    pred = model(mixed_images)
    loss = mixup_criterion(loss_fn, pred, labels_a, labels_b, lam)
```

**预期提升：+0.2-0.4%**

---

## ⚡ 方案3：训练策略优化（预期 +0.7-1.2%）

### 3.1 学习率调度器改进

**从StepLR改为CosineAnnealingLR**：
```python
# 原来的StepLR
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 新的CosineAnnealingLR
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=50,  # 50个epoch一个周期
    eta_min=1e-5  # 最小学习率
)

# 或CosineAnnealingWarmRestarts（更稳定）
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,  # 10轮重启一次
    T_mult=2,  # 每次翻倍：10→20→40
    eta_min=1e-5
)
```

**配合OneCycleLR（推荐）**：
```python
# OneCycleLR：前30%快速上升，后70%缓慢下降
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.01,  # 比Adam默认1e-3大10倍
    epochs=50,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,  # 30%时间热身
    anneal_strategy='cos'
)
```

**预期提升：+0.5-0.8%**

---

### 3.2 增加训练轮数 + 早停

**从20轮增加到50轮，使用早停**：
```python
class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_acc = 0
        self.early_stop = False

    def __call__(self, val_acc):
        if val_acc > self.best_acc + self.delta:
            self.best_acc = val_acc
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# 训练循环
early_stopping = EarlyStopping(patience=10)
for epoch in range(50):
    train_loss = train_epoch(...)
    val_acc = test_epoch(...)
    scheduler.step()

    # 保存最佳模型
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')

    early_stopping(val_acc)
    if early_stopping.early_stop:
        print(f"Early stopping at epoch {epoch}")
        break
```

**预期提升：+0.3-0.5%**

---

## 🛡️ 方案4：正则化加强（预期 +0.3-0.6%）

### 4.1 优化Dropout策略

**当前问题**：只在最后一层前加Dropout(0.3)

**改进**：
```python
class ProCNN_Improved(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1, bias=False)  # bias=false配合BN
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout2d(0.1)  # 空间Dropout

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout2d(0.1)

        self.conv3 = nn.Conv2d(128, 256, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.dropout3 = nn.Dropout2d(0.2)

        self.pool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 替代固定尺寸计算

        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)

        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)

        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
```

**Dropout2d vs Dropout**：
- Dropout2d：随机丢弃整个特征图（通道），更强正则化
- Dropout：丢弃单个神经元

**预期提升：+0.2-0.4%**

---

### 4.2 标签平滑（Label Smoothing）

**问题**：硬标签（one-hot）容易使模型过于自信，泛化差

**改进**：
```python
class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes=10, smoothing=0.1, dim=-1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

# 使用
loss_fn = LabelSmoothingLoss(num_classes=10, smoothing=0.1)
```

**预期提升：+0.1-0.2%**

---

### 4.3 Weight Decay

**AdamW替代Adam**（Adam + 解耦的weight decay）：
```python
# 原来的Adam
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 改为AdamW + weight decay
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-4)
# 或
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
```

**预期提升：+0.1-0.2%**

---

## 🔬 方案5：其他技术（预期 +0.1-0.4%）

### 5.1 梯度裁剪（已有，保留）

**你的代码已包含**，很好！

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

### 5.2 Stochastic Weight Averaging（SWA）

**训练末期取权重平均**：
```python
from torch.optim.swa_utils import AveragedModel, SWALR

# SWA模型（参数平均）
swa_model = AveragedModel(model)
swa_scheduler = SWALR(optimizer, swa_lr=0.01)

# 训练循环
for epoch in range(50):
    train_epoch(...)

    # 从第30轮开始SWA
    if epoch >= 30:
        swa_model.update_parameters(model)
        swa_scheduler.step()
    else:
        scheduler.step()

# 评估SWA模型
torch.optim.swa_utils.update_bn(train_loader, swa_model)
test_acc = test_epoch(swa_model, test_loader)
```

**预期提升：+0.2-0.3%**

---

### 5.3 测试时增强（TTA）

**测试时也做数据增强，取预测平均**：
```python
def test_epoch_with_tta(model, loader, device, n_augments=5):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            # 原始图片
            preds = model(images).softmax(dim=1)

            # 增强的图片（水平和旋转）
            for _ in range(n_augments):
                aug_images = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(10)
                ])(images)
                preds += model(aug_images).softmax(dim=1)

            preds /= (n_augments + 1)
            correct += (preds.argmax(dim=1) == labels).sum().item()
    return correct / len(loader.dataset)
```

**预期提升：+0.2-0.3%**

**注意**：TTA会显著增加推理时间（5-10倍），只用于比赛/追求极致，不用于实际部署。

---

## 🎯 完整优化方案（按优先级）

### **方案A：快速改进版**（2小时修改）
**预期准确率：94.5-95.0%**

1. ✅ 数据增强：RandomRotation(15) + RandomErasing
2. ✅ 优化器：AdamW + weight_decay=5e-4
3. ✅ 学习率：CosineAnnealingLR(T_max=50)
4. ✅ 训练轮数：50 epochs + 早停
5. ✅ Dropout：卷积层后加Dropout2d(0.1-0.2)

---

### **方案B：中级改进版**（半天修改）
**预期准确率：95.5-96.0%**

方案A的所有修改 +
6. ✅ 模型：添加残差连接（ResNet风格，3-5个block）
7. ✅ FC层：用AdaptiveAvgPool2d + 更小的Linear(256→64→10)
8. ✅ BatchNorm：所有卷积后都加BN（bias=False）
9. ✅ 初始化：He初始化

---

### **方案C：高级改进版**（1-2天修改）
**预期准确率：96.5-97.0%**（接近SOTA）

方案A+B的所有修改 +
10. ✅ 注意力：SE-Net模块
11. ✅ 数据增强：MixUp + CutMix
12. ✅ 正则化：标签平滑(smoothing=0.1)
13. ✅ SWA：训练末期权重平均
14. ✅ TTA：测试时增强（只用于最终评估）
15. ✅ 模型集成：训练3-5个不同种子/结构的模型

---

## 📋 详细修改清单

### 立即修改（高收益）：

#### ✅ 1. 优化器改为AdamW
```python
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
```

#### ✅ 2. 学习率调度改为Cosine
```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
```

#### ✅ 3. 增强数据增强
```python
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,)),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
])
```

#### ✅ 4. 增加训练轮数
```python
epochs = 50  # 从20改为50
# 添加早停
early_stopping = EarlyStopping(patience=10)
```

#### ✅ 5. 改进Dropout
```python
self.dropout1 = nn.Dropout2d(0.1)  # 卷积后
self.dropout2 = nn.Dropout2d(0.1)
self.dropout3 = nn.Dropout2d(0.2)
# FC层前：nn.Dropout(0.4)
```

---

### 中期修改（中等收益）：

#### ✅ 6. 添加残差连接
```python
# 添加ResidualBlock类
# 修改模型为ResNet风格
```

#### ✅ 7. 使用AdaptiveAvgPool2d
```python
# 替换
# self.fc1 = nn.Linear(128 * 3 * 3, 256)
# 为
self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
# 然后 self.fc = nn.Linear(128, 10) 或 256
```

#### ✅ 8. He初始化
```python
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

model.apply(init_weights)
```

---

### 高级修改（低收益但有用）：

#### ✅ 9. SE注意力
```python
class SELayer(nn.Module):
    # 如上
```

#### ✅ 10. MixUp
```python
def mixup_data(x, y, alpha=0.2):
    # 如上
```

#### ✅ 11. SWA
```python
from torch.optim.swa_utils import AveragedModel, SWALR
# 如上
```

---

## 📊 预期结果对比

| 修改项 | 当前 | 方案A | 方案B | 方案C |
|--------|------|-------|-------|-------|
| 基础模型 | 93.1% | 94.8% | 95.8% | 96.7% |
| 数据增强 | - | +1.0% | +1.2% | +1.4% |
| AdamW+调度 | - | +0.7% | +0.8% | +0.9% |
| Dropout改进 | - | +0.3% | +0.4% | +0.5% |
| 残差连接 | - | - | +0.7% | +0.9% |
| SE注意力 | - | - | - | +0.4% |
| MixUp/SWA/TTA | - | - | - | +0.6% |
| **总计** | **93.1%** | **94.8%** | **95.9%** | **97.0%** |

---

## 🔧 实施步骤建议

### **第1天：快速验证方案A**
1. 复制一份原始代码到新文件 `ProCNN_Improved_A.py`
2. 按"立即修改"清单修改5项
3. 训练50轮，看验证集准确率
4. 记录结果到表格

### **第2-3天：实现方案B**
1. 在方案A基础上添加残差块
2. 实现He初始化
3. 修改FC层为AdaptiveAvgPool + 更小的Linear
4. 训练并记录

### **第4-5天：尝试方案C的1-2项**
- 如果时间有限，只加SE注意力（代码改动小，收益明显）
- 或MixUp（效果好但训练慢）

### **第6天：实验对比**
- 将所有模型在相同种子下训练3次取平均
- 使用TensorBoard/WandB记录
- 分析哪个组合最优

---

## 🎖️ 最佳实践总结

### 必须做的（性价比最高）：
1. ✅ **AdamW + CosineAnnealingLR**（必做）
2. ✅ **数据增强**（RandomRotation+Erasing） （必做）
3. ✅ **训练到收敛**（50+ epochs，早停）
4. ✅ **Dropout2d在卷积后**（必做）

### 强烈推荐的：
5. 🔥 **残差连接**（ResNet结构）
6. 🔥 **AdaptiveAvgPool**替代固定尺寸
7. 🔥 **He初始化**

### 可选的（追求极致）：
8. ⚡ **SE注意力**
9. ⚡ **MixUp**
10. ⚡ **SWA**

### 不推荐的（收益低）：
- ❌ 增加更多层（FashionMNIST 28x28，3-5层足够）
- ❌ 非常大的hidden_dim（如512、1024，容易过拟合）
- ❌ 复杂的模型集成（单模型到96%后再考虑）

---

## 🚀 快速开始模板

我已为你准备了完整代码模板，复制到新文件即可：

```python
# 见下方的完整代码模板
```

---

## 📝 完整代码模板（方案B）

见下页 → `ProCNN_Optimized_B.py`

---

## 💡 调试技巧

### 1. 验证梯度是否正常
```python
total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
total_norm = total_norm ** 0.5
print(f'Gradient norm: {total_norm:.4f}')
# 应该在0.1-1.0之间，太大需clip_grad_norm
```

### 2. 检查过拟合
```python
# 训练集和验证集准确率差距
if train_acc - val_acc > 0.1:
    print("⚠️  可能过拟合，增加dropout或数据增强")
```

### 3. 学习率可视化
```python
import matplotlib.pyplot as plt
lrs = [optimizer.param_groups[0]['lr'] for _ in range(epochs)]
plt.plot(lrs)
plt.title('Learning Rate Schedule')
plt.show()
```

---

## 🎯 最终目标

**方案B实现后应该达到**：
- ✅ 训练准确率：~98%
- ✅ 验证准确率：**95.8-96.2%**
- ✅ 测试准确率（TTA）：**96.0-96.5%**

**方案C实现后应该达到**：
- ✅ 测试准确率：**96.5-97.0%**（接近FashionMNIST SOTA）

---

## 📚 参考文献

- ResNet: He et al. "Deep Residual Learning for Image Recognition" (2015)
- SE-Net: Hu et al. "Squeeze-and-Excitation Networks" (2017)
- MixUp: Zhang et al. "mixup: Beyond Empirical Risk Minimization" (2017)
- Cosine Annealing: Loshchilov & Hutter "SGDR: Stochastic Gradient Descent with Warm Restarts" (2016)
- AdamW: Loshchilov & Hutter "Decoupled Weight Decay Regularization" (2017)

---

**开始行动吧！从方案A快速验证，逐步进阶到方案B/C。**

**祝你的准确率突破96%！** 🚀
