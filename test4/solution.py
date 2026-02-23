import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import ssl
from sklearn.metrics import (
    precision_recall_curve, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)


ssl._create_default_https_context = ssl._create_unverified_context
datasets.MNIST.mirrors = ["https://ossci-datasets.s3.amazonaws.com/mnist/", "http://yann.lecun.com/exdb/mnist/"]

# 1. ЗАГРУЗКА ИЗОБРАЖЕНИЙ
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

try:
    full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
except:
    # Резервный источник, если основные лежат
    datasets.MNIST.resources = [
        ('https://github.com/cvdfoundation/mnist/raw/master/train-images-idx3-ubyte.gz', 'f68b3c2dc028615953d74f96f8b211a0'),
        ('https://github.com/cvdfoundation/mnist/raw/master/train-labels-idx1-ubyte.gz', 'd53e105ee54ea40751a0d78e5d972202'),
        ('https://github.com/cvdfoundation/mnist/raw/master/t10k-images-idx3-ubyte.gz', '9fb629c4189a5c3046f3d35b5ca43ad1'),
        ('https://github.com/cvdfoundation/mnist/raw/master/t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce57d3e74f01083')
    ]
    full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 2. РАЗДЕЛЕНИЕ ДАННЫХ (70% ТРЕНИРОВКА / 30% ТЕСТ)
train_size = int(0.7 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_subset, test_subset = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

# 3 & 4. СЕГМЕНТАЦИЯ И РАЗМЕТКА ЦИФРЫ "4"
# Метод: Бинарная маска (thresholding) встроенная в логику разметки класса
def get_binary_labels(labels):
    return (labels == 4).float().unsqueeze(1)

# 5. ОБУЧЕНИЕ МОДЕЛИ НА НАХОЖДЕНИЕ ТОЛЬКО "4"
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = SimpleCNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Начало обучения...")
model.train()
for epoch in range(1): # 1 эпоха для теста
    for images, labels in train_loader:
        targets = get_binary_labels(labels)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# 6. ПРОВЕРКА НА ТЕСТОВОЙ ВЫБОРКЕ
model.eval()
y_true, y_probs, y_preds = [], [], []

with torch.no_grad():
    for images, labels in test_loader:
        targets = get_binary_labels(labels)
        outputs = model(images)
        y_true.extend(targets.numpy())
        y_probs.extend(outputs.numpy())
        y_preds.extend((outputs > 0.5).float().numpy())

# 7 & 9. ОЦЕНКА РЕЗУЛЬТАТОВ И МЕТРИКИ
acc = accuracy_score(y_true, y_preds)
prec = precision_score(y_true, y_preds)
rec = recall_score(y_true, y_preds)
f1 = f1_score(y_true, y_preds)
roc_auc = roc_auc_score(y_true, y_probs)

print(f"\n--- МЕТРИКИ ---")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# 8. ГРАФИКИ И ИНТЕРПРЕТАЦИЯ
fpr, tpr, _ = roc_curve(y_true, y_probs)
precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_probs)

plt.figure(figsize=(12, 5))

# ROC-кривая
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='blue', label=f'ROC-AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.title('ROC-кривая')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()

# Precision-Recall кривая
plt.subplot(1, 2, 2)
plt.plot(recall_vals, precision_vals, color='green')
plt.title('Precision-Recall кривая')
plt.xlabel('Recall')
plt.ylabel('Precision')

plt.tight_layout()
plt.show()