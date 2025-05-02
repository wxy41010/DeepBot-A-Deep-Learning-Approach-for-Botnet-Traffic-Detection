import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# === Step 1: 加载数据 ===
X_train, y_train, X_test, y_test = torch.load('cnn_data.pt')

# LSTM 需要 (batch, seq_len, input_size) 格式，我们用 features 作为 seq_len
X_train = X_train.squeeze(1).unsqueeze(-1)  # (B, 6) → (B, 6, 1)
X_test = X_test.squeeze(1).unsqueeze(-1)

batch_size = 4096
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, num_workers=2, pin_memory=True)

# === Step 2: 定义 LSTM 模型 ===
class BotnetLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):  # x: (B, seq_len, 1)
        out, (h_n, _) = self.lstm(x)  # h_n: (num_layers, B, hidden)
        h_last = h_n[-1]  # (B, hidden)
        h_last = self.bn(h_last)
        h_last = self.dropout(h_last)
        return self.fc(h_last)

# === Step 3: 设置训练参数 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BotnetLSTM(input_dim=1).to(device)

# 类别权重
class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=y_train.numpy())
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
if not os.path.exists("model"):
    os.makedirs("model")

# === Step 4: 训练循环 ===
best_loss = float("inf")
patience = 5
counter = 0
loss_list = []
acc_list = []

print(f"🧠 使用设备: {device}")

for epoch in range(50):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    start = time.time()

    for xb, yb in train_loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(out, dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

    epoch_loss = total_loss
    epoch_acc = correct / total
    loss_list.append(epoch_loss)
    acc_list.append(epoch_acc)

    print(f"📘 Epoch {epoch+1:2d} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | Time: {time.time() - start:.2f}s")

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        counter = 0
        torch.save(model.state_dict(), "model/best_lstm_model.pth")
        print("  ✅ 模型已保存：best_lstm_model.pth")
    else:
        counter += 1
        if counter >= patience:
            print("  🛑 Early stopping triggered.")
            break

# === Step 5: 绘图保存 ===
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_list, label="Loss")
plt.title("LSTM - Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(acc_list, label="Accuracy", color='green')
plt.title("LSTM - Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("loss_accuracy_lstm.png")
plt.show()
print("📈 LSTM训练图已保存为 loss_accuracy_lstm.png")
