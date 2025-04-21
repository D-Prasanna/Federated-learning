import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from cryptography.fernet import Fernet
import multiprocessing
import pickle
import os

# ======= Shared Model and Data Setup =======
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def load_data():
    data = load_diabetes()
    X = data.data
    y = (data.target > 140).astype(int)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(y_test, dtype=torch.long))
    return train_dataset, test_dataset, X.shape[1]

def evaluate(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            outputs = model(x_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    return correct / total

def train_model(model, loader, epochs=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
    return model

# ======= 🔐 Enclave Simulation =======
def enclave_process(train_dataset, input_size, enc_key, return_file):
    model = SimpleNN(input_size)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    model = train_model(model, train_loader)

    # Serialize weights and encrypt
    f = Fernet(enc_key)
    state_dict = pickle.dumps(model.state_dict())
    encrypted_weights = f.encrypt(state_dict)

    with open(return_file, 'wb') as f_out:
        f_out.write(encrypted_weights)

# ======= 🚀 Main Logic =======
def main():
    train_dataset, test_dataset, input_size = load_data()
    test_loader = DataLoader(test_dataset, batch_size=64)

    # ----------- Baseline Training -----------
    baseline_model = SimpleNN(input_size)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    train_model(baseline_model, train_loader)
    acc_baseline = evaluate(baseline_model, test_loader)
    print(f"Baseline Accuracy: {acc_baseline:.4f}")

    # ----------- Simulated Enclave Training -----------
    key = Fernet.generate_key()
    enc_file = "encrypted_weights.bin"

    # Simulate enclave in separate process
    proc = multiprocessing.Process(
        target=enclave_process,
        args=(train_dataset, input_size, key, enc_file)
    )
    proc.start()
    proc.join()

    # Server side - decrypt and load weights
    with open(enc_file, 'rb') as f_in:
        encrypted_weights = f_in.read()
    fernet = Fernet(key)
    decrypted = fernet.decrypt(encrypted_weights)
    state_dict = pickle.loads(decrypted)

    secure_model = SimpleNN(input_size)
    secure_model.load_state_dict(state_dict)
    acc_secure = evaluate(secure_model, test_loader)
    print(f"Simulated Enclave Accuracy: {acc_secure:.4f}")

    # Cleanup
    os.remove(enc_file)

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)  # ✅ Works in scripts and notebooks
    except RuntimeError:
        pass  # Already set — fine in Jupyter
    main()
