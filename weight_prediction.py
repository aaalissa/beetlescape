# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate dummy data - replace this with your actual data
X = torch.rand(100, 12)  # 1000 instances, 12 features each
y = torch.rand(100, 3)   # 1000 instances, 3 target values each

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X.numpy(), y.numpy(), test_size=0.2)
X_train, X_test, y_train, y_test = map(torch.tensor, (X_train, X_test, y_train, y_test))

# Create Tensor datasets and data loaders
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10)

# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(12, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Initialize the network
model = Net().to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(100):
    # Wrap train_loader with tqdm for a progress bar
    train_loop = tqdm(train_loader, leave=True, position=0)
    for inputs, targets in train_loop:
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update the progress bar
        train_loop.set_description(f"Epoch [{epoch+1}/100]")
        train_loop.set_postfix(loss=loss.item())
    
# Evaluate the model
model.eval()
test_loss = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        test_loss += criterion(outputs, targets).item()

print(f"Test Loss: {test_loss / len(test_loader)}")

# Make predictions with new data
new_data = torch.rand(5, 12).to(device)
with torch.no_grad():
    predictions = model(new_data)
    print(predictions.cpu().numpy())
