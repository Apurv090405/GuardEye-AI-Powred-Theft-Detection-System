import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from combined import CombinedModel
from dataset import CrimeDataset  # Create this for loading your dataset

# Parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 8
num_epochs = 50
learning_rate = 0.001
num_classes = 5

# Dataset and DataLoader
train_dataset = CrimeDataset(train=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model
model = CombinedModel(num_classes=num_classes)
model = model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
print("Starting Training...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

# Save the trained model
torch.save(model.state_dict(), "crime_detection_model.pth")
print("Training Complete. Model saved as 'crime_detection_model.pth'.")
