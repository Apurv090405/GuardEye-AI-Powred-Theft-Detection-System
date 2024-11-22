import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from combined import CombinedModel
from dataset import CrimeDataset  # Create this for loading your dataset

# Parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 8
num_classes = 8

# Dataset and DataLoader
test_dataset = CrimeDataset(train=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load Model
model = CombinedModel(num_classes=num_classes)
model.load_state_dict(torch.load("crime_detection_model.pth"))
model = model.to(device)
model.eval()

# Evaluation Loop
y_true, y_pred = [], []
with torch.no_grad():
    for data, labels in test_loader:
        data = data.to(device)
        labels = labels.to(device)

        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Confusion Matrix and Classification Report
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=test_dataset.classes)

print("Classification Report:")
print(report)

plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
