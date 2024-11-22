import matplotlib.pyplot as plt
import seaborn as sns

def plot_images(images, labels, grid_shape=(5, 5)):
    plt.figure(figsize=(10, 10))
    for i in range(grid_shape[0] * grid_shape[1]):
        plt.subplot(grid_shape[0], grid_shape[1], i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[i].cpu().numpy(), cmap='gray')
        plt.xlabel(labels[i])
    plt.show()

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
