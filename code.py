import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset, random_split

DATASET_ROOT = Path("BTSD/BelgiumTSC_Training/Training")
IMAGE_SIZE = (32, 32)
BATCH_SIZE = 64
NUM_EPOCHS = 3
NUM_CLASSES = 62
LEARNING_RATE = 0.001
RANDOM_SEED = 42
NOISE_TYPES = [
    "Gaussian Noise",
    "Impulse Noise",
    "Shot Noise",
    "Snow",
    "Fog",
    "Frost",
    "Motion Blur",
    "Zoom Blur",
    "Defocus Blur",
    "Brightness",
    "Contrast",
    "Pixelate",
    "Elastic Transform",
    "JPEG Compression",
]
SEVERITY_LEVELS = [1, 2, 3, 4, 5]

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class BelgiumTSCDataset(Dataset):
    def __init__(self, data_dir, image_size=(32, 32)):
        self.image_size = image_size
        self.samples = []

        for class_dir in sorted(Path(data_dir).iterdir()):
            if not class_dir.is_dir():
                continue

            annotation_path = class_dir / f"GT-{class_dir.name}.csv"
            if not annotation_path.exists():
                continue

            with annotation_path.open("r", newline="") as csv_file:
                reader = csv.DictReader(csv_file, delimiter=";")
                for row in reader:
                    image_name = row["Filename"].strip()
                    image_path = class_dir / image_name
                    if image_path.exists():
                        self.samples.append((image_path, int(row["ClassId"])))

        if not self.samples:
            raise ValueError(f"No BTSD images were found in {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        image = image.resize(self.image_size)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_array = np.transpose(image_array, (2, 0, 1))
        return torch.from_numpy(image_array), label


class NoisyValidationDataset(Dataset):
    def __init__(self, base_dataset, noise_type, severity):
        self.base_dataset = base_dataset
        self.noise_type = noise_type
        self.severity = severity

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image_tensor, label = self.base_dataset[idx]
        image_array = image_tensor.permute(1, 2, 0).numpy()
        noisy_array = apply_noise(image_array, self.noise_type, self.severity)
        noisy_tensor = torch.from_numpy(np.transpose(noisy_array, (2, 0, 1))).float()
        return noisy_tensor, label


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, in_channels=3):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


def compute_metrics(y_true, y_pred, num_classes):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    per_class_precision = precision_score(y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0)

    metrics = {
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "confusion_matrix": cm,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
    }
    return metrics


def plot_loss_and_accuracy(train_losses, val_accuracies):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_losses, label="Training Loss", color="blue")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(val_accuracies, label="Validation Accuracy", color="green")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_metrics.png", dpi=200)
    plt.close(fig)


def plot_confusion_matrix(cm, title="Confusion Matrix", filename="confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = int(cm[i, j])
            ax.text(j, i, value, ha="center", va="center", color="white" if value > cm.max() / 2 else "black")

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close(fig)


def tensor_to_pil(image_tensor):
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(image_np)


def apply_noise(image_array, noise_type, severity):
    image_array = np.clip(image_array.astype(np.float32), 0.0, 1.0)
    height, width, channels = image_array.shape
    severity_strength = severity / 5.0

    if noise_type == "Gaussian Noise":
        noise = np.random.normal(0.0, severity_strength, image_array.shape).astype(np.float32)
        return np.clip(image_array + noise, 0.0, 1.0)

    if noise_type == "Impulse Noise":
        prob = 0.05 + severity_strength * 0.18
        noisy = image_array.copy()
        salt_mask = np.random.random((height, width)) < prob
        pepper_mask = np.random.random((height, width)) < prob
        noisy[salt_mask] = 1.0
        noisy[pepper_mask] = 0.0
        return noisy

    if noise_type == "Shot Noise":
        shot = np.random.poisson(image_array * (255.0 * (1.0 + severity_strength * 1.5))) / (255.0 * (1.0 + severity_strength * 1.5))
        return np.clip(shot, 0.0, 1.0)

    if noise_type == "Snow":
        snow_mask = np.random.random((height, width)) < (0.08 + severity_strength * 0.18)
        noisy = image_array.copy()
        noisy[snow_mask] = 1.0
        return noisy

    if noise_type == "Fog":
        fog_level = severity_strength * 0.8
        return np.clip(image_array + fog_level * (1.0 - image_array), 0.0, 1.0)

    if noise_type == "Frost":
        noisy = image_array.copy()
        frost_mask = np.random.random((height, width)) < (0.10 + severity_strength * 0.20)
        noisy[frost_mask] = np.clip(noisy[frost_mask] + 0.45, 0.0, 1.0)
        stripe_pattern = np.sin(np.linspace(0, 2 * np.pi, width)) * (0.25 + severity_strength * 0.25)
        stripes = np.tile(stripe_pattern.reshape(1, width, 1), (height, 1, 3))
        noisy = np.clip(noisy + stripes, 0.0, 1.0)
        return noisy

    if noise_type == "Motion Blur":
        pil_image = Image.fromarray((image_array * 255).astype(np.uint8)).convert("RGB")
        blur = pil_image.filter(ImageFilter.GaussianBlur(radius=severity * 0.7 + 1.0))
        return np.asarray(blur, dtype=np.float32) / 255.0

    if noise_type == "Zoom Blur":
        pil_image = Image.fromarray((image_array * 255).astype(np.uint8)).convert("RGB")
        scale_factor = 1.0 + severity_strength * 1.1
        small_size = (max(1, int(width / scale_factor)), max(1, int(height / scale_factor)))
        scaled = pil_image.resize(small_size, Image.Resampling.BICUBIC)
        blurred = scaled.resize((width, height), Image.Resampling.BICUBIC)
        return np.asarray(blurred, dtype=np.float32) / 255.0

    if noise_type == "Defocus Blur":
        pil_image = Image.fromarray((image_array * 255).astype(np.uint8)).convert("RGB")
        blur = pil_image.filter(ImageFilter.GaussianBlur(radius=severity * 0.6 + 1.0))
        return np.asarray(blur, dtype=np.float32) / 255.0

    if noise_type == "Brightness":
        factor = 1.0 + severity_strength * 0.9
        return np.clip(image_array * factor, 0.0, 1.0)

    if noise_type == "Contrast":
        mean = image_array.mean()
        contrast = 1.0 + severity_strength * 1.4
        adjusted = (image_array - mean) * contrast + mean
        return np.clip(adjusted, 0.0, 1.0)

    if noise_type == "Pixelate":
        pil_image = Image.fromarray((image_array * 255).astype(np.uint8)).convert("RGB")
        block = max(2, 16 - severity * 2)
        small = pil_image.resize((max(1, width // block), max(1, height // block)), Image.Resampling.NEAREST)
        pixelated = small.resize((width, height), Image.Resampling.NEAREST)
        return np.asarray(pixelated, dtype=np.float32) / 255.0

    if noise_type == "Elastic Transform":
        row_offsets = np.round(np.sin(np.linspace(0, 2 * np.pi, height)) * (severity * 2.5)).astype(int)
        col_offsets = np.round(np.cos(np.linspace(0, 2 * np.pi, width)) * (severity * 2.5)).astype(int)
        warped = np.zeros_like(image_array)
        for y in range(height):
            for x in range(width):
                src_y = (y + row_offsets[y]) % height
                src_x = (x + col_offsets[x]) % width
                warped[y, x] = image_array[src_y, src_x]
        return np.clip(warped, 0.0, 1.0)

    if noise_type == "JPEG Compression":
        pil_image = Image.fromarray((image_array * 255).astype(np.uint8)).convert("RGB")
        quality = max(15, 95 - severity * 15)
        buffer = __import__("io").BytesIO()
        pil_image.save(buffer, format="JPEG", quality=quality)
        compressed = Image.open(buffer)
        return np.asarray(compressed, dtype=np.float32) / 255.0

    return image_array


def evaluate_model(model, data_loader, num_classes):
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            targets.extend(labels.cpu().tolist())

    return compute_metrics(targets, predictions, num_classes)


def save_results_csv(rows, filename="noise_comparison_results.csv"):
    fieldnames = [
        "noise_type",
        "severity",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
    ]
    with Path(filename).open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


full_dataset = BelgiumTSCDataset(DATASET_ROOT, image_size=IMAGE_SIZE)
train_size = min(2500, int(0.8 * len(full_dataset)))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(RANDOM_SEED),
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = SimpleCNN(num_classes=NUM_CLASSES, in_channels=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses = []
val_accuracies = []

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / max(1, len(train_loader))
    train_losses.append(avg_train_loss)

    val_metrics = evaluate_model(model, val_loader, NUM_CLASSES)
    val_accuracy = val_metrics["accuracy"]
    val_accuracies.append(val_accuracy)

    print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

plot_loss_and_accuracy(train_losses, val_accuracies)

train_metrics = evaluate_model(model, train_loader, NUM_CLASSES)
baseline_metrics = evaluate_model(model, val_loader, NUM_CLASSES)

print("\nBaseline Metrics (clean dataset)")
print(f"Train Accuracy: {train_metrics['accuracy']:.4f}")
print(f"Train F1 (macro): {train_metrics['f1_macro']:.4f}")
print(f"Test Accuracy: {baseline_metrics['accuracy']:.4f}")
print(f"Test F1 (macro): {baseline_metrics['f1_macro']:.4f}")

baseline_cm = baseline_metrics["confusion_matrix"]
plot_confusion_matrix(baseline_cm, title="Baseline Confusion Matrix", filename="baseline_confusion_matrix.png")

noise_results = []
aggregated_targets = []
aggregated_predictions = []

for noise_type in NOISE_TYPES:
    for severity in SEVERITY_LEVELS:
        noisy_dataset = NoisyValidationDataset(val_dataset, noise_type, severity)
        noisy_loader = DataLoader(noisy_dataset, batch_size=BATCH_SIZE, shuffle=False)
        noise_metrics = evaluate_model(model, noisy_loader, NUM_CLASSES)
        row = {
            "noise_type": noise_type,
            "severity": severity,
            "accuracy": float(noise_metrics["accuracy"]),
            "precision_macro": float(noise_metrics["precision_macro"]),
            "recall_macro": float(noise_metrics["recall_macro"]),
            "f1_macro": float(noise_metrics["f1_macro"]),
        }
        noise_results.append(row)

        noisy_targets = []
        noisy_predictions = []
        with torch.no_grad():
            for inputs, labels in noisy_loader:
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                noisy_predictions.extend(preds.cpu().tolist())
                noisy_targets.extend(labels.cpu().tolist())

        aggregated_targets.extend(noisy_targets)
        aggregated_predictions.extend(noisy_predictions)

        print(
            f"{noise_type:20} | Severity {severity} | "
            f"Accuracy: {noise_metrics['accuracy']:.4f} | "
            f"F1: {noise_metrics['f1_macro']:.4f}"
        )

save_results_csv(noise_results)

overall_noise_metrics = compute_metrics(aggregated_targets, aggregated_predictions, NUM_CLASSES)
combined_noise_cm = overall_noise_metrics["confusion_matrix"]
plot_confusion_matrix(combined_noise_cm, title="Combined Noise Confusion Matrix", filename="noise_confusion_matrix.png")

print("\nSummary Comparison")
print(f"Baseline Accuracy: {baseline_metrics['accuracy']:.4f}")
print(f"Average Noise Accuracy: {np.mean([row['accuracy'] for row in noise_results]):.4f}")
print(f"Worst Noise Accuracy: {min(row['accuracy'] for row in noise_results):.4f}")
print(f"Best Noise Accuracy: {max(row['accuracy'] for row in noise_results):.4f}")
print(f"Average Noise F1: {np.mean([row['f1_macro'] for row in noise_results]):.4f}")
print(f"Combined Noise Accuracy: {overall_noise_metrics['accuracy']:.4f}")
print(f"Combined Noise F1: {overall_noise_metrics['f1_macro']:.4f}")
print("\nSaved metrics to noise_comparison_results.csv")
print("Saved plots to baseline_confusion_matrix.png, noise_confusion_matrix.png, and training_metrics.png")