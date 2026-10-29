import os
import glob
import time
import argparse
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from scipy.ndimage import gaussian_filter, map_coordinates
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

# ==========================================
# 1. 15 BENCHMARK CORRUPTIONS (RGB DOMAIN)
# ==========================================

class ImageCorruptions:
    @staticmethod
    def add_gaussian_noise(img, severity=1):
        c = [0.04, 0.08, 0.12, 0.18, 0.26][severity - 1]
        return np.clip(img + np.random.normal(0, c, img.shape), 0, 1)

    @staticmethod
    def add_shot_noise(img, severity=1):
        c = [500, 250, 100, 50, 20][severity - 1]
        return np.clip(np.random.poisson(img * c) / float(c), 0, 1)

    @staticmethod
    def add_impulse_noise(img, severity=1):
        c = [0.02, 0.05, 0.09, 0.15, 0.23][severity - 1]
        corrupted = img.copy()
        prob = np.random.uniform(0, 1, img.shape[:2])
        corrupted[prob < (c / 2)] = 0
        corrupted[prob > (1 - c / 2)] = 1
        return corrupted

    @staticmethod
    def apply_defocus_blur(img, severity=1):
        c = [0.5, 1.0, 1.5, 2.0, 2.5][severity - 1]
        k = int(c * 2) * 2 + 1
        return cv2.GaussianBlur(img, (k, k), c)

    @staticmethod
    def apply_motion_blur(img, severity=1):
        c = [3, 5, 7, 9, 11][severity - 1]
        kernel = np.zeros((c, c))
        kernel[int((c - 1) / 2), :] = np.ones(c) / c
        return cv2.filter2D(img, -1, kernel)

    @staticmethod
    def apply_zoom_blur(img, severity=1):
        c = [1.02, 1.04, 1.06, 1.08, 1.10][severity - 1]
        h, w = img.shape[:2]
        out = np.zeros_like(img, dtype=np.float32)
        for zoom in np.linspace(1.0, c, 5):
            M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, zoom)
            out += cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return np.clip(out / 5.0, 0, 1)

    @staticmethod
    def add_fog(img, severity=1):
        c = [0.2, 0.35, 0.5, 0.65, 0.8][severity - 1]
        return np.clip(img * (1 - c) + (np.ones_like(img) * c), 0, 1)

    @staticmethod
    def add_frost(img, severity=1):
        c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
        frost = gaussian_filter(np.random.uniform(0, 1, img.shape), sigma=1.5)
        return np.clip(img * (1 - c) + frost * c, 0, 1)

    @staticmethod
    def add_snow(img, severity=1):
        c = [0.05, 0.1, 0.15, 0.25, 0.35][severity - 1]
        snow_layer = np.random.normal(0.8, 0.1, img.shape)
        snow_mask = np.random.uniform(0, 1, img.shape[:2]) < c
        corrupted = img.copy()
        corrupted[snow_mask] = snow_layer[snow_mask]
        return np.clip(corrupted, 0, 1)

    @staticmethod
    def adjust_brightness(img, severity=1):
        c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
        return np.clip(img + c, 0, 1)

    @staticmethod
    def adjust_contrast(img, severity=1):
        c = [0.8, 0.6, 0.4, 0.3, 0.2][severity - 1]
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        return np.clip((img - mean) * c + mean, 0, 1)

    @staticmethod
    def apply_pixelate(img, severity=1):
        c = [0.8, 0.6, 0.4, 0.3, 0.2][severity - 1]
        h, w = img.shape[:2]
        small = cv2.resize(img, (max(1, int(w * c)), max(1, int(h * c))), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def apply_elastic_transform(img, severity=1):
        alpha, sigma = [(10, 2), (15, 3), (20, 4), (25, 5), (30, 6)][severity - 1]
        shape = img.shape[:2]
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        corrupted = np.zeros_like(img)
        for i in range(img.shape[2]):
            corrupted[:, :, i] = map_coordinates(img[:, :, i], indices, order=1).reshape(shape)
        return np.clip(corrupted, 0, 1)

    @staticmethod
    def apply_jpeg_compression(img, severity=1):
        q = [80, 50, 30, 15, 5][severity - 1]
        img_255 = (img * 255).astype(np.uint8)
        _, encimg = cv2.imencode('.jpg', img_255, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        decimg = cv2.imdecode(encimg, 1)
        if decimg is None:
            return img
        return decimg.astype(np.float32) / 255.0

    @classmethod
    def apply_corruption(cls, img, corruption_type, severity=1):
        corruptions_map = {
            'gaussian': cls.add_gaussian_noise,
            'shot': cls.add_shot_noise,
            'impulse': cls.add_impulse_noise,
            'defocus_blur': cls.apply_defocus_blur,
            'motion_blur': cls.apply_motion_blur,
            'zoom_blur': cls.apply_zoom_blur,
            'fog': cls.add_fog,
            'frost': cls.add_frost,
            'snow': cls.add_snow,
            'brightness': cls.adjust_brightness,
            'contrast': cls.adjust_contrast,
            'pixelate': cls.apply_pixelate,
            'elastic': cls.apply_elastic_transform,
            'jpeg': cls.apply_jpeg_compression
        }
        return corruptions_map.get(corruption_type, lambda x, s: x)(img, severity)


# ==========================================
# 2. PREPROCESSING & SGW FEATURE EXTRACTION
# ==========================================

class SimplifiedGaborFilterBank:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.scales = [0.3 * np.pi, 0.5 * np.pi]
        self.orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        self.kernels = self._generate_sgw_kernels()

    def _generate_sgw_kernels(self):
        kernels = []
        half_w = self.window_size // 2
        y, x = np.mgrid[-half_w:half_w+1, -half_w:half_w+1]
        for w in self.scales:
            for theta in self.orientations:
                g_img = np.exp(-(x**2 + y**2) / (2 * (1.0**2))) * np.sin(w * (x * np.cos(theta) + y * np.sin(theta)))
                sgw_kernel = np.zeros_like(g_img)
                sgw_kernel[g_img > 0.1] = 1.0
                sgw_kernel[g_img < -0.1] = -1.0
                kernels.append(sgw_kernel.astype(np.float32))
        return kernels

    def apply_sgw(self, img_gray):
        sgw_channels = [cv2.filter2D(img_gray, cv2.CV_32F, k) for k in self.kernels]
        return np.stack(sgw_channels, axis=-1)

def preprocess_single_image(img_rgb, target_size=(32, 32)):
    # CLAHE
    lab = cv2.cvtColor((img_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_rgb = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)
    
    # Resize & Grayscale
    resized = cv2.resize(enhanced_rgb, target_size)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    
    # 8-Channel SGW
    sgw_bank = SimplifiedGaborFilterBank(window_size=5)
    sgw_features = sgw_bank.apply_sgw(gray)
    
    # Normalize [0, 1]
    min_v, max_v = np.min(sgw_features), np.max(sgw_features)
    return (sgw_features - min_v) / (max_v - min_v + 1e-8)


# ==========================================
# 3. DATASET LOADER ENGINE
# ==========================================

def load_dataset_from_path(root_path, max_samples_per_class=300):
    root = os.path.abspath(root_path)
    if not os.path.exists(root):
        print(f"[WARNING] Path '{root_path}' does not exist.")
        return None, None

    images_rgb = []
    labels = []

    print(f"[+] Scanning dataset directory: {root}")
    class_folders = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    
    # Handle Train/Test nested subfolder layout
    if 'Train' in class_folders or 'train' in class_folders:
        train_folder = 'Train' if 'Train' in class_folders else 'train'
        sub_root = os.path.join(root, train_folder)
        class_folders = sorted([d for d in os.listdir(sub_root) if os.path.isdir(os.path.join(sub_root, d))])
    else:
        sub_root = root

    for class_id, class_name in enumerate(class_folders):
        if class_id >= 43:  # Map to 43 GTSRB target classes
            break
        class_dir = os.path.join(sub_root, class_name)
        img_files = glob.glob(os.path.join(class_dir, "*.*"))
        
        for img_path in img_files[:max_samples_per_class]:
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.ppm', '.bmp')):
                img = cv2.imread(img_path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_rgb = cv2.resize(img_rgb, (32, 32)).astype(np.float32) / 255.0
                    images_rgb.append(img_rgb)
                    labels.append(class_id)

    if not images_rgb:
        print(f"[WARNING] No images loaded from '{root_path}'.")
        return None, None

    print(f"  └─ Successfully loaded {len(images_rgb)} images across {len(set(labels))} classes.")
    return np.array(images_rgb), np.array(labels)


# ==========================================
# 4. DEEP CLASSIFIER ARCHITECTURE (Swish + Residual)
# ==========================================

def build_innovative_traffic_sign_net(input_shape=(32, 32, 8), num_classes=43):
    inputs = layers.Input(shape=input_shape)

    # Entry Block
    x = layers.Conv2D(32, (5, 5), padding='same')(inputs)
    x = layers.Activation('swish')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Residual Block 1
    res = layers.Conv2D(64, (1, 1), strides=(2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.Activation('swish')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.add([x, res])
    x = layers.Activation('swish')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.Activation('swish')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Global Classification Head
    x = layers.Flatten()(x)
    x = layers.Dense(256)(x)
    x = layers.Activation('swish')(x)
    x = layers.Dropout(0.4)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax', name='primary_output')(x)
    return models.Model(inputs=inputs, outputs=outputs, name="InnovativeTrafficSignNet")


# ==========================================
# 5. HEARTBEAT CALLBACK & EXECUTION PIPELINE
# ==========================================

class TrainingHeartbeatCallback(tf.keras.callbacks.Callback):
    def __init__(self, ping_interval_sec=5):
        super(TrainingHeartbeatCallback, self).__init__()
        self.ping_interval_sec = ping_interval_sec
        self.last_ping = time.time()

    def on_batch_end(self, batch, logs=None):
        curr = time.time()
        if curr - self.last_ping >= self.ping_interval_sec:
            print(f"[HEARTBEAT] System Active | Batch: {batch:03d} | Loss: {logs.get('loss'):.4f} | Acc: {logs.get('accuracy'):.4f}")
            self.last_ping = curr


def run_pipeline(gtsrb_path, btsd_path, ctsd_path, epochs=30, batch_size=64):
    print("\n=========================================================================")
    print("  INNOVATIVE MULTI-DATASET TRAFFIC SIGN CLASSIFICATION PIPELINE")
    print("=========================================================================\n")

    # Load All Three Datasets
    all_rgb = []
    all_labels = []

    for name, path in [("GTSRB", gtsrb_path), ("BTSD", btsd_path), ("CTSD", ctsd_path)]:
        rgb, lbl = load_dataset_from_path(path)
        if rgb is not None:
            all_rgb.append(rgb)
            all_labels.append(lbl)

    if not all_rgb:
        print("[ERROR] No datasets loaded. Please verify dataset folder paths.")
        return

    X_rgb = np.concatenate(all_rgb, axis=0)
    y_raw = np.concatenate(all_labels, axis=0)

    print(f"\n[+] Total Unified Dataset Size: {len(X_rgb)} samples across 43 classes.")

    # Shuffle and Split (70% Train, 15% Val, 15% Test)
    indices = np.arange(len(X_rgb))
    np.random.seed(42)
    np.random.shuffle(indices)

    X_rgb = X_rgb[indices]
    y_raw = y_raw[indices]

    n_train = int(len(X_rgb) * 0.70)
    n_val = int(len(X_rgb) * 0.15)

    X_rgb_train, y_train = X_rgb[:n_train], y_raw[:n_train]
    X_rgb_val, y_val = X_rgb[n_train:n_train+n_val], y_raw[n_train:n_train+n_val]
    X_rgb_test, y_test = X_rgb[n_train+n_val:], y_raw[n_train+n_val:]

    # Extract 8-Channel SGW Features
    print("\n[+] Extracting 8-Channel SGW Features & CLAHE Normalization...")
    X_sgw_train = np.array([preprocess_single_image(img) for img in X_rgb_train])
    X_sgw_val = np.array([preprocess_single_image(img) for img in X_rgb_val])
    X_sgw_test = np.array([preprocess_single_image(img) for img in X_rgb_test])

    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=43)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=43)

    # Class Weights to combat GTSRB imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    # Build and Compile Model
    model = build_innovative_traffic_sign_net(input_shape=(32, 32, 8), num_classes=43)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\n--- STAGE 1: TRAINING MODEL ON CLEAN SGW FEATURES ---")
    model.fit(
        X_sgw_train, y_train_cat,
        validation_data=(X_sgw_val, y_val_cat),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=[TrainingHeartbeatCallback(ping_interval_sec=5)],
        verbose=1
    )

    # Clean Test Set Evaluation
    print("\n--- STAGE 2: CLEAN TEST SET EVALUATION METRICS ---")
    preds_clean = np.argmax(model.predict(X_sgw_test), axis=1)
    clean_acc = np.mean(preds_clean == y_test) * 100
    print(f"Overall Clean Test Accuracy: {clean_acc:.2f}%\n")
    print(classification_report(y_test, preds_clean, zero_division=0))

    # 15 Corruptions Evaluation Engine
    print("\n=========================================================================")
    print("  STAGE 3: BENCHMARK EVALUATION ACROSS 15 CORRUPTIONS (SEVERITIES 1-5)")
    print("=========================================================================\n")

    corruptions = [
        'gaussian', 'shot', 'impulse', 'defocus_blur', 'motion_blur',
        'zoom_blur', 'fog', 'frost', 'snow', 'brightness',
        'contrast', 'pixelate', 'elastic', 'jpeg'
    ]

    for corr in corruptions:
        print(f"\n--- Corruption Type: {corr.upper()} ---")
        for sev in range(1, 6):
            # Apply corruption to RAW RGB images FIRST, then extract SGW
            X_corrupted_rgb = np.array([ImageCorruptions.apply_corruption(img, corr, sev) for img in X_rgb_test])
            X_corrupted_sgw = np.array([preprocess_single_image(img) for img in X_corrupted_rgb])
            
            preds = np.argmax(model.predict(X_corrupted_sgw, verbose=0), axis=1)
            acc = np.mean(preds == y_test) * 100
            print(f"  Severity {sev}: Accuracy = {acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Innovative Traffic Sign ML Pipeline")
    parser.add_argument("--gtsrb_path", type=str, default=os.path.abspath("./data/GTSRB"))
    parser.add_argument("--btsd_path", type=str, default=os.path.abspath("./data/BTSD"))
    parser.add_argument("--ctsd_path", type=str, default=os.path.abspath("./data/CTSD"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    run_pipeline(
        gtsrb_path=os.path.abspath(args.gtsrb_path),
        btsd_path=os.path.abspath(args.btsd_path),
        ctsd_path=os.path.abspath(args.ctsd_path),
        epochs=args.epochs,
        batch_size=args.batch_size
    )