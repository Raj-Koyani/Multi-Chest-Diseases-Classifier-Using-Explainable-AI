if __name__ == "__main__":
    import os
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms, models
    from torch.utils.data import DataLoader
    from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from tqdm import tqdm

    # ========================
    # CONFIG
    # ========================
    DATA_DIR = "dataset_final"  # Train/val/test folders saved
    BATCH_SIZE = 16
    IMG_SIZE = 224
    NUM_CLASSES = 4
    EPOCHS = 25
    LR = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_SAVE_PATH = "densenet_best.pth"

    # ========================
    # DATA TRANSFORMS
    # ========================
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # ========================
    # DATASETS & DATALOADERS
    # ========================
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=val_test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ========================
    # MODEL
    # ========================
    model = models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, NUM_CLASSES)
    model = model.to(DEVICE)

    # ========================
    # LOSS & OPTIMIZER
    # ========================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # ========================
    # TRAINING LOOP
    # ========================
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        total = 0

        # ========================
        # Live tqdm loop for training
        # ========================
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS}")

        all_preds_train_batch = []
        all_labels_train_batch = []

        for i, (images, labels) in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            total += labels.size(0)

            # Collect predictions for live metrics
            _, preds = torch.max(outputs, 1)
            all_preds_train_batch.extend(preds.cpu().numpy())
            all_labels_train_batch.extend(labels.cpu().numpy())

            # Compute live metrics for this batch
            acc = (np.array(all_preds_train_batch) == np.array(all_labels_train_batch)).mean()
            precision = precision_score(all_labels_train_batch, all_preds_train_batch, average='macro', zero_division=0)
            recall = recall_score(all_labels_train_batch, all_preds_train_batch, average='macro', zero_division=0)
            f1 = f1_score(all_labels_train_batch, all_preds_train_batch, average='macro', zero_division=0)

            # Update tqdm description live
            loop.set_postfix(Loss=loss.item(), Acc=f"{acc:.4f}", Precision=f"{precision:.4f}", Recall=f"{recall:.4f}", F1=f"{f1:.4f}")

        # ========================
        # VALIDATION METRICS
        # ========================
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        all_preds_val = []
        all_labels_val = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                all_preds_val.extend(preds.cpu().numpy())
                all_labels_val.extend(labels.cpu().numpy())
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        val_precision = precision_score(all_labels_val, all_preds_val, average='macro', zero_division=0)
        val_recall = recall_score(all_labels_val, all_preds_val, average='macro', zero_division=0)
        val_f1 = f1_score(all_labels_val, all_preds_val, average='macro', zero_division=0)

        # Scheduler step
        scheduler.step(val_loss)

        # ========================
        # Epoch-level metrics
        # ========================
        train_loss = running_loss / total
        print(f"\nEpoch [{epoch+1}/{EPOCHS}] Summary --> "
              f"Train Loss: {train_loss:.4f}, Train Acc: {acc:.4f}, "
              f"Train Precision: {precision:.4f}, Train Recall: {recall:.4f}, Train F1: {f1:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
              f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ Best model saved with val acc: {best_val_acc:.4f}")

    # ========================
    # EVALUATION ON TEST SET
    # ========================
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=train_dataset.classes, yticklabels=train_dataset.classes, cmap="Blues")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.show()