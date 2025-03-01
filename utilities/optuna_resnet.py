import optuna
import optuna.visualization as vis
import torch.optim as optim

def objective(trial):
    learning_rate = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    dropout_rate = trial.suggest_uniform("dropout", 0.2, 0.5)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    epochs = trial.suggest_int("epochs", 5, 15)

    model = models.resnet50(pretrained=True)
    num_classes = len(class_mapping)
    model.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(model.fc.in_features, num_classes)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    num_epochs = 10
    best_val_acc = 0.0
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        correct_preds, total_preds = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

        train_loss = running_train_loss / len(train_loader)
        train_accuracy = correct_preds / total_preds
        train_losses.append(train_loss)

        model.eval()
        running_val_loss = 0.0
        correct_preds, total_preds = 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).long()
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_preds += (predicted == labels).sum().item()
                total_preds += labels.size(0)

        val_loss = running_val_loss / len(val_loader)
        val_accuracy = correct_preds / total_preds
        val_losses.append(val_loss)

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy

    trial.set_user_attr("train_loss", train_losses[-1])
    trial.set_user_attr("val_loss", val_losses[-1])

    return best_val_acc  # Optuna maximizes this
    trial.report(val_accuracy, epoch)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return best_val_acc

study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
study.optimize(objective, n_trials=20)