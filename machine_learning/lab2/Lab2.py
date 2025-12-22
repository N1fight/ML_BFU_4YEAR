import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # Для отображения графиков
import matplotlib.pyplot as plt
import sys
import os

# Создаем папку для сохранения результатов
if not os.path.exists('lab2_results'):
    os.makedirs('lab2_results')

# Открываем файл для записи вывода консоли
original_stdout = sys.stdout
with open('lab2_results/output_log.txt', 'w', encoding='utf-8') as f:
    # Перенаправляем вывод в файл и на консоль
    class Tee:
        def __init__(self, *files):
            self.files = files

        def write(self, obj):
            for file in self.files:
                file.write(obj)
                file.flush()

        def flush(self):
            for file in self.files:
                file.flush()


    sys.stdout = Tee(original_stdout, f)

    print("=" * 60)
    print("Лабораторная работа №2: Классификация MNIST")
    print("=" * 60)

    # Часть 1: MLP с scikit-learn
    print("\n1. Обучение MLP (scikit-learn)")
    print("-" * 40)

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import fetch_openml
    from sklearn.neural_network import MLPClassifier

    # Загрузка датасета MNIST
    print("Загрузка датасета MNIST...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype("int")

    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.2
    )

    # Нормировка данных
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Обучение MLP
    print("Обучение MLP...")
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=20,
        random_state=42
    )
    mlp_model.fit(X_train_scaled, y_train)

    # Предсказания и метрики MLP
    mlp_preds = mlp_model.predict(X_test_scaled)

    print('\nМетрики MLP:')
    print('Accuracy:', accuracy_score(y_test, mlp_preds))
    print('F1 micro:', f1_score(y_test, mlp_preds, average='micro'))
    print('F1 macro:', f1_score(y_test, mlp_preds, average='macro'))

    # Часть 2: CNN с PyTorch
    print("\n\n2. Обучение CNN (PyTorch, LeNet)")
    print("-" * 40)

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    # Загрузка датасета и разделение данных
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_data = datasets.MNIST(root="./data", train=False, transform=transform)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000)


    # Определение архитектуры CNN (LeNet)
    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
            self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

            self.fc1 = nn.Linear(16 * 4 * 4, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.max_pool2d(x, 2)
            x = torch.relu(self.conv2(x))
            x = torch.max_pool2d(x, 2)
            x = x.view(-1, 16 * 4 * 4)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x


    # Обучение модели
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5
    print(f"\nОбучение CNN в течение {num_epochs} эпох...")

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Оценка модели
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Вывод метрик CNN
    print('\nМетрики CNN:')
    print('Accuracy:', accuracy_score(all_targets, all_preds))
    print('F1 micro:', f1_score(all_targets, all_preds, average='micro'))
    print('F1 macro:', f1_score(all_targets, all_preds, average='macro'))

    # Сравнение моделей
    print("\n\n3. Сравнение моделей")
    print("-" * 40)

    mlp_acc = accuracy_score(y_test, mlp_preds)
    cnn_acc = accuracy_score(all_targets, all_preds)

    print(f"\nMLP Accuracy: {mlp_acc:.4f}")
    print(f"CNN Accuracy: {cnn_acc:.4f}")
    print(f"Разница (CNN - MLP): {cnn_acc - mlp_acc:+.4f}")
    print(f"Прирост точности: {(cnn_acc - mlp_acc) * 100:.2f}%")

    # Создание графика сравнения
    plt.figure(figsize=(10, 5))

    models = ['MLP', 'CNN']
    accuracy = [mlp_acc, cnn_acc]

    bars = plt.bar(models, accuracy, color=['skyblue', 'lightgreen'], edgecolor='black')
    plt.ylabel('Accuracy')
    plt.title('Сравнение точности MLP и CNN на MNIST')
    plt.ylim([0.9, 1.0])

    # Добавление значений на столбцы
    for bar, acc in zip(bars, accuracy):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                 f'{acc:.4f}', ha='center', va='bottom')

    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('lab2_results/model_comparison.png', dpi=300)
    print("\nГрафик сохранен: lab2_results/model_comparison.png")

    # Визуализация нескольких примеров из датасета
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X_train[i].reshape(28, 28), cmap='gray')
        ax.set_title(f"Цифра: {y_train[i]}")
        ax.axis('off')
    plt.suptitle('Примеры изображений из датасета MNIST', fontsize=14)
    plt.tight_layout()
    plt.savefig('lab2_results/mnist_samples.png', dpi=300)
    print("Примеры изображений сохранены: lab2_results/mnist_samples.png")

    # Выводы
    print("\n\n4. Выводы")
    print("=" * 60)
    print("""
CNN демонстрирует несколько лучшие значения метрик Accuracy и F1-score 
по сравнению с MLP. Это связано с тем, что CNN учитывает пространственную 
структуру изображений и эффективно извлекает локальные признаки, в то время 
как MLP работает с изображением как с плоским вектором признаков.

Основные преимущества CNN:
1. Учет пространственной структуры данных
2. Инвариантность к небольшим смещениям и искажениям
3. Эффективное извлечение локальных признаков
4. Меньшее количество параметров за счет разделения весов

Несмотря на более высокую точность, CNN требует больше времени на обучение
и вычислительных ресурсов по сравнению с MLP.
""")

    print(f"\nВсе результаты сохранены в папке 'lab2_results/'")
    print("=" * 60)

# Восстанавливаем стандартный вывод
sys.stdout = original_stdout
print("\nЛабораторная работа завершена!")