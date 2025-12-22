import numpy as np
import tensorflow as tf
import random
import os


def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


def prepare_training_data(text, seq_length=50):
    chars = sorted(list(set(text)))

    char_to_idx = {char: i for i, char in enumerate(chars)}
    idx_to_char = {i: char for i, char in enumerate(chars)}

    indices = [char_to_idx[char] for char in text]

    X = []
    y = []

    for i in range(len(indices) - seq_length):
        X.append(indices[i:i + seq_length])
        y.append(indices[i + seq_length])

    X = np.array(X)
    y = np.array(y)

    return X, y, char_to_idx, idx_to_char, chars


def build_model(vocab_size, seq_length=50):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 256),
        tf.keras.layers.LSTM(256, return_sequences=True, stateful=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(256, stateful=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(vocab_size, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def generate_text(model, seed_text, char_to_idx, idx_to_char, seq_length=50, num_chars=500):

    generated = seed_text

    for _ in range(num_chars):
        encoded = [char_to_idx.get(char, 0) for char in seed_text[-seq_length:]]

        if len(encoded) < seq_length:
            encoded = [0] * (seq_length - len(encoded)) + encoded

        encoded = np.array(encoded).reshape(1, -1)

        predictions = model.predict(encoded, verbose=0)[0]

        next_idx = np.random.choice(len(predictions), p=predictions)
        next_char = idx_to_char[next_idx]

        generated += next_char

        seed_text = seed_text[1:] + next_char if len(seed_text) >= seq_length else seed_text + next_char

    return generated


def create_dataset(X, y, batch_size=64):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset


def main():
    print("=" * 60)
    print("ЛАБОРАТОРНАЯ РАБОТА №3: ГЕНЕРАЦИЯ ТЕКСТА НА LSTM")
    print("=" * 60)

    # Параметры
    TRAINING_FILE = "training_text.txt"

    if not os.path.exists(TRAINING_FILE):
        print(f"Файл '{TRAINING_FILE}' не найден")
        print("Создание демонстрационного обучающего текста...")

        demo_text = """Я помню чудное мгновенье:
Передо мной явилась ты,
Как мимолетное виденье,
Как гений чистой красоты.
В томленьях грусти безнадежной,
В тревогах шумной суеты,
Звучал мне долго голос нежный
И снились милые черты.
Шли годы. Бурь порыв мятежный
Рассеял прежние мечты,
И я забыл твой голос нежный,
Твои небесные черты.
В глуши, во мраке заточенья
Тянулись тихо дни мои
Без божества, без вдохновенья,
Без слез, без жизни, без любви.
Душе настало пробужденье:
И вот опять явилась ты,
Как мимолетное виденье,
Как гений чистой красоты.
И сердце бьется в упоенье,
И для него воскресли вновь
И божество, и вдохновенье,
И жизнь, и слезы, и любовь."""

        with open(TRAINING_FILE, 'w', encoding='utf-8') as f:
            f.write(demo_text)
        print(f"Демонстрационный текст создан в '{TRAINING_FILE}'")

    print(f"Загрузка текста из '{TRAINING_FILE}'...")
    training_text = load_text(TRAINING_FILE)
    print(f"Загружено символов: {len(training_text)}")
    print(f"Уникальных символов: {len(set(training_text))}")

    SEQ_LENGTH = 50
    BATCH_SIZE = 64
    EPOCHS = 20
    GENERATION_LENGTH = 5000

    print(f"Подготовка данных (длина последовательности: {SEQ_LENGTH})...")
    X, y, char_to_idx, idx_to_char, chars = prepare_training_data(training_text, SEQ_LENGTH)

    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"Создано обучающих примеров: {len(X)}")
    print(f"Размер словаря: {len(chars)}")

    train_dataset = create_dataset(X_train, y_train, BATCH_SIZE)
    val_dataset = create_dataset(X_val, y_val, BATCH_SIZE)

    print("Создание модели LSTM...")
    model = build_model(len(chars))

    print(f"\nНачало обучения ({EPOCHS} эпох)")

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=1,
        verbose=1
    )

    print("\n" + "=" * 60)
    print("СВОДКА МОДЕЛИ")
    print("=" * 60)
    model.summary()

    if EPOCHS > 1:
        print(f"\nПродолжение обучения (еще {EPOCHS - 1} эпох)")
        model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=EPOCHS - 1,
            verbose=1
        )

    print("Выбор seed для генерации")
    start_idx = random.randint(0, len(training_text) - SEQ_LENGTH - 1)
    seed_text = training_text[start_idx:start_idx + SEQ_LENGTH]
    print(f"Seed текст ({len(seed_text)} символов): '{seed_text}'")

    print(f"Генерация текста ({GENERATION_LENGTH} символов)")
    generated_text = generate_text(
        model=model,
        seed_text=seed_text,
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
        seq_length=SEQ_LENGTH,
        num_chars=GENERATION_LENGTH
    )

    GENERATED_FILE = "generated_text.txt"
    with open(GENERATED_FILE, 'w', encoding='utf-8') as f:
        f.write("СГЕНЕРИРОВАННЫЙ ТЕКСТ LSTM МОДЕЛЬЮ\n")
        f.write("=" * 60 + "\n\n")
        f.write(generated_text)

    print(f"Сохранение сгенерированного текста в '{GENERATED_FILE}'")

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ГЕНЕРАЦИИ")
    print("=" * 60)

    print(f"\nСгенерированный текст (первые 500 символов):")
    print("-" * 60)
    print(generated_text[:500])
    print("-" * 60)

    print(f"\nДлина сгенерированного текста: {len(generated_text)} символов")

    print("\n" + "=" * 60)
    print("ЛАБОРАТОРНАЯ РАБОТА ВЫПОЛНЕНА")
    print("=" * 60)


if __name__ == "__main__":
    main()
