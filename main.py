import os
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

HUMAN_DIR = r"D:\Рабочий стол\Project\data\human"
AI_DIR = r"D:\Рабочий стол\Project\data\ai"
TEST_FILE = r"D:\Рабочий стол\Project\test_song.mp3"


def safe_scalar(x):
    arr = np.asarray(x)
    return float(arr.reshape(-1)[0])


def extract_features(path: str) -> np.ndarray:
    y, sr = librosa.load(path, sr=22050, mono=True, duration=20)

    if y is None or len(y) == 0:
        raise ValueError("Пустой или повреждённый аудиофайл")

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_value = safe_scalar(tempo)

    feature_vector = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),

        np.mean(contrast, axis=1),
        np.std(contrast, axis=1),

        np.array([safe_scalar(np.mean(flatness))]),
        np.array([safe_scalar(np.std(flatness))]),

        np.array([safe_scalar(np.mean(zcr))]),
        np.array([safe_scalar(np.std(zcr))]),

        np.mean(chroma, axis=1),
        np.std(chroma, axis=1),

        np.array([tempo_value])
    ])

    return feature_vector.astype(np.float32)


def load_dataset(human_dir: str, ai_dir: str):
    X = []
    y = []

    if not os.path.exists(human_dir):
        raise FileNotFoundError(f"Папка не найдена: {human_dir}")
    if not os.path.exists(ai_dir):
        raise FileNotFoundError(f"Папка не найдена: {ai_dir}")

    human_files = [f for f in os.listdir(human_dir) if f.lower().endswith((".wav", ".mp3"))]
    ai_files = [f for f in os.listdir(ai_dir) if f.lower().endswith((".wav", ".mp3"))]

    if len(human_files) == 0:
        raise ValueError(f"В папке {human_dir} нет файлов .wav или .mp3")
    if len(ai_files) == 0:
        raise ValueError(f"В папке {ai_dir} нет файлов .wav или .mp3")

    print("Загружаю HUMAN файлы...")
    for fname in human_files:
        path = os.path.join(human_dir, fname)
        try:
            feats = extract_features(path)
            X.append(feats)
            y.append(0)
            print(f"OK: {fname}")
        except Exception as e:
            print(f"Ошибка при обработке {fname}: {e}")

    print("\nЗагружаю AI файлы...")
    for fname in ai_files:
        path = os.path.join(ai_dir, fname)
        try:
            feats = extract_features(path)
            X.append(feats)
            y.append(1)
            print(f"OK: {fname}")
        except Exception as e:
            print(f"Ошибка при обработке {fname}: {e}")

    if len(X) < 4:
        raise ValueError("Слишком мало успешно обработанных файлов для обучения. Нужно хотя бы 4 файла.")

    return np.array(X, dtype=np.float32), np.array(y)


def predict_file(path: str, model):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Тестовый файл не найден: {path}")

    feats = extract_features(path).reshape(1, -1)
    prob_ai = model.predict_proba(feats)[0][1]

    if prob_ai > 0.7:
        label = "IA"
    elif prob_ai < 0.3:
        label = "Humain"
    else:
        label = "Incertain"

    return label, float(prob_ai)


def main():
    print("Текущая папка запуска:", os.getcwd())
    print("Папка human существует?", os.path.exists(HUMAN_DIR))
    print("Папка ai существует?", os.path.exists(AI_DIR))
    print()

    X, y = load_dataset(HUMAN_DIR, AI_DIR)

    if len(np.unique(y)) < 2:
        raise ValueError("Нужны файлы и в human, и в ai.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print("\n=== Rapport de classification ===")
    print(classification_report(y_test, pred, target_names=["human", "ai"]))

    print("\n=== Test d'un fichier ===")
    label, score = predict_file(TEST_FILE, model)
    print(f"Résultat: {label}, probabilité IA = {score:.2f}")


if __name__ == "__main__":
    main()
