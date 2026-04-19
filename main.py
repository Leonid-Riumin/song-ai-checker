import os
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

HUMAN_DIR = r"C:\Users\rlv11\Desktop\Project\data\human"
AI_DIR = r"C:\Users\rlv11\Desktop\Project\data\ai"
TEST_DIR = r"C:\Users\rlv11\Desktop\Project\test"


def safe_scalar(x):
    arr = np.asarray(x)
    return float(arr.reshape(-1)[0])


def extract_features(path: str) -> np.ndarray:
    y, sr = librosa.load(path, sr=22050, mono=True, duration=20)

    if y is None or len(y) == 0:
        raise ValueError("fichier audio vide ou endommagé")

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
        raise FileNotFoundError(f"Dossier introuvable : {human_dir}")
    if not os.path.exists(ai_dir):
        raise FileNotFoundError(f"Dossier introuvable : {ai_dir}")

    human_files = [f for f in os.listdir(human_dir) if f.lower().endswith((".wav", ".mp3"))]
    ai_files = [f for f in os.listdir(ai_dir) if f.lower().endswith((".wav", ".mp3"))]

    if len(human_files) == 0:
        raise ValueError(f"Il n'y a pas de fichiers .wav ou .mp3 dans le dossier {human_dir}")
    if len(ai_files) == 0:
        raise ValueError(f"Il n'y a pas de fichiers .wav ou .mp3 dans le dossier {ai_dir}")

    print("Chargement des fichiers HUMAINS...")
    for fname in human_files:
        path = os.path.join(human_dir, fname)
        try:
            feats = extract_features(path)
            X.append(feats)
            y.append(0)
            print(f"OK: {fname}")
        except Exception as e:
            print(f"Erreur lors du traitement de {fname} : {e}")

    print("\nChargement des fichiers IA...")
    for fname in ai_files:
        path = os.path.join(ai_dir, fname)
        try:
            feats = extract_features(path)
            X.append(feats)
            y.append(1)
            print(f"OK: {fname}")
        except Exception as e:
            print(f"Erreur lors du traitement de {fname} : {e}")

    if len(X) < 4:
        raise ValueError("Trop peu de fichiers traités avec succès pour l'entraînement. Au moins 4 fichiers sont nécessaires.")

    return np.array(X, dtype=np.float32), np.array(y)


def predict_folder(folder_path: str, model):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Dossier de test introuvable : {folder_path}")

    test_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".wav", ".mp3"))]

    if len(test_files) == 0:
        print("Aucun fichier audio (.wav ou .mp3) n'a été trouvé dans le dossier de test.")
        return

    print("\n=== Analyse des fichiers du dossier test ===")
    for fname in test_files:
        path = os.path.join(folder_path, fname)
        try:
            feats = extract_features(path).reshape(1, -1)
            prob_ai = model.predict_proba(feats)[0][1]

            if prob_ai > 0.7:
                label = "IA"
            elif prob_ai < 0.3:
                label = "Humain"
            else:
                label = "Incertain"

            print(f"{fname} -> {label} (probabilité IA = {prob_ai:.2f})")
        except Exception as e:
            print(f"Erreur lors de l'analyse de {fname} : {e}")


def main():
    print("Dossier de démarrage actuel :", os.getcwd())
    print("Le dossier humain existe-t-il ?", os.path.exists(HUMAN_DIR))
    print("Le dossier ai existe-t-il ?", os.path.exists(AI_DIR))
    print("Le dossier test existe-t-il ?", os.path.exists(TEST_DIR))
    print()

    X, y = load_dataset(HUMAN_DIR, AI_DIR)

    if len(np.unique(y)) < 2:
        raise ValueError("Les deux classes (humain et IA) sont nécessaires pour l'entraînement.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print("\n=== Rapport de classification ===")
    print(classification_report(y_test, pred, target_names=["human", "ai"]))

    predict_folder(TEST_DIR, model)


if __name__ == "__main__":
    main()
