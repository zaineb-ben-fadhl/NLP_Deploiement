# drift_data_gen.py
import pandas as pd
import numpy as np
import os
from fastapi import HTTPException

def generate_reference_csv(reference_file="data/drift_reference.csv"):
    """Crée un CSV de référence avec la colonne 'text' si elle n'existe pas"""
    os.makedirs(os.path.dirname(reference_file), exist_ok=True)

    if os.path.exists(reference_file):
        df = pd.read_csv(reference_file)
        if "text" in df.columns:
            return reference_file
        else:
            print(f"❌ La colonne 'text' est manquante dans {reference_file}, création d'un nouveau fichier...")
    else:
        print(f"ℹ️ CSV de référence introuvable, création d'un nouveau fichier: {reference_file}")

    # Exemple de textes pour le CSV
    sample_texts = [
        "Je me sens stressé aujourd'hui.",
        "Tout va bien, je suis détendu.",
        "Je suis anxieux à propos de mon travail.",
        "C'est une belle journée, je suis heureux.",
        "Je suis très nerveux avant la réunion.",
        "Je me sens calme et relaxé."
    ]

    df = pd.DataFrame({"text": sample_texts})
    df.to_csv(reference_file, index=False)
    print(f"✅ CSV de référence créé: {reference_file}")
    return reference_file


def generate_drifted_text_data(
    reference_file: str = "data/drift_reference.csv",
    output_file: str = "data/drift_production.csv",
    drift_level: str = "medium"
):
    """Génère un CSV drifté sur la colonne 'text'"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    ref = pd.read_csv(reference_file)
    if "text" not in ref.columns:
        raise HTTPException(status_code=400, detail="❌ La colonne 'text' est manquante dans le CSV de référence.")

    prod = ref.copy()

    np.random.seed(42)
    drift_map = {"low": 0.05, "medium": 0.15, "high": 0.30}
    intensity = drift_map.get(drift_level, 0.15)

    def drift_text(text: str) -> str:
        words = text.split()
        n_words = len(words)
        if n_words < 5:
            return text
        if np.random.rand() < intensity:
            cut = max(1, int(n_words * intensity))
            words = words[:-cut]  # tronque quelques mots
        elif np.random.rand() < intensity:
            repeat = max(1, int(n_words * intensity))
            words = words + words[:repeat]  # répète quelques mots
        return " ".join(words)

    prod["text"] = prod["text"].astype(str).apply(drift_text)
    prod.to_csv(output_file, index=False)
    print(f"✅ CSV de production drifté généré: {output_file}")
    return output_file


if __name__ == "__main__":
    ref_file = generate_reference_csv()
    generate_drifted_text_data(reference_file=ref_file)
