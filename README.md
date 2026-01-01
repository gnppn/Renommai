# Renommai

⚠️ Petit projet de vacances **vibecodé**. N'en attendez rien de fiable.

## Quoi

 - Un outil pour trier et renommer par IA des documents administratifs
 - Sources images ou PDF acceptés
 - Analyse du contenu :
    - (OCR) des fichiers
    - extraction du contenu par IA (llava via ollama)
    - renommage guidé par IA (modèle de votre choix sur ollama)
 - Sortie : `Date Institution Objet.ext`
 - L'IA extrait 3 possibilités pour l'institution et l'objet, et retient la meilleure.
 - L'IA tente d'harmoniser-simplifier la sortie.
 - Ecrit pour un vieux laptop peu puissant.

## Scripts disponibles

### `renommeur.py` - Version principale
Script standard qui laisse l'IA extraire librement l'Institution et l'Objet.

```bash
python renommeur.py
```

### `renommeur_categories_fixes.py` - Version expérimentale 🧪
Script avec **listes restrictives** pour normaliser les résultats :
- **categories_documents.txt** : ~100 types de documents administratifs prédéfinis (impôts, banque, assurance, salariat, etc.)
- **institutions_cache.txt** : ~250 variantes d'institutions connues (DGFIP, CPAM, banques, assureurs, etc.)

L'IA est guidée par ces listes et les résultats sont automatiquement normalisés via fuzzy matching.

```bash
python renommeur_categories_fixes.py
```

> Voir [FICHIERS.md](FICHIERS.md) pour le détail des fichiers nécessaires à chaque script.

## Installation éclair

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Pré-requis rapides

- Python 3.10+ (venv recommandé).
- Dépendances clés : `pytesseract`, `pdfplumber`, `pypdf`, `pdf2image`, `ollama`.
- Tesseract installé sur le système (langue fra).
- Modèle Ollama disponible localement (`ollama pull mistral:7b-instruct-v0.3-q4_K_M`).

## Lancement minimal

```bash
python renommeur.py
```
- Choisir le dossier source (par défaut `documents`).
- Les exports et échecs sont créés dans `Export_<timestamp>` et `Echec_<timestamp>`.
- Un log CSV est ajouté dans le dossier d’export.
- Interrompez le script **avec Ctrl+C**

## Notes de fiabilité

- Dates exigent `YYYY-MM` ou, à défaut, `YYYY`.
- En cas de doute, les champs passent à `inconnu`.
- Les OCR bruyants ou les scans de mauvaise qualité peuvent dégrader les résultats.

## Tests

- Tests rapides : `pytest tests/test_ocr_pipeline.py` (utilise `./Fichiers test`).
- Pipeline complet : `pytest tests/test_full_pipeline.py` (peut être long/IO-heavy).

## État d’esprit

- Ce dépôt est vraiment pensé en __one shot__ pour un besoin précis. Mais si ça peut servir...
