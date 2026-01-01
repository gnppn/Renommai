# Fichiers nécessaires par script

## renommeur.py (Script principal)

### Fichiers requis
```
renommeur.py              # Script principal
requirements.txt          # Dépendances Python
config.json               # Configuration (créé automatiquement)
prompts/
  ├── ollama_analysis_4k.txt
  ├── ollama_analysis_8k.txt
  ├── ollama_analysis_16k.txt
  └── vision_prompt.txt
```

### Description
Version standard qui laisse l'IA extraire librement l'Institution et l'Objet du document.

---

## renommeur_categories_fixes.py (Script expérimental)

### Fichiers requis
```
renommeur_categories_fixes.py   # Script expérimental
requirements.txt                # Dépendances Python
config.json                     # Configuration (créé automatiquement)
categories_documents.txt        # Liste restrictive des types de documents
institutions_cache.txt          # Cache des institutions connues
prompts/
  ├── ollama_analysis_4k.txt
  ├── ollama_analysis_8k.txt
  ├── ollama_analysis_16k.txt
  └── vision_prompt.txt
```

### Description
Version expérimentale qui utilise des listes restrictives pour :
- **categories_documents.txt** : ~100 types de documents administratifs prédéfinis
- **institutions_cache.txt** : ~250 variantes d'institutions connues

L'IA est guidée par ces listes et les résultats sont normalisés automatiquement.

---

## Fichiers communs

| Fichier | Description |
|---------|-------------|
| `requirements.txt` | Dépendances Python communes |
| `prompts/` | Prompts pour l'IA (analyse et vision) |
| `config.json` | Configuration sauvegardée (dossier source, modèle) |

## Fichiers de développement/test

| Fichier | Description |
|---------|-------------|
| `renommeur_simplifié.py` | Version simplifiée (prototype) |
| `reanalyze_rejected.py` | Réanalyse des fichiers rejetés |
| `test_*.py` | Scripts de test |
| `debug_*/` | Dossiers de debug |
| `docs/` | Documentation additionnelle |
