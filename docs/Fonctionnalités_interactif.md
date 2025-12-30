# Fonctionnalités - RenAIme Interactif (Approche Non-Bloquante)

## 🎯 Objectif Global
Script interactif de tri et renommage automatique de documents avec suggestions multiples en temps réel :
- **OCR** : Extraction de texte via Tesseract
- **Analyse IA** : Détection Institution/Objet/Date via Ollama
- **Suggestions interactives** : 5 variantes de noms proposées pour chaque fichier
- **Analyse en arrière-plan** : Traitement parallèle pendant que l'utilisateur choisit
- **Renommage flexible** : Acceptation d'une suggestion, édition manuelle ou rejet

---

## 🏗️ Architecture Globale

### Approche Non-Bloquante (Multi-Threading)

```
┌─────────────────────────────────────────────────────────────┐
│                    MAIN THREAD (UI)                         │
│  - Affiche suggestions                                      │
│  - Attend réponse utilisateur (bloquant)                    │
│  - Exporte fichiers                                         │
│  - Logs CSV                                                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        ↓                                      ↓
┌──────────────────────────────┐   ┌──────────────────────────┐
│  ANALYSIS THREAD             │   │   QUEUE (Thread-Safe)    │
│  - Analyse fichier 1         │   │  - AnalysisResult        │
│  - Analyse fichier 2         │──→│  - Inst/Obj/Date         │
│  - Analyse fichier 3 (...)   │   │  - 5 suggestions         │
│  - Analyse fichier N         │   │  - Tmp PDF path          │
└──────────────────────────────┘   └──────────────────────────┘
```

**Avantage** : Pendant que vous choisissez un nom, le système analyse déjà les fichiers suivants. Zéro temps d'attente.

---

## 📋 Configuration & Initialisation

### Configuration JSON (`config.json`)
- **SOURCE_DIR** : Dossier contenant les fichiers à traiter
- **OLLAMA_MODEL** : Modèle LLM à utiliser (par défaut : `llama3:8b-instruct-q4_0`)

### Interaction Utilisateur
1. Demande interactive du dossier source
2. Liste et sélection du modèle Ollama disponible
3. Création des dossiers `Export_YYYYMMDD_HHMMSS/` et `Echec_YYYYMMDD_HHMMSS/`
4. Initialisation du log CSV horodaté

---

## ✅ Vérification des Dépendances

### Packages Requis
Identique au script principal `renommeur.py` :
- `pdfplumber`, `PIL/Pillow`, `pytesseract`, `ollama`
- Optionnels : `PyPDF2`, `python-docx`, `openpyxl`, `cv2/opencv`, `pdf2image`

---

## 🔍 Multi-Variant Detection (Nouveau en v2.1)

### Détection Multi-Variante
Le script interactif utilise désormais la **détection multi-variante** :
- **Ollama génère 3 candidats** pour Institution et Objet (au lieu d'une seule valeur)
- **Sélection intelligente** : Le système choisit la variante la plus proche du titre détecté
- **Meilleure précision** : Plusieurs candidats augmentent la chance d'une détection correcte

### Fonctions de Sélection
1. **`title_from_first_page(first_page_text)`** : Extrait le titre du document depuis la première page
2. **`best_match_with_title(variants, title_text)`** : Compare les 3 variantes avec le titre et retourne la meilleure
3. **`simplify_institution_name(name)`** : Nettoie l'institution (supprime articles et formes juridiques)

Voir `docs/Multi_Variant_Detection.md` pour les détails techniques.

---

## 🔄 Flux de Traitement (Non-Bloquant)

### 1️⃣ **Démarrage du Thread Analyse**

```python
analysis_thread = threading.Thread(
    target=analysis_worker,
    args=(source_dir, model, log_path),
    daemon=False
)
analysis_thread.start()
```

Le thread `analysis_worker()` :
- Itère sur TOUS les fichiers du dossier source
- Extrait texte, dates, analyse Ollama pour CHAQUE fichier
- Envoie résultats dans `analysis_queue` dès qu'un fichier est traité
- Continue l'analyse même pendant que l'utilisateur choisit

### 2️⃣ **Main Thread - Affichage et Interaction**

```python
while True:
    result = analysis_queue.get(timeout=1)  # Attendre résultat
    
    # Afficher
    print(f"[FICHIER] {result.filename}")
    print(f"  Institution: {result.inst}")
    print(f"  Objet: {result.obj}")
    print(f"  Date: {result.date}")
    
    # Afficher 5 suggestions
    for i, sugg in enumerate(result.suggestions, 1):
        print(f"    {i}. {sugg}")
    
    # Attendre choix utilisateur (UI bloquante, analyse continue en BG)
    choix = input("Choix (1-5) ou 'q' ou 'e' : ").strip()
    
    # Traiter choix et exporter
```

---

## 🎯 Système de Suggestions (5 Variantes)

Pour chaque fichier, **5 noms alternatifs** sont générés :

### Format 1: Standard (YYYY-MM Institution Objet)
```
2024-12 Banque De France Releve De Compte.pdf
```
→ Meilleur pour tri chronologique puis par institution

### Format 2: Objet en Premier (YYYY-MM Objet Institution)
```
2024-12 Releve De Compte Banque De France.pdf
```
→ Meilleur pour retrouver par type de document

### Format 3: Institution en Tête (Institution YYYY-MM Objet)
```
Banque De France 2024-12 Releve De Compte.pdf
```
→ Meilleur pour grouper par organisme

### Format 4: Date à la Fin (Institution Objet YYYY-MM)
```
Banque De France Releve De Compte 2024-12.pdf
```
→ Lisibilité maximale

### Format 5: Format Compact (YYYY-MM-Institution-Objet)
```
2024-12-Banque-De-France-Releve-De-Compte.pdf
```
→ Compatible web/URLs

---

## 💬 Interaction Utilisateur

### Options au Moment du Choix

```
  Suggestions de noms:
    1. 2024-12 Banque De France Releve De Compte.pdf
    2. 2024-12 Releve De Compte Banque De France.pdf
    3. Banque De France 2024-12 Releve De Compte.pdf
    4. Banque De France Releve De Compte 2024-12.pdf
    5. 2024-12-Banque-De-France-Releve-De-Compte.pdf

  Choix (1-5) ou 'q' pour rejeter, 'e' pour personnaliser :
```

#### Réponse `1-5` : Accepter une suggestion
- Fichier copié avec le nom choisi → `Export_YYYYMMDD_HHMMSS/`
- PDF OCRisé (si généré) également copié
- Log CSV : Statut "Succès"

#### Réponse `q` : Rejeter le fichier
- Fichier copié → `Echec_YYYYMMDD_HHMMSS/`
- Log CSV : Statut "Rejeté"
- Pas d'export

#### Réponse `e` : Éditer manuellement
```
  Nouveau nom: 2024-12 Ma Banque Relevé Personnel.pdf
```
- Vous tapez un nom personnalisé
- Fichier exporté avec ce nom
- Log CSV : Statut "Succès (personnalisé)"

---

## 📊 Logging & Traçabilité

### Fichier CSV Horodaté
Créé dans `SOURCE_DIR/Export_YYYYMMDD_HHMMSS/log_YYYYMMDD_HHMMSS.csv`

Colonnes :
- `Fichier` : Nom original
- `Statut` : "Succès", "Succès (personnalisé)", "Rejeté", ou "Échec"
- `Nouveau nom` : Nom final attribué
- `Institution` : Valeur détectée
- `Objet` : Valeur détectée
- `Date` : Valeur détectée

### Affichage Console

```
[DÉMARRAGE] Analyse en arrière-plan des fichiers...

[FICHIER 1] document1.pdf
  Institution: Banque De France
  Objet: Releve De Compte
  Date: 2024-12

  Suggestions de noms:
    1. 2024-12 Banque De France Releve De Compte.pdf
    2. 2024-12 Releve De Compte Banque De France.pdf
    3. Banque De France 2024-12 Releve De Compte.pdf
    4. Banque De France Releve De Compte 2024-12.pdf
    5. 2024-12-Banque-De-France-Releve-De-Compte.pdf

  Choix (1-5) ou 'q' pour rejeter, 'e' pour personnaliser : 1
  ✅ EXPORTÉ: 2024-12 Banque De France Releve De Compte.pdf

[FICHIER 2] document2.jpg
  Institution: Mairie De Paris
  Objet: Certificat Scolarite
  Date: 2024-11

  Suggestions de noms:
    ...
```

---

## 🚀 Performance & Avantages

### Gains de Performance

**Sans Threading (Approche 1)** :
```
Fichier 1: Analyse (5s) → Choix (10s) → Export (1s)
Fichier 2: Analyse (5s) → Choix (10s) → Export (1s)
Fichier 3: Analyse (5s) → Choix (10s) → Export (1s)
───────────────────────
Total: 51 secondes
```

**Avec Threading (Approche 2)** :
```
FIC 1: Analyse (5s) → [FIC 2 & 3 s'analysent en parallèle] → Choix (10s) → Export (1s)
FIC 2: [Analyse pendant choix FIC 1] → Affichage immédiat → Choix (10s) → Export (1s)
FIC 3: [Analyse pendant choix FIC 1 & 2] → Affichage immédiat → Choix (10s) → Export (1s)
───────────────────────
Total: ~30 secondes (-41% vs approche 1)
```

### Avantages Utilisateur

✅ **Zéro temps d'attente** : Suggestions disponibles immédiatement après le fichier précédent  
✅ **Flexibilité** : 5 formats propose différents ordres pour s'adapter à vos besoins  
✅ **Contrôle total** : Modifier chaque nom, rejeter sans danger  
✅ **Transparence** : Voir Institution/Objet/Date détectés avant de choisir  
✅ **Traçabilité** : CSV log avec tous les choix effectués  

---

## 🛡️ Gestion des Erreurs

### Erreurs d'Extraction Texte
```
[FICHIER 1] document.pdf
  ❌ ERREUR: Aucun texte détecté
```
→ Fichier copié dans `Echec_YYYYMMDD_HHMMSS/`

### Erreurs Ollama
```
[FICHIER 2] document2.jpg
  ❌ ERREUR: Erreur Ollama
```
→ Fichier copié dans `Echec_YYYYMMDD_HHMMSS/`

### Interruption Utilisateur (Ctrl+C)
```
[⚠️  INTERRUPTION] Signal SIGINT reçu - Nettoyage en cours...
[NETTOYAGE] Fichier temporaire supprimé: /tmp/tmpXXXXXX.pdf
[✓] Nettoyage terminé. Au revoir!
```
→ Tous les PDF temporaires supprimés, thread arrêté proprement

---

## 📁 Structure Finale

```
SOURCE_DIR/
├── document1.pdf (original)
├── document2.jpg (original)
├── Export_20251230_143022/
│   ├── 2024-12 Banque De France Releve De Compte.pdf
│   ├── 2024-12 Banque De France Releve De Compte.pdf (PDF OCRisé si image)
│   ├── 2024-11 Mairie De Paris Certificat Scolarite.jpg
│   └── log_20251230_143022.csv
└── Echec_20251230_143022/
    └── unrecognized_file.pdf (si rejeté ou erreur)
```

---

## 🎛️ Utilisation

### Lancer le script
```bash
python3 renommeur_interactif.py
```

### Flux complet
1. Entrez dossier source : `/chemin/vers/documents`
2. Choisissez modèle Ollama : `2` (pour llama3:8b-instruct-q4_0)
3. Attendez l'analyse du premier fichier
4. Choisissez un nom : `1` (ou `e` pour éditer, ou `q` pour rejeter)
5. Répétez pour chaque fichier
6. Résumé final avec log CSV

---

## 📝 Notes de Développement

### Points Forts
- **Thread-safe** : Utilisation de `queue.Queue()` pour communication sécurisée
- **Résilience** : Nettoyage des fichiers temp même en cas d'interruption
- **Flexibilité** : 5 formats proposés couvrent la plupart des usages
- **Transparence** : Voir les champs détectés avant de décider

### Possibilités d'Extension
- Ajouter plus de formats de suggestion (8-10 au lieu de 5)
- Historique des choix pour suggestions futures
- Batch acceptance (accepter format similaire pour plusieurs fichiers)
- Intégration avec un gestionnaire de fichiers pour double-clic
- Support de regex personnalisée pour édition rapide

### Limitations Connues
- Pas de sauvegarde en cas de crash du système (rarement nécessaire)
- Si Ollama crash, l'analyse du fichier courant est perdue (relancer le script)
- Les PDFs temporaires occupent de l'espace disque (nettoyés à la fin)

---

## ✨ Résumé Fonctionnel

**Script interactif, intelligente et parallèle** pour :
- ✅ OCR intelligent (PDF + Images)
- ✅ Analyse IA sophistiquée
- ✅ 5 suggestions de noms pour flexibilité
- ✅ **Analyse non-bloquante** (threading)
- ✅ Édition manuelle des noms
- ✅ Rejet sélectif sans risque
- ✅ Traçabilité complète (CSV)
- ✅ Nettoyage automatique

**Gain de temps** : -40% vs approche séquentielle grâce au threading.
