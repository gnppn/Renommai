# Fonctionnalités - RenAIme OCR & Renommage de Documents

## 🎯 Objectif Global
Script de tri et renommage automatique de documents (PDF, PNG, JPG, DOCX, XLSX) basé sur :
- **OCR** : Extraction de texte via Tesseract
- **Analyse Vision** : Pré-analyse visuelle via modèle vision adaptatif (minicpm-v ou llava-llama3)
- **Analyse IA** : Détection Institution/Objet/Date via Ollama (llama3)
- **Renommage** : Génération automatique de noms fichiers selon le format `YYYY-MM Institution Objet.ext` (en Title Case)

---

## 📋 Configuration & Initialisation

### Configuration JSON (`config.json`)
- **SOURCE_DIR** : Dossier contenant les fichiers à traiter (demandé à l'utilisateur au lancement)
- **OLLAMA_MODEL** : Modèle LLM à utiliser (par défaut : `llama3:8b-instruct-q4_0`)

Note: Les dossiers `Export_YYYYMMDD_HHMMSS/` et `Echec_YYYYMMDD_HHMMSS/` sont créés automatiquement **dans le dossier source** avec timestamp pour traçabilité.

### Chargement de la Configuration
- Lecture du fichier `config.json` s'il existe
- Fusion avec les valeurs par défaut (`DEFAULT_CONFIG`)
- Validation des chemins sources (demande interactive si dossier invalide)
- Invite simple : "Dossier source [documents]: "
  - Appui sur Entrée → utilise le défaut de config.json
  - Nouvelle saisie → remplace le défaut
- Sauvegarde de la nouvelle configuration dans `config.json`

### Sélection Interactive du Modèle Ollama
- Affichage des modèles disponibles via `ollama list`
- Sélection par numéro ou validation du modèle actuel
- Feedback utilisateur et confirmation du modèle choisi

---

## ✅ Vérification des Dépendances

### Packages Requis (Critiques)
- `pdfplumber` : Extraction de texte/images de PDF
- `PIL/Pillow` : Manipulation d'images (redimensionnement, filtres)
- `pytesseract` : Interface Python pour Tesseract OCR
- `ollama` : API Python pour intégration LLM locale

### Packages Optionnels
- `pypdf` : Fusion de pages PDF OCRisées (searchable PDFs)
- `python-docx` : Extraction texte de documents Word (.docx)
- `openpyxl` : Extraction texte de feuilles Excel (.xlsx)
- `cv2` (OpenCV) + `numpy` : Prétraitement OCR avancé (CLAHE, Otsu, morphologie)
- `pdf2image` : Fallback pour conversion PDF → images

### Vérification des Modèles Ollama

Fonction : `ensure_models()`
- **Détection puissance système** : RAM + VRAM GPU (nvidia-smi)
- **Sélection modèle vision adaptatif** :
  - PC faible (low) : `minicpm-v:latest` - Léger et efficace
  - PC moyen/puissant (medium/high) : `llava-llama3:latest` - Plus performant
- Vérifie la présence des modèles via `ollama list`
- **Téléchargement automatique** des modèles manquants via `ollama pull`
- Affichage du statut pour chaque modèle (présent ou téléchargé)

### Gestion des Erreurs
- Arrêt du script si une dépendance critique est manquante
- Message clair avec commande `pip install` suggérée
- Arrêt si Ollama n'est pas installé ou pas dans le PATH

---

## 🔄 Traitement de Fichiers

### Types de Fichiers Supportés
- **PDF** : Documents numériques (texte ou image)
- **PNG/JPG/JPEG** : Images scannées
- **DOCX** : Documents Word (extraction texte directe)
- **XLSX** : Feuilles Excel (extraction texte de cellules)

### Flux de Traitement Principal

#### 1️⃣ **Extraction de Texte selon le Type**

##### Pour les PDF :
- Extraction par page via `pdfplumber.open()`
- Si texte détecté : Utilisation directe du texte natif
- Si pas de texte (PDF image) : Déclenchement de l'OCR complet

##### Pour les Images :
- Charge l'image via PIL
- **Autorotation** : Détection automatique de l'orientation via pytesseract OSD
  - Rotation à 360° - angle détecté si nécessaire
- **Prétraitement d'image** pour meilleure reconnaissance OCR

##### Pour les DOCX :
- Extraction directe du texte via `python-docx`
- Récupération de tout le contenu (paragraphes)
- Pas d'OCR nécessaire (texte natif)

##### Pour les XLSX :
- Extraction du texte via `openpyxl`
- Parcours de toutes les feuilles et cellules
- Concaténation du contenu pour analyse complète

#### 2️⃣ **Prétraitement OCR Avancé**

Fonction : `preprocess_image_for_ocr(img)`

**Étapes de traitement :**
1. Conversion en niveaux de gris
2. Filtre médian pour nettoyage du bruit (taille 3)
3. **Si cv2 disponible** (mode avancé) :
   - CLAHE (Contrast Limited Adaptive Histogram Equalization) - clipLimit=2.0
   - Renforcement du contraste global (1.5x)
   - Threshold adaptatif Otsu pour binarisation optimale
   - Morphologie (close) avec kernel 2x2
4. **Si cv2 non disponible** (fallback Pillow) :
   - Renforcement contraste simple (1.5x)
5. Filtre de netteté final (SHARPEN)

#### 3️⃣ **Génération de PDF OCRisé Searchable**

##### Architecture PDF Hybrid (Searchable)

Tous les PDFs OCRisés générés combinent deux couches :

```
PDF Final
├── Layer 1 (Visuelle): Image originale préservée
└── Layer 2 (Texte): Couche OCR caché mais searchable (HOCR)
   → Permet recherche Ctrl+F, copie/sélection texte
   → Invisible à l'écran (texte positionné sous l'image)
```

Fonction : `create_searchable_pdf_page(img, vision_description=None)`
1. Génère PDF image simple (PIL → PDF)
2. Génère PDF OCR avec tesseract (HOCR = texte caché)
3. Fusionne les deux layers (pypdf merge_page)
4. **Enrichissement optionnel** : Si `vision_description` fournie (de llava), l'intègre dans la couche texte
5. Retourne bytes du PDF searchable

##### Pour PDF sans texte :
- Conversion chaque page en image haute résolution (300 DPI)
- Essai 1 : pdfplumber.to_image()
- Essai 2 : Fallback pdf2image si peu/pas d'images
- Autorotation automatique (Tesseract OSD)
- Prétraitement OCR avancé
- Génération PDF searchable par page
- Fusion en PDF multi-pages (pypdf PdfWriter)
- Stockage en fichier temporaire (`/tmp/tmp*.pdf`)
- Retour : texte complet + chemin temp PDF

##### Pour Images PNG/JPG :
- Autorotation détectée via Tesseract OSD
- Prétraitement OCR avancé
- Extraction OCR via Tesseract (lang="fra")
- Génération PDF searchable
- Retour : texte + chemin temp PDF

#### 4️⃣ **Analyse Vision avec modèle adaptatif (Pré-Analyse)**

**⚠️ Effectuée AVANT extraction de dates et AVANT Ollama**

Fonction : `analyze_vision(image_path, model=None)`

##### Sélection du modèle vision
- Fonction `get_system_power_level()` détecte RAM et VRAM
- Fonction `select_vision_model()` choisit le modèle optimal :
  - **PC faible** (RAM < 16GB, VRAM < 4GB) : `minicpm-v:latest`
  - **PC moyen/puissant** : `llava-llama3:latest`

##### Objectif
Analyser visuellement la PREMIÈRE PAGE UNIQUEMENT du document pour extraire une description concise (institution, type, date visibles).

##### Processus
1. **Préparation de l'image** :
   - Pour images : utilisation directe du fichier
   - Pour PDF OCRisé : extraction de la 1ère page en image PNG temporaire
2. **Encodage** : Image → base64
3. **Prompt vision** (chargé depuis `prompts/vision_prompt.txt`)
4. **Appel modèle vision** : `ollama.generate()` avec image encodée
5. **Limite** : Maximum 4000 caractères de réponse

##### Usage
- La description vision enrichit la recherche de dates
- Elle est intégrée dans le prompt Ollama pour contexte additionnel
- Elle est intégrée dans les PDFs searchable (améliore la recherche)

#### 5️⃣ **Extraction de Dates Candidates**

**⚠️ Effectuée APRÈS analyse vision, AVANT Ollama**

Fonction : `extract_dates(text)`

##### Source d'extraction
- Texte combiné : description llava + texte OCR/natif

##### Formats reconnus (par ordre de priorité)
1. **YYYY-MM** ou **YYYY-MM-DD** (ISO standard) - priorité absolue
2. **DD/MM/YYYY** ou **D/M/YYYY** (European) → normalise en YYYY-MM
3. **YYYY/MM/DD** → normalise en YYYY-MM
4. **Mois nommés** : "DD mois YYYY" (français/anglais) → normalise en YYYY-MM
5. **YYYY seul** (fallback) - année valide uniquement

##### Validation stricte
- Plage temporelle : `[Année actuelle - 20]` ≤ YYYY ≤ `[Année actuelle + 1]`
- Mois : 01-12
- Jour : 01-31 (si présent)
- *Exemple en 2025* : Accepte 2005-2026

##### Retour
- **Maximum 1 date** (première trouvée valide)
- Format : `YYYY-MM` ou `YYYY`
- Dédupliplication automatique

#### 6️⃣ **Régénération des PDFs avec Enrichissement Vision**

**⚠️ Si analyse vision disponible**

Après l'analyse llava, les PDFs sont régénérés avec la description vision intégrée :
- Appel à `ocr_pdf()` ou `extract_from_image()` avec paramètre `vision_description`
- La description est ajoutée à la couche texte searchable
- Améliore la recherche textuelle dans les PDFs générés

#### 7️⃣ **Analyse IA avec Ollama - Multi-Variante**

Fonction : `analyze_ollama(text, dates, model, vision_analysis=None, pass_level="initial")`

##### Optimisation: Première Page (~3500 chars)

Fonction : `extract_first_page(text)`
- Extrait les lignes jusqu'à ~3500 caractères
- Correspond à environ la 1ère page du document
- Optimisé pour la fenêtre de contexte 4K

##### Intégration Vision
Si `vision_analysis` fournie :
```
[ANALYSE VISION]
{description llava}

[TEXTE OCR]
{texte première page}
```

##### Prompt Ollama (Format Multi-Variante avec 3 Candidats)

```
Tu es un assistant d'analyse de documents. Analyse le texte ci-dessous (première page d'un document) et extrais STRICTEMENT trois champs : Institution, Objet et Date.

IMPORTANT: Pour Institution et Objet, propose 3 variantes différentes classées par confiance (Variante 1 = plus probable, Variante 3 = moins probable).

INSTRUCTIONS DÉTAILLÉES:

1. INSTITUTION (Nom de l'émetteur/organisme)
   - Identifie l'organisation qui émet le document
   - Simplifie AGRESSIVEMENT : supprime articles, formes juridiques
   - Propose 3 variantes différentes (de la plus à la moins probable)
   - Format : Title Case
   - Si impossible, retourne "inconnu"

2. OBJET (Type/Nature du document)
   - Déduis le type GÉNÉRAL du document à partir de son contenu
   - Exemples : "Facture", "Releve Bancaire", "Contrat De Travail", "Fiche De Paie"
   - Propose 3 variantes différentes (de la plus à la moins probable)
   - Format : Title Case, court et descriptif (2-5 mots)
   - Si le type n'est pas identifiable, retourne "inconnu"

3. DATE (Horodatage du document)
   - Cherche la date d'émission du document
   - Format attendu : YYYY-MM (année-mois)
   - Format accepté : YYYY (année seule) en dernier recours
   - Candidates prioritaires : {dates}
   - Si aucune date fiable, retourne "inconnu"

FORMAT DE SORTIE STRICT (chaque ligne sur une nouvelle ligne):
Institution Variante 1: <valeur>
Institution Variante 2: <valeur>
Institution Variante 3: <valeur>
Objet Variante 1: <valeur>
Objet Variante 2: <valeur>
Objet Variante 3: <valeur>
Date: <valeur>

Exemple:
Institution Variante 1: Banque De France
Institution Variante 2: Banque Nationale
Institution Variante 3: inconnu
Objet Variante 1: Releve De Compte
Objet Variante 2: Releve Bancaire
Objet Variante 3: Document Bancaire
Date: 2024-12

Texte du document:
{text}
```

##### Appel Ollama
- Modèle configurable (par défaut : `llama3:8b-instruct-q4_0`)
- Mode non-streaming (`stream=False`)
- Gestion d'erreurs avec affichage et retour None

#### 8️⃣ **Extraction des Champs d'Analyse - Multi-Variante avec Sélection Intelligente**

##### Parsing de la Réponse Ollama - 3 Variantes par Champ

Fonction : `parse_analysis(text, first_page_text=None)`

**Étape 1 : Extraction des 3 Variantes**
- Recherche labels : `Institution Variante 1:`, `Institution Variante 2:`, `Institution Variante 3:`
- Recherche labels : `Objet Variante 1:`, `Objet Variante 2:`, `Objet Variante 3:`
- Recherche label : `Date:`
- Suppression des commentaires entre parenthèses/crochets
- Auto-remplissage avec `"inconnu"` si moins de 3 variantes trouvées

**Étape 2 : Extraction du Titre du Document**

Fonction : `title_from_first_page(first_page_text)`
- Heuristique : Identifie le premier titre plausible de la première page
- Filtres appliqués :
  - Doit être 6-80 caractères
  - Doit contenir suffisamment de lettres (min 5 ou 1/3 du texte)
  - Ignore les en-têtes génériques ("page", "document", "table", "annexe", "index", "sommaire")
- Retourne : Le titre détecté ou `None` si aucun titre trouvé

**Étape 3 : Sélection Intelligente de la Meilleure Variante**

Fonction : `best_match_with_title(variants, title_text)`
- Algorithme de scoring (pour chaque variante) :
  - **Match exact** (variante == titre) : Score = 1000
  - **Sous-ensemble** (variante ⊆ titre) : Score = 500 + (mots_communs × 50)
  - **Chevauchement de mots** : Score = mots_communs × 50
- Ignore les variantes "inconnu" dans le scoring
- Sélectionne la variante avec le score le plus élevé
- Fallback : Si aucun titre ou tous "inconnu", retourne variante 1 par défaut

**Exemple de Sélection** :
```
Titre détecté: "Relevé de compte bancaire"
Variantes:     ["Relevé De Compte", "Relevé Bancaire", "Document Financier"]
Scores:        [600 (substring), 500, 0]
Résultat:      "Relevé De Compte" ✓
```

**Étape 4 : Simplification du Nom Institution**

Fonction : `simplify_institution_name(name)`
- Supprime articles au début : `la`, `le`, `les`, `l'`, `the`
- Supprime formes juridiques à la fin (avec ou sans points) : `S.A.`, `S.A.S.`, `SA`, `SAS`, `SARL`, `SCS`, `SNC`, `SCA`, `GMBH`, `INC`, `LTD`, `PLC`, `LLC`, `CORP`, `COMPANY`, `LIMITED`, `ANONYME`, `SOCIÉTÉ`
- Exemple : `"La Banque Nationale S.A.S."` → `"Banque Nationale"`

**Tuple de Retour** :
```python
(institution, objet, date, certitude)
```
- **institution** : Meilleure variante sélectionnée + simplifiée
- **objet** : Meilleure variante sélectionnée
- **date** : Valeur extraite (YYYY-MM ou "inconnu")
- **certitude** : `True` si date valide (YYYY-MM) ET max 1 champ "inconnu" parmi Institution/Objet; `False` sinon

##### Validation Finale
- **Date OBLIGATOIRE** : Format `YYYY-MM` strict (regex `\d{4}-\d{2}`)
- **Tolérance 1 champ manquant** : Max 1 "inconnu" sur Institution/Objet
- **Rejet si** : date invalide OU 2+ champs "inconnu"

#### 9️⃣ **Stratégie Deux-Passes (Fallback)**

Si certitude insuffisante après Passe 1 ET source alternative disponible (texte natif pour PDF) :
1. **Passe 2** : Réanalyse avec le texte fallback
2. Si Passe 2 réussit → utilisation des résultats Passe 2
3. Sinon → conservation des résultats Passe 1

#### 🔟 **Validation, Renommage Strict et Export**

##### Vérifications Préalables
- **Date OBLIGATOIRE** : Format `YYYY-MM` (année et mois requis)
  - Validation : Regex `\d{4}-\d{2}` stricte
  - Rejet si absent ou au format incorrect
- **Aucun champ "inconnu"** : Institution, Objet ET Date doivent être renseignés
- **Logique de validation** :
  - Échec si : date invalide OU au moins un champ = "inconnu"
  - Succès si : date valide + 3 champs présents

##### Génération du Nom de Fichier - Format Strict

Fonction : `sanitize(s)` - Nettoyage AGRESSIF
- Supprime caractères invalides : `\ / * ? : " < > | ( ) [ ] { }`
- Supprime caractères de contrôle : `\n \t \r`
- Retourne "inconnu" si vide ou égal à "inconnu" après nettoyage

Fonction : `generate_name(inst, obj, date, ext)` - Format strict en Title Case
- Applique `sanitize()` à Institution et Objet
- Format FINAL : `{YYYY-MM} {Institution} {Objet}.{ext}`
- Capitalisation Title Case (sauf date)
- Exemple : `2024-04 Banque De France Fiche De Paie.pdf`

##### Actions sur Succès
- **Copie** du fichier original → `SOURCE_DIR/Export_YYYYMMDD_HHMMSS/nouveau_nom`
  - Préservation des fichiers originaux dans le dossier source
- **Copie** du PDF OCRisé (si généré) → `SOURCE_DIR/Export_YYYYMMDD_HHMMSS/nouveau_nom.pdf`
  - PDF searchable (image + couche OCR texte + description vision)
- Enregistrement dans le CSV de log avec statut "Succès"
- Affichage : `✅ EXPORTÉ: {nouveau_nom}`

##### Actions sur Échec
- **Copie** du fichier avec nom généré depuis les champs obtenus → `SOURCE_DIR/Echec_YYYYMMDD_HHMMSS/`
- Suppression du PDF temporaire de la liste de suivi
- Enregistrement dans le CSV avec statut "Échec" et champs détectés
- Affichage : `✗ (ÉCHEC)` + raison avec champs détectés

---

## 📊 Logging & Traçabilité

### Fichier CSV Horodaté
- Créé dans `SOURCE_DIR/Export_YYYYMMDD_HHMMSS/log_YYYYMMDD_HHMMSS.csv`
- Colonnes détaillées :
  - Fichier (nom original)
  - Statut (Succès/Échec)
  - Nouveau nom (généré)
  - Institution (détectée)
  - Objet (détecté)
  - Date (extraite)

### Affichage Console

Chaque fichier génère un flux d'affichage structuré montrant la progression :

```
[FILE] document.pdf
  [PDF] Extraction texte natif... ✓ (natif)
  [LLAVA] Analyse vision (1ère page)... ✓ (850 chars)
  [DATES] Recherche (Tesseract + Llava)... 1 trouvée(s)
  [PDF] Régénération avec enrichissement vision... ✓
  [OLLAMA] Passe 1 (initial)... ✓
  [PARSE] RenAIme | Contrat | 2024-05 ✓ (OK)
  ✅ EXPORTÉ: 2024-05 Renaime Contrat.pdf
```

Ou en cas d'OCR nécessaire :

```
[FILE] document_scan.pdf
  [PDF] Extraction texte natif... ✗ (image)
  [OCR] Prétraitement... ✓ (PDF OCRisé créé)
  [LLAVA] Analyse vision (1ère page)... ✓ (720 chars)
  [DATES] Recherche (Tesseract + Llava)... 1 trouvée(s)
  [PDF] Régénération avec enrichissement vision... ✓
  [OLLAMA] Passe 1 (initial)... ✓
  [PARSE] Banque | Fiche De Paie | 2024-12 ✓ (OK)
  ✅ EXPORTÉ: 2024-12 Banque Fiche De Paie.pdf
```

Ou image seule :

```
[FILE] scan.jpg
  [OCR] Prétraitement image... ✓ (Tesseract FRA + PDF OCRisé créé)
  [LLAVA] Analyse vision (1ère page)... ✓ (650 chars)
  [DATES] Recherche (Tesseract + Llava)... 2 trouvée(s)
  [PDF] Régénération avec enrichissement vision... ✓
  [OLLAMA] Passe 1 (initial)... ✓
  [PARSE] Mairie | Certificat | 2025-01 ✓ (OK)
  ✅ EXPORTÉ: 2025-01 Mairie Certificat.jpg
```

Tags utilisés :
- `[FILE]` : Début de traitement d'un fichier
- `[PDF]` : Extraction texte natif de PDF (natif = texte présent, image = OCR requis)
- `[OCR]` : Prétraitement et OCR image/PDF (Tesseract FRA)
- `[DOCX]` / `[XLSX]` : Extraction formats Word/Excel
- `[LLAVA]` : Analyse vision de la première page
- `[DATES]` : Extraction et validation des dates candidates
- `[OLLAMA]` : Appel au modèle IA et analyse
- `[PARSE]` : Validation et parsing des résultats
- Symboles : `✓` (succès), `✗` (échec), `⚠` (certitude insuffisante)

**Ordre d'exécution garanti :**
1️⃣ Extraction texte (natif ou OCR)
2️⃣ Analyse vision Llava (1ère page)
3️⃣ Extraction dates candidates (Tesseract + Llava)
4️⃣ Régénération PDF avec enrichissement vision
5️⃣ Analyse Ollama (avec contexte vision)
6️⃣ Parsing et validation
7️⃣ Renommage et export

---

## 🛡️ Gestion des Fichiers Temporaires

### Création
- PDF OCRisés générés dans `/tmp/tmp*.pdf`
- Images temporaires pour llava dans `/tmp/tmp*.png`
- Suivi dans liste globale `_temp_files`

### Gestion du Cycle de Vie
- Si succès : PDF temp copié vers Export, puis supprimé de la liste
- Si échec : Fichier temp supprimé de la liste (nettoyé à la fin)
- Nettoyage automatique via `cleanup_temp_files()`

### Nettoyage

Fonction : `cleanup_temp_files()`
- Parcourt tous les fichiers temporaires enregistrés
- Supprime chaque fichier existant
- Affiche confirmation : `[NETTOYAGE] Fichier temporaire supprimé: {path}`

Moments de nettoyage :
- À la fin du script (via `atexit.register`)
- En cas d'interruption (Ctrl+C ou SIGTERM)
- En cas d'exception non gérée

---

## ⌚ Gestion des Interruptions & Signaux

### Gestionnaire de Signaux

Fonction : `signal_handler(signum, frame)`
- Capture `SIGINT` (Ctrl+C) et `SIGTERM`
- Affiche message d'interruption avec nom du signal
- Appelle `cleanup_temp_files()`
- Termine avec code 130 (interruption standard)

### Enregistrement
```python
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Arrêt système
atexit.register(cleanup_temp_files)            # Fermeture normale
```

### Try/Except dans main()
- `KeyboardInterrupt` : Nettoyage + exit(130)
- Autres exceptions : Nettoyage + re-raise
- Finally : Nettoyage final garanti

---

## 🔍 Cas de Traitement Particuliers

### PDF avec Texte Natif
- Extraction directe via pdfplumber
- Pas d'OCR si texte suffisant
- Analyse llava sur image de 1ère page (si disponible)
- Analyse rapide par Ollama

### PDF Scannés (Image)
- Détection automatique (pas de texte pdfplumber)
- Déclenchement OCR complet
- Génération PDF OCRisé temporaire searchable
- Analyse llava sur 1ère page
- Régénération avec enrichissement vision

### Images PNG/JPG
- OCR direct de l'image
- Autorotation si nécessaire
- Génération PDF searchable
- Analyse llava sur l'image
- Régénération avec enrichissement vision

### Documents Word/Excel
- Extraction texte natif (pas d'OCR)
- Pas d'analyse llava (pas d'image)
- Analyse Ollama directe

### Texte Insuffisant
- Si extraction retourne texte vide
- Copie vers Echec/ avec log
- Affichage erreur : `[ERREUR] Aucun texte détecté`

### Analyse Incomplète
- Champ "inconnu" après analyse
- Tentative Passe 2 avec source alternative (si disponible)
- Si toujours incomplète : Copie vers Echec/

---

## 💾 Stockage & Organisation

### Arborescence Finale
```
SOURCE_DIR/
├── fichier1.pdf           (original non modifié)
├── fichier2.jpg           (original non modifié)
├── Export_20241230_143022/
│   ├── 2024-05 Le Monde Contrat.pdf
│   ├── 2024-03 Bank X Relevé.pdf     (PDF searchable enrichi vision)
│   └── log_20241230_143022.csv
└── Echec_20241230_143022/
    ├── inconnu Unparsable Document.pdf
    └── inconnu Unrecognized File.png
```

### Fichiers Générés Côté Export
- Fichier original copié et renommé selon pattern
- PDF OCRisé searchable (si image ou PDF scannés)
- CSV de log horodaté

---

## 📚 Référence des Fonctions

### Configuration & Initialisation

#### `load_config()`
- **Objectif** : Charger configuration depuis `config.json` ou retourner défauts
- **Retour** : Dict fusionné DEFAULT_CONFIG + config.json
- **Fallback** : Utilise DEFAULT_CONFIG si fichier absent

#### `save_config(config)`
- **Objectif** : Persister configuration en JSON
- **Paramètre** : Dict de configuration
- **Effet** : Écrit `config.json` avec indentation et UTF-8

#### `check_deps()`
- **Objectif** : Vérifier présence packages critiques
- **Packages vérifiés** : pdfplumber, PIL, pytesseract, ollama
- **Effet** : Arrête script (exit 1) si dépendance manquante

#### `ensure_models()`
- **Objectif** : Vérifier et télécharger modèles Ollama manquants
- **Modèles requis** : llava:latest, llama3:8b-instruct-q4_0
- **Processus** : Liste modèles → vérifie présence → télécharge si absent
- **Effet** : Arrête si Ollama non installé

### Nettoyage & Signaux

#### `cleanup_temp_files()`
- **Objectif** : Supprimer tous les fichiers temporaires créés
- **Source** : Liste globale `_temp_files`
- **Effet** : Supprime chaque fichier, affiche confirmation

#### `signal_handler(signum, frame)`
- **Objectif** : Gérer interruptions SIGINT/SIGTERM
- **Effet** : Affiche message, nettoie temps, exit(130)

### Traitement d'Images & OCR

#### `preprocess_image_for_ocr(img)`
- **Objectif** : Améliorer image pour meilleure reconnaissance OCR
- **Entrée** : PIL Image object
- **Processus** : Grayscale → Median → CLAHE/Otsu (si cv2) → Contrast → Sharpen
- **Retour** : PIL Image prétraitée

#### `create_searchable_pdf_page(img, vision_description=None)`
- **Objectif** : Créer page PDF hybrid (image + OCR texte searchable)
- **Entrée** : PIL Image, description vision optionnelle
- **Processus** : Image→PDF + Tesseract→HOCR + Fusion pypdf
- **Retour** : Bytes du PDF searchable
- **Impact** : PDFs générés sont searchable (Ctrl+F fonctionne)

### Extraction de Texte

#### `extract_from_pdf(path)`
- **Objectif** : Extraire texte natif d'un PDF
- **Entrée** : Path vers fichier PDF
- **Retour** : String (texte brut) ou None si échec/vide

#### `ocr_pdf(path, vision_description=None)`
- **Objectif** : OCR complet d'un PDF scannés
- **Entrée** : Path PDF, description vision optionnelle
- **Retour** : (texte_complet, chemin_pdf_temp) ou (None, None)
- **Fichiers créés** : PDF temporaire searchable

#### `extract_from_image(path, vision_description=None)`
- **Objectif** : OCR d'une image avec PDF searchable
- **Entrée** : Path image, description vision optionnelle
- **Retour** : (texte, chemin_pdf_temp) ou (None, None)

#### `extract_from_docx(path)`
- **Objectif** : Extraire texte d'un document Word
- **Retour** : String (texte brut) ou None

#### `extract_from_xlsx(path)`
- **Objectif** : Extraire texte d'une feuille Excel
- **Retour** : String (texte brut) ou None

### Extraction de Dates

#### `extract_dates(text)`
- **Objectif** : Extraire max 1 date YYYY-MM valide (< 20 ans)
- **Formats** : YYYY-MM, DD/MM/YYYY, YYYY/MM/DD, mois nommés, YYYY seul
- **Validation** : Plage [année-20, année+1], mois 01-12, jour 01-31
- **Retour** : List max 1 date en format YYYY-MM ou YYYY

### Analyse Vision

#### `analyze_vision(image_path, model=None)`
- **Objectif** : Analyser visuellement la 1ère page d'un document
- **Entrée** : Path image, modèle vision (auto-sélectionné si None)
- **Processus** : Encode base64 → prompt vision → ollama.generate
- **Retour** : String description (max 4000 chars) ou None

#### `get_system_power_level()`
- **Objectif** : Détecter puissance système (RAM + VRAM GPU)
- **Retour** : Tuple (level, ram_gb, vram_gb) où level = 'low', 'medium', 'high'

#### `select_vision_model(available_models)`
- **Objectif** : Sélectionner modèle vision optimal selon puissance
- **Retour** : Tuple (model_name, power_level, ram_gb, vram_gb)

### Optimisation Ollama

#### `extract_first_page(text)`
- **Objectif** : Limiter texte à 1ère page (~3500 chars)
- **Retour** : String tronqué

#### `extract_essential_sections(text)`
- **Objectif** : Extraire sections critiques (30 premières lignes + dates)
- **Retour** : String compact

### Analyse Ollama

#### `analyze_ollama(text, dates, model, vision_analysis=None, pass_level="initial")`
- **Objectif** : Analyser texte avec Ollama LLM
- **Paramètres** : texte, dates candidates, modèle, analyse vision optionnelle
- **Retour** : String (réponse brute) ou None

#### `simplify_institution_name(name)`
- **Objectif** : Supprimer articles et formes juridiques
- **Retour** : Nom simplifié

#### `title_from_first_page(first_page_text)`
- **Objectif** : Extraire titre plausible de la 1ère page
- **Retour** : String titre ou None

#### `best_match_with_title(variants, title_text)`
- **Objectif** : Sélectionner meilleure variante par scoring
- **Retour** : String variante optimale

#### `parse_analysis(text, first_page_text=None)`
- **Objectif** : Parser réponse Ollama multi-variante
- **Retour** : (institution, objet, date, certitude)

### Renommage

#### `sanitize(s)`
- **Objectif** : Nettoyer nom de fichier
- **Retour** : String propre ou "inconnu"

#### `generate_name(inst, obj, date, ext)`
- **Objectif** : Générer nom de fichier strict
- **Format** : `{YYYY-MM} {Institution} {Objet}.{ext}` (Title Case)
- **Retour** : String nom complet

### Programme Principal

#### `main()`
- **Objectif** : Orchestrer tout le pipeline de traitement
- **Processus** :
  1. Vérifier dépendances (`check_deps`)
  2. Vérifier modèles Ollama (`ensure_models`)
  3. Charger configuration
  4. Demander dossier source (interactif)
  5. Sélectionner modèle Ollama (interactif)
  6. Créer Export/ et Echec/ dossiers horodatés
  7. Initialiser log CSV
  8. Itérer sur fichiers du dossier source :
     - Extraire texte (natif ou OCR)
     - Analyser avec llava (si image disponible)
     - Extraire dates (texte combiné)
     - Régénérer PDF avec enrichissement vision
     - Analyser avec Ollama (+ contexte vision)
     - Parser résultats
     - Valider champs
     - Renommer et copier vers Export/ ou Echec/
     - Enregistrer log
  9. Nettoyer fichiers temporaires

---

## 🎛️ Paramètres & Seuils

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| Résolution OCR | 300 DPI | Qualité optimale pour documents texte |
| Contraste Image | 1.5x | Renforcement pour texte clair |
| Limite 1ère page | 3500 chars | Fenêtre contexte 4K |
| Limite vision | 4000 chars | Description vision max |
| Format Date | YYYY-MM | Année + mois minimum requis |
| Plage Date | Actuelle -20ans | Filtrage dates obsolètes |
| Kernel Morpho | 2x2 | Taille pour close operation |
| CLAHE clipLimit | 2.0 | Limite contraste adaptatif |
| Filtre médian | 3 | Taille pour réduction bruit |

---

## 📝 Notes de Développement

### Points Forts de l'Architecture
- **Vision + OCR** : Analyse hybride visuelle et textuelle
- **Multi-variante** : 3 propositions avec scoring intelligent
- **Enrichissement PDF** : Description vision intégrée au searchable
- **Robustesse** : Gestion exhaustive des cas d'erreur
- **Transparence** : Logging détaillé avec tags catégorisés
- **Flexibilité** : Configuration JSON + CLI interactif
- **Nettoyage** : Suppression automatique des fichiers temporaires via signaux
- **Fallback** : Analyse deux-passes si certitude insuffisante

### Dépendances Système
- **Tesseract OCR** (non inclus dans pip)
  - Linux : `apt-get install tesseract-ocr tesseract-ocr-fra`
  - macOS : `brew install tesseract`
  - Windows : Télécharger depuis https://github.com/UB-Mannheim/tesseract/wiki
- **Ollama** : https://ollama.ai

---

## 🧪 Tests & Validation

### Cas de Test Recommandés
1. PDF natif avec texte lisible → Succès direct
2. PDF scannés dégradés → OCR + llava + analyse
3. Image mal orientée → Autorotation + OCR
4. Document sans date → Échec attendu
5. Champ "inconnu" → Tentative fallback
6. Interruption Ctrl+C → Nettoyage fichiers temp
7. Dossier source inexistant → Demande interactive
8. Modèle Ollama manquant → Téléchargement automatique

---

## ✨ Résumé Fonctionnel

**Script complet et robuste** pour :
- ✅ OCR intelligent (PDF + Images) avec prétraitement avancé
- ✅ Analyse vision (Llava) pour contexte enrichi
- ✅ Analyse IA multi-variante (Ollama) avec sélection intelligente
- ✅ PDFs searchable enrichis avec description vision
- ✅ Renommage automatique cohérent (Title Case)
- ✅ Traçabilité complète (CSV horodaté)
- ✅ Gestion d'erreurs gracieuse avec signaux
- ✅ Nettoyage automatique des fichiers temporaires
- ✅ Configuration flexible (JSON + interactif)
- ✅ Téléchargement automatique des modèles Ollama
