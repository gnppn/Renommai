# Fonctionnalités - RenAIme OCR & Renommage de Documents

## 🎯 Objectif Global
Script de tri et renommage automatique de documents (PDF, PNG, JPG) basé sur :
- **OCR** : Extraction de texte via Tesseract
- **Analyse IA** : Détection Institution/Objet/Date via Ollama
- **Renommage** : Génération automatique de noms fichiers selon le format `YYYY-MM Institution Objet.ext` (en Title Case)

---

## 📋 Configuration & Initialisation

### Configuration JSON (`config.json`)
- **SOURCE_DIR** : Dossier contenant les fichiers à traiter (demandé à l'utilisateur au lancement)
- **OLLAMA_MODEL** : Modèle LLM à utiliser (mistral, llama3, etc.)

Note: Les dossiers `Export_YYYYMMDD_HHMMSS/` et `Echec_YYYYMMDD_HHMMSS/` sont créés automatiquement **dans le dossier source** avec timestamp pour traçabilité.

### Chargement de la Configuration
- Lecture du fichier `config.json` s'il existe
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

### Packages Requis
- `pdfplumber` : Extraction de texte/images de PDF
- `PIL/Pillow` : Manipulation d'images (redimensionnement, filtres)
- `pytesseract` : Interface Python pour Tesseract OCR
- `ollama` : API Python pour intégration LLM locale
- `PyPDF2` : Fusion de pages PDF OCRisées (optionnel mais recommandé)
- `python-docx` : Extraction texte de documents Word (.docx)
- `openpyxl` : Extraction texte de feuilles Excel (.xlsx)

### Gestion des Erreurs
- Arrêt du script si une dépendance critique est manquante
- Message clair avec commande `pip install -r requirements.txt`

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
- Si texte détecté : Utilisation directe de la première page
- Si pas de texte (PDF image) : Déclenchement de l'OCR complet

##### Pour les Images :
- Charge l'image via PIL
- **Autorotation** : Détection automatique de l'orientation via pytesseract OSD
  - Rotation à 360° - angle détecté si nécessaire
- **Prétraitement d'image** pour meilleure reconnaissance OCR :
  - Conversion en niveaux de gris (L)
  - Autocontraste normalisé
  - Renforcement du contraste (facteur 2.0)
  - Détection et inversion optionnelle (si fond sombre)
  - Filtre de netteté

##### Pour les DOCX :
- Extraction directe du texte via `python-docx`
- Récupération de tout le contenu (paragraphes, tableaux)
- Pas d'OCR nécessaire (texte natif)

##### Pour les XLSX :
- Extraction du texte via `openpyxl`
- Parcours de toutes les feuilles et cellules
- Concaténation du contenu pour analyse complète

#### 2️⃣ **Génération de PDF OCRisé Searchable (Temporaire)**

**⚠️ Effectué AVANT Ollama**

##### Architecture PDF Hybrid (Searchable)

Tous les PDFs OCRisés générés combinent deux couches :

```
PDF Final
├── Layer 1 (Visuelle): Image originale préservée
└── Layer 2 (Texte): Couche OCR caché mais searchable (HOCR)
   → Permet recherche Ctrl+F, copie/sélection texte
   → Invisible à l'écran (texte positionné sous l'image)
```

Fonction dédiée : `create_searchable_pdf_page(img)`
- 1. Génère PDF image simple (PIL)
- 2. Génère PDF OCR avec tesseract (HOCR = texte caché)
- 3. Fusionne les deux pages (PyPDF2 merge_page)
- 4. Retourne bytes du PDF searchable

##### Pour PDF sans texte :
- Conversion chaque page en image haute résolution (300 DPI)
- Autorotation automatique (Tesseract OSD)
- **Prétraitement OCR avancé** (meilleures pratiques pour contraste difficile) :
  - Filtre médian pour nettoyage du bruit
  - CLAHE (Contrast Limited Adaptive Histogram Equalization) pour contraste adaptatif
  - Renforcement du contraste global
  - Threshold adaptatif Otsu pour binarisation optimale
  - Morphologie (érosion + dilatation) pour clarifier les caractères
  - Filtre de netteté final
- **Génération PDF searchable** : Image + couche OCR texte fusionnées (via `create_searchable_pdf_page()`)
- Stockage en fichier temporaire (`/tmp/tmp*.pdf`)
- Retour : texte complet + chemin temp PDF

##### Pour Images PNG/JPG :
- Autorotation détectée via Tesseract OSD
- **Prétraitement OCR avancé** (mêmes étapes que PDF) :
  - Filtre médian, CLAHE, threshold Otsu, morphologie
- Extraction OCR via Tesseract FR
- **Génération PDF searchable** (image + couche OCR texte, stocké temporairement via `create_searchable_pdf_page()`)
- Texte retourné pour Ollama

#### 3️⃣ **Extraction de Dates Candidates**

**⚠️ Effectuée APRÈS OCR mais AVANT Ollama**

##### Source d'extraction
- **Si PDF OCRisé généré** : Extraction depuis le PDF OCRisé (plus fiable)
- **Sinon** : Extraction depuis le texte natif (PDF/DOCX/XLSX)
- Format ISO : `YYYY-MM-DD` et `YYYY-MM`
- Formats alphanumériques : `DD/MM/YYYY`, `DD MMM YYYY`
- Années isolées : `YYYY`

##### Normalisation des Dates
- Normalisation en format `YYYY-MM` (année-mois)
- Dédupliplication des doublons

#### 4️⃣ **Analyse IA avec Ollama - Stratégie Optimisée Deux-Passes**

**⚠️ Effectuée APRÈS OCR et extraction dates**

##### Optimisation: Première Page COMPLÈTE

Le texte envoyé à Ollama est TOUTE la première page du document pour maximiser le contexte disponible pour l'analyse :

- **`extract_first_page(text)`** : Extraire 1ère page complète (~1200 chars heuristique)
  - Parcourt les lignes du texte
  - Accumule jusqu'à 1200 caractères
  - Coupe à limite pour respecter "1ère page"

- **`extract_essential_sections(text)`** : Sections critiques pour Passe 1
  - Extrait 30 premières lignes (en-têtes typiquement)
  - Ajoute dates candidates trouvées dans text[:1500]
  - Total ~400-500 chars : juste ce qui faut pour identifier Institution/Objet/Date
  - Très compact pour réponse ultra-rapide Ollama

##### Flux Unique: Analyse Complète sur 1ère Page
- Texte envoyé : 1ère page COMPLÈTE du document (~1200 chars)
- Contexte maximal pour meilleure détection Institution/Objet/Date
- Analyse directe sans extrait essentialisé
- Impact : Meilleure précision sur détection de l'objet du document

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

Texte :
{text}
```

##### Appel Ollama
- Modèle configurable (par défaut : mistral)
- Texte : 1ère page COMPLÈTE du document (~1200 chars)
- Streaming de réponse (affichage point par point `...`)
- Timeout de 120 secondes max par analyse
- Gestion d'erreurs avec fallback gracieux

#### 5️⃣ **Extraction des Champs d'Analyse - Multi-Variante avec Sélection Intelligente**

##### Parsing de la Réponse Ollama - 3 Variantes par Champ

Fonction : `parse_analysis(text, first_page_text=None)`

**Étape 1 : Extraction des 3 Variantes**
- Recherche labels : `Institution Variante 1:`, `Institution Variante 2:`, `Institution Variante 3:`
- Recherche labels : `Objet Variante 1:`, `Objet Variante 2:`, `Objet Variante 3:`
- Recherche label : `Date:`
- Auto-remplissage avec `"inconnu"` si moins de 3 variantes trouvées

**Étape 2 : Extraction du Titre du Document**
- Fonction : `title_from_first_page(first_page_text)`
- Heuristique : Identifie le premier titre plausible de la première page
- Filtres appliqués :
  - Doit être 6-80 caractères
  - Doit contenir suffisamment de lettres
  - Ignore les en-têtes génériques ("page", "document", "table", etc.)
- Retourne : Le titre détecté ou `None` si aucun titre trouvé

**Étape 3 : Sélection Intelligente de la Meilleure Variante**
- Fonction : `best_match_with_title(variants, title_text)`
- Algorithme de scoring (pour chaque variante) :
  - **Match exact** (variante == titre) : Score = 1000
  - **Sous-ensemble** (variante ⊆ titre) : Score = 500 + (mots_communs × 50)
  - **Chevauchement de mots** : Score = mots_communs × 50
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
- Fonction : `simplify_institution_name(name)`
- Supprime articles au début : `la`, `le`, `les`, `l'`, `the`
- Supprime formes juridiques à la fin : `S.A.`, `S.A.S.`, `SA`, `SAS`, `SARL`, `SACS`, `SCS`, `GMBH`, `INC`, `LTD`, `PLC`, `LLC`, `CORP`, `COMPANY`, `LIMITED`, `anonyme`, `société`
- Supporte formes avec ou sans points : `S.A.S.` et `SAS` indifféremment
- Exemple : `"La Banque Nationale S.A.S."` → `"Banque Nationale"`

- **Tuple de Retour** :
  ```python
  (institution, objet, date, certitude)
  ```
  - **institution** : Meilleure variante sélectionnée + simplifiée
  - **objet** : Meilleure variante sélectionnée
  - **date** : Valeur extraite (YYYY-MM ou "inconnu")
  - **certitude** : `True` si date valide (YYYY-MM) ET max 1 champ "inconnu" parmi Institution/Objet; `False` sinon

##### Validation Finale
- **Date OBLIGATOIRE** : Format `YYYY-MM` stricte (regex `\d{4}-\d{2}`)
- **Tolérance 1 champ manquant** : Max 1 "inconnu" sur Institution/Objet
- **Rejet si** : date invalide OU 2+ champs "inconnu"

#### 6️⃣ **Validation, Renommage Strict et Export**

##### Vérifications Préalables
- **Date OBLIGATOIRE** : Format `YYYY-MM` (année et mois requis)
  - Validation : Regex `\d{4}-\d{2}` stricte
  - Rejet si absent ou au format incorrect
- **Aucun champ manquant** : Institution, Objet ET Date doivent être renseignés (aucun "inconnu")
- **Logique de validation** :
  - Échec si : date invalide OU au moins un champ = "inconnu"
  - Succès si : date valide + 3 champs présents

##### Génération du Nom de Fichier - Format Strict

Fonction : `sanitize(s)` - Nettoyage AGRESSIF
- Supprime caractères invalides : `\ / * ? : " < > | ( ) [ ] { }`
- Supprime caractères de contrôle : `\n \t \r`
- Pas de limite de longueur : Conservation de la longueur complète
- Retourne "inconnu" si vide après nettoyage

Fonction : `generate_name(inst, obj, date, ext)` - Format strict en Title Case
- Applique `sanitize()` à Institution et Objet
- Format FINAL : `{YYYY-MM} {Institution} {Objet}.{ext}` (en Title Case : une majuscule par mot)
- Exemple : `2024-04 Banque De France Fiche De Paie.pdf`
- Pas de commentaires ou métadata dans le nom

##### Actions sur Succès
- Copie du fichier original (NOT move) → `SOURCE_DIR/Export_YYYYMMDD_HHMMSS/nouveau_nom`
  - Préservation des fichiers originaux dans le dossier source
- Copie du PDF OCRisé (si généré) → `SOURCE_DIR/Export_YYYYMMDD_HHMMSS/nouveau_nom.pdf`
  - PDF searchable (image + couche OCR texte fusionnées)
- Enregistrement dans le CSV de log avec statut "Succès"
- Affichage : `✅ EXPORTÉ: {nouveau_nom}`

##### Actions sur Échec
- Copie du fichier avec nom généré depuis les champs obtenus (même si "inconnu") → `SOURCE_DIR/Echec_YYYYMMDD_HHMMSS/`
- Suppression du PDF OCRisé temporaire (ne pas conserver les fichiers temp)
- Enregistrement dans le CSV avec statut "Échec" et nom généré
- Affichage : `✗ (ÉCHEC)` + raison avec champs détectés

---

## 📊 Logging & Traçabilité

### Fichier CSV Horodaté
- Créé dans `SOURCE_DIR/Export_YYYYMMDD_HHMMSS/log_YYYYMMDD_HHMMSS.csv`
- Colonnes détaillées :
  - Fichier original
  - Statut (Succès/Échec)
  - Nouveau nom généré
  - Institution détectée
  - Objet détecté
  - Date extraite
  - Message d'erreur (le cas échéant)

### Affichage Console

Chaque fichier génère un flux d'affichage structuré montrant la progression :

```
[FILE] document.pdf
  [PDF] Extraction texte natif... ✓
  [DATES] Extraction... 2 trouvées
  [OLLAMA] Analyse... ✓
  [PARSE] RenAIme | Contrat | 2024-05 ✓
  [SUCCESS] 2024-05 RenAIme Contrat.pdf
```

Ou en cas d'OCR nécessaire :

```
[FILE] document_scan.pdf
  [PDF] Extraction texte natif... ✗ (image)
  [OCR] Prétraitement image... ✓ (PDF OCRisé créé)
  [DATES] Recherche... 1 trouvée(e)
  [OLLAMA] Analyse... ✓
  [PARSE] Banque | Fiche de paie | 2024-12 ✓
  [SUCCESS] 2024-12 Banque Fiche de paie.pdf
```

Ou image seule :

```
[FILE] scan.jpg
  [OCR] Prétraitement image... ✓ (Tesseract FRA)
  [DATES] Recherche... 2 trouvée(e)s
  [OLLAMA] Analyse... ✓
  [PARSE] Mairie | Certificat | 2025-01 ✓
  [SUCCESS] 2025-01 Mairie Certificat.jpg
```

Tags utilisés :
- `[FILE]` : Début de traitement d'un fichier
- `[PDF]` : Extraction texte natif de PDF (natif = texte présent, image = OCR requis)
- `[OCR]` : Prétraitement et OCR image/PDF (Tesseract FRA)
- `[DOCX]` / `[XLSX]` : Extraction formats Word/Excel
- `[DATES]` : Extraction et validation des dates candidates (effectuée après OCR)
- `[OLLAMA]` : Appel au modèle IA et analyse (effectuée après OCR et dates)
- `[PARSE]` : Validation et parsing des résultats
- `[SUCCESS]` : Fichier renommé et déplacé avec succès
- `[ERREUR]` : Erreurs de traitement
- Symboles : `✓` (succès), `✗` (échec), `⚠` (avertissement)

**Ordre d'exécution garanti :**
1️⃣ Extraction texte (natif ou OCR)
2️⃣ Extraction dates candidates
3️⃣ Analyse Ollama

---

## 🛡️ Gestion des Fichiers Temporaires

### Création
- PDF OCRisés générés dans `/tmp/tmp*.pdf`
- Suivi dans une liste `temp_files`

### Gestion du Cycle de Vie
- Si succès : Fichier temp déplacé vers `EXPORT_DIR`
- Si échec : Fichier temp supprimé automatiquement
- Si interruption (Ctrl+C) : Tous les temps supprimés + cleanup message

### Nettoyage
- Suppression automatique en cas de KeyboardInterrupt
- Suppression en cas d'exception non gérée
- Affichage de chaque suppression pour transparence

---

## ⌚ Gestion des Interruptions

### KeyboardInterrupt (Ctrl+C)
- Capture du signal
- Affichage du message d'interruption
- Nettoyage des fichiers temporaires
- Fin gracieuse du script

### Exceptions
- Try/except global autour de main()
- Message d'erreur descriptif
- Nettoyage des fichiers temporaires même en cas d'erreur
- Arrêt contrôlé sans corruption de données

---

## 🔍 Cas de Traitement Particuliers

### PDF avec Texte Natif
- Extraction directe de la première page
- Pas d'OCR
- Analyse rapide par Ollama

### PDF Scannés (Image)
- Détection automatique (pas de texte pdfplumber)
- Déclenchement OCR complet
- Génération PDF OCRisé temporaire
- Analyse du texte OCRisé en priorité

### Images PNG/JPG
- OCR direct de l'image
- Autorotation si nécessaire
- Génération optionnelle PDF searchable
- Même flux d'analyse Ollama

### Texte Insuffisant
- Si extraction retourne texte vide
- Déplacement vers FAILURE_DIR
- Log de l'erreur avec motif

### Analyse Ollama Incomplète
- Champ "inconnu" après analyse
- Tentative sur second source (fallback)
- Si toujours incomplète : Déplacement vers FAILURE_DIR

---

## 💾 Stockage & Organisation

### Arborescence Finale
```
SOURCE_DIR/
├── fichier1.pdf
├── fichier2.jpg
└── Export/
    ├── 2024-05 Le Monde Contrat.pdf
    ├── 2024-05 Le Monde Contrat (searchable).pdf
    ├── 2024-03 Bank X Relevé.pdf
    └── log_traitement_20241230_143022.csv
└── Echec/
    ├── unparsable_document.pdf
    └── unrecognized_file.png
```

### Fichiers Générés Coté Export
- Fichier original renommé selon pattern
- PDF OCRisé (si image ou PDF scannés)
- CSV de log horodaté

---

## � Référence des Fonctions

### Configuration & Initialisation

#### `load_config()`
- **Objectif** : Charger configuration depuis `config.json` ou retourner défauts
- **Retour** : Dict avec clés SOURCE_DIR, EXPORT_DIR, FAILURE_DIR, OLLAMA_MODEL
- **Fallback** : Utilise DEFAULT_CONFIG si fichier absent

#### `save_config(config)`
- **Objectif** : Persister configuration en JSON
- **Paramètre** : Dict de configuration
- **Effet** : Écrit `config.json` avec clés de configuration

#### `check_deps()`
- **Objectif** : Vérifier présence packages critiques
- **Packages vérifiés** : pdfplumber, PIL, pytesseract, ollama
- **Effet** : Arrête script avec message si dépendance manquante

### Traitement d'Images & OCR

#### `preprocess_image_for_ocr(img)`
- **Objectif** : Améliorer image pour meilleure reconnaissance OCR
- **Entrée** : PIL Image object
- **Processus** :
  1. Conversion niveaux de gris
  2. Filtre médian (noise reduction)
  3. CLAHE si cv2 disponible (contraste adaptatif)
  4. Renforcement contraste global (1.5x)
  5. Threshold Otsu adaptatif
  6. Morphologie (érosion + dilatation)
  7. Filtre netteté final
- **Retour** : PIL Image preprocessée

#### `create_searchable_pdf_page(img)`
- **Objectif** : Créer page PDF hybrid (image visible + OCR texte searchable)
- **Entrée** : PIL Image object
- **Processus** :
  1. Convertir image → PDF simple (fond visible)
  2. Générer PDF OCR avec tesseract HOCR (texte caché)
  3. Fusionner les deux layers avec PyPDF2
- **Retour** : Bytes du PDF searchable
- **Fallback** : Retourne PDF OCR simple si fusion échoue
- **Impact** : PDFs générés sont searchable (Ctrl+F fonctionne)

### Extraction de Texte

#### `extract_from_pdf(path)`
- **Objectif** : Extraire texte natif d'un PDF
- **Entrée** : Path vers fichier PDF
- **Processus** : 
  - Ouvre PDF avec pdfplumber
  - Extrait texte de chaque page
  - Concatène avec newline
- **Retour** : String (texte brut) ou None si échec
- **Note** : N'effectue pas d'OCR (texte natif seulement)

#### `ocr_pdf(path)`
- **Objectif** : OCR complet d'un PDF (notamment scannés sans texte)
- **Entrée** : Path vers fichier PDF
- **Processus** :
  1. Essai 1 : Extraire images via pdfplumber (résolution 300 DPI)
  2. Essai 2 : Fallback pdf2image si peu/pas d'images
  3. Pour chaque image :
     - Autorotation (Tesseract OSD)
     - Prétraitement OCR avancé
     - Extraction texte (Tesseract FR)
     - Génération PDF searchable via `create_searchable_pdf_page()`
  4. Fusionner toutes les pages en PDF temporaire
- **Retour** : (texte_complet, chemin_pdf_temp) ou (None, None)
- **Fichiers créés** : PDF temporaire `/tmp/tmp*.pdf` (searchable)

#### `extract_from_image(path)`
- **Objectif** : OCR d'une image (PNG/JPG) avec PDF searchable
- **Entrée** : Path vers fichier image
- **Processus** :
  1. Charger image avec PIL
  2. Autorotation si nécessaire (OSD)
  3. Prétraitement OCR
  4. Extraction texte (Tesseract FR)
  5. Génération PDF searchable via `create_searchable_pdf_page()`
- **Retour** : (texte, chemin_pdf_temp) ou (None, None)
- **Fichiers créés** : PDF temporaire searchable

#### `extract_from_docx(path)`
- **Objectif** : Extraire texte d'un document Word
- **Entrée** : Path vers fichier .docx
- **Processus** : Lecture via python-docx, extraction paragraphes
- **Retour** : String (texte brut) ou None
- **Pas d'OCR** : Texte natif uniquement

#### `extract_from_xlsx(path)`
- **Objectif** : Extraire texte d'une feuille Excel
- **Entrée** : Path vers fichier .xlsx
- **Processus** : Lecture via openpyxl, extraction cellules toutes feuilles
- **Retour** : String (texte brut) ou None
- **Pas d'OCR** : Texte natif uniquement

### Extraction de Dates

#### `extract_dates(text)`
- **Objectif** : Extraire date YYYY-MM, valider YYYY présent, fiable et < 20 ans
- **Entrée** : String (texte à analyser)
- **Formats reconnus** (hiérarchie) :
  1. **YYYY-MM** (ISO standard) - priorité absolue
  2. **YYYY seul** (fallback) - année valide dans plage < 20 ans
  3. DD/MM/YYYY (European) → normalise en YYYY-MM
  4. YYYY/MM/DD → normalise en YYYY-MM
  5. "DD mois_nommé YYYY" (français/anglais) → normalise en YYYY-MM
- **Validation stricte** :
  - Année: [Année actuelle - 20] ≤ YYYY ≤ [Année actuelle + 1]
  - Mois: 01-12 (si présent)
  - Jour: 01-31 (si présent)
  - *Exemple 2025*: Accepte 2005-2026
- **Processus** : Regex patterns + validation stricte + extraction progressive + temporelle
- **Retour** : List max 1 date (première trouvée) en format `YYYY-MM` ou `YYYY`
- **Notes** : Priorité au format complet YYYY-MM; rejet automatique dates > 20 ans

### Optimisation Ollama

#### `extract_first_page(text)`
- **Objectif** : Limiter texte à 1ère page (~1200 chars heuristique)
- **Entrée** : String (texte complet)
- **Processus** :
  - Split par newline
  - Accumule lignes jusqu'à 1200 chars
  - Coupe à limite
- **Retour** : String (~1200 chars max)
- **Usage** : Avant envoi à Ollama pour réduire latence

#### `extract_essential_sections(text)`
- **Objectif** : Extraire sections critiques pour Passe 1 Ollama
- **Entrée** : String (texte de 1ère page)
- **Processus** :
  1. Extrait 30 premières lignes (en-têtes typiques)
  2. Ajoute dates candidates trouvées dans text[:1500]
  3. Formate : "en-têtes\n\nDates: D1, D2, D3"
- **Retour** : String (~400-500 chars)
- **Usage** : Passe 1 ultra-compact pour gain latence -55%

### Analyse Ollama

#### `analyze_ollama(text, dates, model, pass_level="initial")`
- **Objectif** : Analyser texte avec Ollama LLM
- **Paramètres** :
  - `text` : String à analyser
  - `dates` : List de dates candidates
  - `model` : Nom du modèle Ollama à utiliser
- **Processus** :
  - Extrait 1ère page COMPLÈTE du document (~1200 chars)
  - Formatte prompt avec 1ère page complète et dates candidates
  - Appelle `ollama.generate()` en mode non-streaming
- **Retour** : String (réponse brute de Ollama) ou None si erreur
- **Gestion erreurs** : Affiche erreur, retourne None

#### `parse_analysis(text)`
- **Objectif** : Parser réponse Ollama strict (avant commentaires)
- **Entrée** : String (réponse Ollama)
- **Processus** :
  1. Recherche labels `Institution:`, `Objet:`, `Date:`
  2. Extrait valeur AVANT premier `(` ou `[` (commentaire)
  3. Utilise les valeurs directement (déjà adaptées par Ollama au format nom)
  4. Valide format date (YYYY-MM strict)
  5. Calcule certitude : date valide ET max 1 champ "inconnu"
- **Retour** : (institution, objet, date, certitude)
- **Exemple** :
  ```
  Input:  "Institution: Banque De France\nObjet: Releve De Compte\nDate: 2024-04"
  Output: ("Banque De France", "Releve De Compte", "2024-04", True)
  ```

### Renommage

#### `sanitize(s)`
- **Objectif** : Nettoyer AGRESSIVEMENT un nom de fichier
- **Entrée** : String (nom brut)
- **Processus** :
  1. Supprime caractères invalides FS : `\ / * ? : " < > | ( ) [ ] { }`
  2. Supprime contrôle chars : `\n \t \r`
  3. Pas de limitation de longueur
  4. Strip whitespace
- **Retour** : String propre ou "inconnu" si vide
- **Impact** : Noms portables sur tous les OS, longueur illimitée

#### `generate_name(inst, obj, date, ext)`
- **Objectif** : Générer nom de fichier strict
- **Entrée** :
  - `inst` : Institution (string)
  - `obj` : Objet/Type (string)
  - `date` : Date (format YYYY-MM)
  - `ext` : Extension (avec point, ex. ".pdf")
- **Processus** :
  1. Applique `sanitize()` à Institution et Objet
  2. Format : `{date} {inst_clean} {obj_clean}{ext}`
- **Retour** : String (ex. "2024-04 Banque de France Fiche de paie.pdf")
- **Garanties** : Nom valide, sans commentaires, FS-compatible

### Programme Principal

#### `main()`
- **Objectif** : Orchestrer tout le pipeline de traitement
- **Processus** :
  1. Vérifier dépendances
  2. Charger configuration
  3. Demander dossier source (interactif)
  4. Sélectionner modèle Ollama (interactif)
  5. Créer Export/ et Echec/ dossiers horodatés
  6. Initialiser log CSV
  7. Itérer sur fichiers du dossier source :
     - Détecter type
     - Extraire texte (natif ou OCR)
     - Extraire dates candidates
     - Analyser avec Ollama (Passe 1 + fallback Passe 2)
     - Parser résultats strict
     - Valider 3 champs
     - Renommer et déplacer vers Export/
     - Ou déplacer vers Echec/
     - Enregistrer log
  8. Afficher résumé
  9. Nettoyer fichiers temporaires
- **Gestion Interruption** : Ctrl+C → cleanup + fin gracieuse
- **Gestion Erreurs** : Try/except global, nettoyage même en erreur

---

1. **Initialisation**
   - Vérification dépendances
   - Chargement config.json
   - Demande interactive dossier source si absent
   - Sélection modèle Ollama

2. **Préparation**
   - Création dossiers Export/Echec
   - Initialisation fichier log CSV

3. **Itération Fichier par Fichier**
   - Détection type fichier
   - Extraction texte (natif ou OCR)
   - Génération PDF temporaire si nécessaire
   - Extraction dates candidates
   - Analyse Ollama (dual-source si applicable)
   - Parsing réponse Ollama
   - Validation 3 champs
   - Renommage ou déplacement Echec
   - Déplacement PDF OCRisé si succès
   - Enregistrement log

4. **Finalisation**
   - Affichage résumé
   - Nettoyage fichiers temporaires (si erreur)
   - Fin du script

---

## 🎛️ Paramètres & Seuils

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| Résolution OCR | 300 DPI | Qualité optimale pour documents texte |
| Contraste Image | 2.0x | Renforcement fort pour texte clair |
| Timeout Ollama | 120s | Maximum d'attente par appel LLM |
| Format Date | YYYY-MM | Année + mois minimum requis |
| Plage Date | Actuelle -20ans | Filtrage dates candidates obsolètes |
| Thread Ollama | Daemon | Arrêt automatique si processus parent meurt |
| Seuil Inversion Image | >80 | Luminosité min pour inverser couleurs |

---

## 📝 Notes de Développement

### Points Forts de l'Architecture
- **Robustesse** : Gestion exhaustive des cas d'erreur
- **Transparence** : Logging détaillé avec préfixes catégorisés
- **Flexibilité** : Configuration JSON + CLI args
- **Non-bloquant** : Threading pour Ollama sans freeze UI
- **Nettoyage** : Suppression automatique des fichiers temporaires
- **Dual-Analysis** : Fallback intelligent PDF OCRisé → fichier original

### Optimisations Possibles
- Parallélisation du traitement multi-fichiers
- Cache des modèles Ollama
- Compression PDF avant déplacement
- Interface GUI pour sélection dossier
- Support formats supplémentaires (ODP, PPT, CSV, etc.)

### Dépendances de Tesseract
- Doit être installé séparément (non inclus dans pip)
- Linux : `apt-get install tesseract-ocr tesseract-ocr-fra`
- macOS : `brew install tesseract`
- Windows : Télécharger depuis https://github.com/UB-Mannheim/tesseract/wiki

---

## 🧪 Tests & Validation

### Cas de Test Recommandés
1. PDF natif avec texte lisible → Succès direct
2. PDF scannés dégradés → OCR + PDF temp + analyse
3. Image mal orientée → Autorotation + OCR
4. Texte trop court → Analyse Ollama sur peu de contexte
5. Champ "inconnu" → Fallback second source
6. Interruption Ctrl+C → Nettoyage fichiers temp
7. Dossier source inexistant → Demande interactive

---

## ✨ Résumé Fonctionnel

**Script minimaliste, complet et robuste** pour :
- ✅ OCR intelligent (PDF + Images)
- ✅ Analyse IA sophistiquée (dual-source)
- ✅ Renommage automatique cohérent
- ✅ Traçabilité complète
- ✅ Gestion d'erreurs gracieuse
- ✅ Nettoyage des fichiers temporaires
- ✅ Configuration flexible
- ✅ Logging détaillé
