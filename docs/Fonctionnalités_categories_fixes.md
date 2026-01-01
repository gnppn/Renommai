# Fonctionnalités - RenAIme OCR & Renommage (Version Catégories Fixes) 🧪

> **Version expérimentale** basée sur `renommeur.py` avec listes restrictives de catégories et institutions.

## 🎯 Objectif Global

Script de tri et renommage automatique de documents (PDF, PNG, JPG, DOCX, XLSX) basé sur :
- **OCR** : Extraction de texte via Tesseract
- **Analyse Vision** : Pré-analyse visuelle via modèle vision (minicpm-v ou llava-llama3)
- **Analyse IA** : Détection Institution/Objet/Date via Ollama (llama3)
- **Normalisation** : Correspondance avec listes restrictives d'institutions et de catégories
- **Renommage** : Génération automatique de noms fichiers selon le format `YYYY-MM Institution Objet.ext` (en Title Case)

---

## 🆕 Différences avec renommeur.py

| Fonctionnalité | renommeur.py | renommeur_categories_fixes.py |
|----------------|--------------|-------------------------------|
| **Catégories** | Libres (IA décide) | Restrictives (~100 types prédéfinis) |
| **Institutions** | Libres (IA décide) | Normalisées via cache (~250 variantes) |
| **Fichiers requis** | prompts/ uniquement | prompts/ + categories_documents.txt + institutions_cache.txt |
| **Normalisation** | Simplification basique | Fuzzy matching + mapping mots-clés |
| **Prompt IA** | Standard | Enrichi avec liste des catégories autorisées |

---

## 📁 Fichiers Spécifiques

### categories_documents.txt

Liste restrictive des types de documents administratifs (~100 catégories) :

```
# Format: Catégorie # Indice pour l'IA

# === IMPÔTS ET FISCALITÉ ===
Avis d'imposition                # Document fiscal annuel
Déclaration de revenus           # Formulaire 2042, etc.
Taxe foncière                    # Propriétés bâties/non bâties
...

# === BANQUE ET FINANCE ===
Relevé de compte                 # Relevé bancaire mensuel
Contrat de prêt                  # Convention de crédit
...

# === ASSURANCE ===
Contrat d'assurance              # Police d'assurance
Attestation d'assurance          # Certificat de couverture
...
```

**Catégories couvertes** :
- Impôts et fiscalité
- Banque et finance
- Assurance
- Emploi et salariat
- Retraite
- Sécurité sociale et santé
- CAF et prestations sociales
- Logement
- Énergie et services
- Véhicule
- Identité et état civil
- Éducation
- Justice et administration
- Divers

### institutions_cache.txt

Cache des institutions connues avec variantes (~250 entrées) :

```
# Format: Nom officiel | Variantes (séparées par des virgules)

# === IMPÔTS ET ADMINISTRATION FISCALE ===
Direction Générale des Finances Publiques | DGFIP, Impôts, SIP, Service des Impôts

# === SÉCURITÉ SOCIALE ===
Caisse Primaire d'Assurance Maladie | CPAM, Assurance Maladie, Sécurité Sociale, Ameli

# === BANQUES ===
BNP Paribas | BNP, BNPP
Société Générale | SG, Socgen
Crédit Agricole | CA, LCL
...
```

**Domaines couverts** :
- Impôts et administration fiscale
- Sécurité sociale
- Retraite complémentaire
- Emploi
- Banques
- Assurances
- Énergie
- Télécommunications
- État et administration
- Justice
- Santé
- Éducation
- Commerce

---

## 📋 Configuration & Initialisation

### Fichiers de Configuration

| Fichier | Description |
|---------|-------------|
| `config.json` | Configuration source/modèle (identique à renommeur.py) |
| `categories_documents.txt` | Liste restrictive des catégories |
| `institutions_cache.txt` | Cache des institutions connues |

### Chargement des Listes Restrictives

**Au démarrage du script** :

```
📋 Chargement des listes restrictives...
      ✅ 97 catégories de documents chargées
      ✅ 247 variantes d'institutions chargées
```

#### Fonction : `load_categories()`
- Lit `categories_documents.txt`
- Ignore lignes vides et commentaires (`#`)
- Extrait catégorie avant le commentaire explicatif
- Retourne : `dict {nom_lower: nom_original}`
- Cache en mémoire pour performance

#### Fonction : `load_institutions()`
- Lit `institutions_cache.txt`
- Parse format `Nom officiel | Variante1, Variante2, ...`
- Crée mapping variante → nom officiel
- Retourne : `dict {variante_lower: nom_officiel}`
- Cache en mémoire pour performance

---

## 🔄 Flux de Traitement (Différences)

### Étapes identiques à renommeur.py

1. ✅ Extraction texte (PDF natif, OCR, DOCX, XLSX)
2. ✅ Analyse Vision (minicpm-v ou llava-llama3)
3. ✅ Extraction dates candidates
4. ✅ Parsing réponse IA multi-variante (3 variantes)
5. ✅ Génération nom fichier
6. ✅ Export/Échec avec log CSV

### Étapes modifiées

#### 🆕 Analyse IA enrichie avec catégories

Fonction modifiée : `analyze_ollama()`

Le prompt envoyé à Ollama inclut maintenant la liste des catégories autorisées :

```
[CATÉGORIES AUTORISÉES POUR L'OBJET]
Acte de mariage, Acte de naissance, Acte de vente, Arrêt de travail, ...

[NOM FICHIER ORIGINAL]
document_scan.pdf

[TEXTE VISION IA]
Logo EDF en haut. Facture électricité...

[TEXTE TESSERACT (OCR)]
EDF - FACTURE - Montant: 127,84€...

[DATES CANDIDATES]
2024-03
```

#### 🆕 Normalisation post-parsing

Après le parsing de la réponse IA, les champs sont normalisés :

```python
# Sélection meilleure variante (comme renommeur.py)
inst = best_match_with_title(inst_variants[:3], title)
obj = best_match_with_title(obj_variants[:3], title)

# Simplification institution (comme renommeur.py)
inst = simplify_institution_name(inst)

# 🆕 Normalisation via cache institutions
inst = normalize_institution(inst)

# 🆕 Normalisation via liste catégories
obj = normalize_object(obj)
```

---

## 🔍 Fonctions de Normalisation

### normalize_institution(extracted_name)

**Objectif** : Convertir un nom d'institution extrait vers le nom officiel du cache.

**Algorithme** :
1. Si "inconnu" → retourne "inconnu"
2. **Nettoyage phrases IA** : Détecte "specified", "deduced", "related" → cherche institution dans le texte ou retourne "inconnu"
3. **Recherche exacte** : Si nom exact dans le cache → retourne nom officiel
4. **Recherche par inclusion** : Si une variante (≥3 chars) est contenue dans le nom → retourne nom officiel
5. **Fuzzy matching** : Similarité SequenceMatcher ≥ 70% → retourne meilleur match
6. **Fallback** : Retourne le nom original (permet découverte nouvelles institutions)

**Exemples** :
| Extrait par IA | Résultat normalisé |
|----------------|-------------------|
| "CPAM" | "Caisse Primaire d'Assurance Maladie" |
| "BNP" | "BNP Paribas" |
| "Impots" | "Direction Générale des Finances Publiques" |
| "tax-related document" | "Direction Générale des Finances Publiques" |
| "None specified, but deduced from context" | "inconnu" |

### normalize_object(extracted_object)

**Objectif** : Convertir un type de document extrait vers une catégorie standard.

**Algorithme** :
1. Si "inconnu" → retourne "inconnu"
2. Nettoyage guillemets
3. **Mapping mots-clés prioritaires** :
   - "impôt", "imposition", "revenu" → "Avis d'imposition"
   - "salaire", "paie" → "Bulletin de salaire"
   - "facture" → "Facture"
   - "relevé", "bancaire" → "Relevé de compte"
   - etc.
4. **Recherche exacte** dans les catégories
5. **Recherche par mots** : Tous les mots de la catégorie dans l'objet (ou inverse)
6. **Fuzzy matching** : Similarité ≥ 60% → retourne meilleur match
7. **Fallback** : Retourne l'objet original (permet découverte nouvelles catégories)

**Exemples** :
| Extrait par IA | Résultat normalisé |
|----------------|-------------------|
| "AVIS D'IMPÔT SUR LE REVENU" | "Avis d'imposition" |
| "Fiche De Paie" | "Bulletin de salaire" |
| "Facture Electricite" | "Facture" |
| "Releve Bancaire Mensuel" | "Relevé de compte" |
| "Tracking Package" | "Tracking Package" (non normalisé) |

---

## 📊 Logging Spécifique

### Affichage Console

```
🤖 Vérification des modèles Ollama...
      ✅ Modèles disponibles
      📱 Puissance détectée: Limitée (RAM: 15.5GB, VRAM: 0.0GB)
      👁️  Modèle vision sélectionné: minicpm-v:latest

📋 Chargement des listes restrictives...
      ✅ 97 catégories de documents chargées
      ✅ 247 variantes d'institutions chargées

============================================================
📄 FICHIER: 008.jpg
============================================================
  🖼️  [IMAGE] Création PDF searchable... ✅
  👁️  [MINICPM-V] Analyse vision 1ère page...
      ✅ 2010 caractères extraits
  📅 [DATES] Recherche...
      ✅ 1 date(s) trouvée(s): ['2009']
  🧠 [OLLAMA] Analyse IA (passe 1)...
      📝 Réponse IA: Institution 1: Impots...
      ✅ Confiance haute
  🏷️  [RÉSULTAT] Direction Générale des Finances Publiques | Avis d'imposition | 2009-01
      ✅ Validation OK
  🎉 EXPORTÉ: 2009-01 Direction Générale Des Finances Publiques Avis D'imposition.pdf
```

---

## 🎛️ Paramètres Spécifiques

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| Seuil fuzzy institutions | 70% | Similarité minimum pour match institution |
| Seuil fuzzy catégories | 60% | Similarité minimum pour match catégorie |
| Longueur min variante | 3 chars | Pour recherche par inclusion |

---

## 📚 Référence des Fonctions Spécifiques

### Chargement des Listes

#### `load_categories()`
- **Objectif** : Charger la liste des catégories depuis `categories_documents.txt`
- **Format fichier** : `Catégorie # Commentaire explicatif`
- **Retour** : `dict {nom_lower: nom_original}`
- **Cache** : Variable globale `_categories_cache`

#### `load_institutions()`
- **Objectif** : Charger le cache des institutions depuis `institutions_cache.txt`
- **Format fichier** : `Nom officiel | Variante1, Variante2, ...`
- **Retour** : `dict {variante_lower: nom_officiel}`
- **Cache** : Variable globale `_institutions_cache`

### Normalisation

#### `normalize_institution(extracted_name)`
- **Objectif** : Normaliser un nom d'institution vers le nom officiel
- **Entrée** : Nom extrait par l'IA
- **Algorithme** : Exact → Inclusion → Fuzzy (70%)
- **Retour** : Nom officiel ou nom original si non trouvé

#### `normalize_object(extracted_object)`
- **Objectif** : Normaliser un type de document vers une catégorie standard
- **Entrée** : Type de document extrait par l'IA
- **Algorithme** : Mots-clés → Exact → Mots → Fuzzy (60%)
- **Retour** : Catégorie standard ou objet original si non trouvé

#### `get_categories_for_prompt()`
- **Objectif** : Générer la liste des catégories pour le prompt IA
- **Retour** : String des catégories séparées par virgules

---

## 🧪 État Expérimental

### Avantages attendus
- ✅ Cohérence des noms de fichiers (catégories standardisées)
- ✅ Reconnaissance des institutions même avec variantes
- ✅ Réduction des erreurs de l'IA (liste restrictive)
- ✅ Facilité d'extension (ajout de catégories/institutions)

### Limitations connues
- ⚠️ Documents hors liste → Catégorie non normalisée
- ⚠️ Institutions inconnues → Nom brut de l'IA
- ⚠️ Fuzzy matching peut produire faux positifs
- ⚠️ Performance légèrement impactée par le matching

### Améliorations futures possibles
- Apprentissage des nouvelles catégories/institutions rencontrées
- Suggestions de nouvelles entrées pour les listes
- Statistiques d'utilisation des catégories
- Mode interactif de validation

---

## ✨ Résumé Fonctionnel

**Script expérimental** ajoutant à renommeur.py :
- ✅ Liste restrictive de ~100 catégories de documents administratifs
- ✅ Cache de ~250 variantes d'institutions connues
- ✅ Normalisation automatique via fuzzy matching
- ✅ Mapping de mots-clés vers catégories
- ✅ Prompt IA enrichi avec catégories autorisées
- ✅ Fallback vers valeur originale si non reconnu

**Fichiers requis supplémentaires** :
- `categories_documents.txt`
- `institutions_cache.txt`
