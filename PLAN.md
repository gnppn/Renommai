J'ai des centaines de documents administratif scannés, et mis en vrac dans un dossier depuis des années. Je veux un outil automatisé qui les range pour moi.

# Objectif
Je veux obtenir des fichiers PDF :
1. complets (sous-couche texte recherchable)
2. renommés dans un format YYYY-MM-DD Institution Objet.ext
3. accompagnés d'un fichier texte contenant le texte extrait

# Instructions

Tu dois créer un script, le plus simple et réparable possible.

Le script doit : 
- exporter les PDF dans un dossier "Export" horodaté
- copier les échecs (pas de texte ou erreur de traitement) dans un sous-dossier du dossier Exports horodaté
- fournir la sous-couche texte exportée de chaque PDF dans un fichier texte, dans le même dossier que les exports et les échecs
- être multiplateformes (Windows, macOS, Debian)
- fournir un log CSV du traitement dans le dossier Exports horodaté

L'analyse des fichiers :
- en entrée des fichiers images (PDF complets, PDF sans sous-couche texte, JPG, PNG, etc)
- exploiter les bonnes pratiques d'analyse de ce type de fichiers
- exploiter au mieux l'OCR et un modèle d'IA pour analyser le contenu
dans une logique d'analyse minimale des documents
- s'appuyer sur des outils 100% locaux (bibliothèques et modèles d'IA, dont vision)

En fonctionnement, le script doit :
- être le plus transparent sur l'action en cours, dans un format convivial
- automatiser l'installation, par exemple les dépendances (y compris les modèles d'IA)


# Configuration

L'outil doit être le plus simple possible à utiliser :

1. avoir une configuration par défaut
2. être interactif pour l'utilisateur, donc au lancement 
    a. demander à chaque utilisation de changer les éléments configurables
    b. modifier le fichier de configuration si l'utilisateur la change dans l'outil
3. installer lui-même les dépendances manquantes (y compris les modèles d'IA)

Au lancement et via le fichier config, l'utilisateur doit pouvoir configurer :
- le dossier source des fichiers
- le dossier d'export des fichiers
- les modèles d'IA utilisés
- les messages affichés lors de l'exécution de l'outil

La configuration doit :
- tenir dans un fichier séparé
- contenir les prompts d'IA dans des fichiers séparés

# Sécurité
- Limite autant que possible l'utilisation de fichiers temporaires
- Assure-toi que les fichiers temporaires et informations d'exécution sont supprimées après l'interruption (volontaire ou complétion) de l'outil