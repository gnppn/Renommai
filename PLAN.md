J'ai des centaines de documents administratif scannés, et mis en vrac dans un dossier depuis des années. Je veux un outil automatisé qui les range pour moi.

# Objectif
Je veux obtenir des fichiers PDF :
1. complets (sous-couche texte recherchable)
2. renommés dans un format YYYY-MM-DD Institution Objet.ext
3. accompagnés d'un fichier texte contenant le texte extrait

# Instructions

Tu dois créer un script, un outil, le plus simple et réparable possible

dans un dossier "Export" horodaté
copier les échecs dans
multiplateformes (Windows, macOS, Debian)

en entrée des fichiers images (PDF complets, PDF sans sous-couche texte, JPG, PNG, etc)
exploitant les bonnes pratiques d'analyse de ce type de fichiers
exploitant au mieux l'OCR et un modèle d'IA pour analyser le contenu
dans une logique d'analyse minimale des documents

échec (pas de texte ou erreur de traitement), placer le 

être le plus transparent sur l'action en cours, dans un format convivial
tenir un log de ses actions
automatiser l'installation, par exemple les dépendances (y compris les modèles d'IA)


# Configuration

L'outil doit être le plus simple possible à utiliser :

1. avoir une configuration par défaut
2. demander à chaque utilisation de changer les éléments configurables
3. modifier le fichier de configuration si l'utilisateur la change dans l'outil

L'utilisateur doit pouvoir configurer :
- le dossier source des fichiers
- le dossier d'export des fichiers
- les modèles d'IA utilisés
- les messages affichés lors de l'exécution de l'outil

La configuration doit :
- tenir dans un fichier séparé
- contenir les prompts d'IA dans des fichiers séparés
- 

# Sécurité
- Limite autant que possible l'utilisation de fichiers temporaires
- Assure-toi que les fichiers temporaires et informations d'exécution sont supprimées après l'interruption (volontaire ou complétion) de l'outil