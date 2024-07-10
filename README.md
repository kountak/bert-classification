# bert-classification
Ce répertoire est un exercice  pour illustrer l'implémentation du modèle BERT de HugginFace.  


## bert-classification.py
Ce script Python implémente et entraîne un modèle BERT personnalisé pour la classification de texte à partir de données IMDB. Voici un résumé des composants et des étapes du script :

### Importations
- **Bibliothèques** : pandas, torch, transformers, sklearn, gradio.
- **Modules spécifiques** : nn, Dataset, DataLoader, BertModel, AutoTokenizer, AutoModelForMaskedLM, Adam, tqdm.

### Classes et Fonctions

1. **Classe `IMDBDataset`** :
   - **Initialisation** : Lit un fichier CSV, encode les labels de classe, initialise un tokenizer BERT.
   - **Méthodes** :
     - `__len__()` : Retourne la longueur du DataFrame.
     - `__getitem__(index)` : Tokenise le texte et retourne les inputs et les labels sous forme de tenseurs.

2. **Classe `CustomBert`** :
   - **Initialisation** : Charge un modèle BERT pré-entraîné et ajoute un classifieur linéaire.
   - **Méthode `forward(input_ids, attention_mask)`** : Passe les inputs dans BERT et le classifieur.
   - **Méthode `save_checkpoint(path)`** : Sauvegarde les poids du modèle.

3. **Fonction `training_step(model, data_loader, loss_fn, optimizer)`** :
   - Entraîne le modèle pour une époque, calcule et accumule la perte totale.

4. **Fonction `evaluation(model, test_dataloader, loss_fn)`** :
   - Évalue le modèle sur les données de test, calcule la perte et la précision.

5. **Fonction `main()`** :
   - Définit les hyperparamètres (nombre d'époques, taux d'apprentissage, taille de lot).
   - Initialise le dispositif (GPU/CPU), les ensembles de données et les chargeurs de données.
   - Crée une instance du modèle BERT personnalisé, définit la fonction de perte et l'optimiseur.
   - Entraîne et évalue le modèle sur plusieurs époques.
   - Sauvegarde les poids du modèle après l'entraînement.

### Exécution
- La fonction `main()` est exécutée si le script est lancé directement.


https://drive.google.com/file/d/1-KU489KJs40N-M8yZz1S2zPAAc7JVTTR/view?usp=sharing

http://127.0.0.1:4445/predict
