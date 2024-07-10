# bert-classification
Ce répertoire est un exercice  pour illustrer l'implémentation du modèle BERT de HugginFace.  


## I. bert-classification.py
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
  
Le lien google drive pour le modèle est https://drive.google.com/file/d/1-KU489KJs40N-M8yZz1S2zPAAc7JVTTR/view?usp=sharing

# II. demo.py  

1. **Classe `CustomBert`** :
   - **Initialisation** : Charge un modèle BERT pré-entraîné et ajoute un classifieur linéaire pour 6 classes.
   - **Méthode `forward(input_ids, attention_mask)`** : Passe les inputs dans BERT et le classifieur pour obtenir les prédictions.

2. **Chargement du modèle** :
   - Crée une instance du modèle `CustomBert`.
   - Charge les poids du modèle sauvegardés à partir du fichier `my_custom_bert_kk.pth`. Le modèle étant entrainer sur le GPU de google colab, on lui précise l'option `map_location=torch.device('cpu')` vu que l'on veut lancer le démo sur le cpu.

3. **Fonction `classifier_fn(text: str)`** :
   - Définition des labels de classe.
   - Tokenise le texte d'entrée avec un tokenizer BERT.
   - Passe les inputs dans le modèle pour obtenir les prédictions.
   - Retourne le label de classe prédite.

4. **Interface utilisateur avec Gradio** :
   - Crée une interface `gr.Interface` avec la fonction `classifier_fn`.
   - Définit l'entrée comme du texte et la sortie comme du texte.
   - Lance l'interface.

# III. api.py
Ce script Python crée une API REST avec FastAPI pour classifier des textes à l'aide d'un modèle BERT personnalisé. Voici un résumé des composants et des étapes du script :


### Composants

1. **Initialisation de FastAPI** :
   - Création d'une instance de l'application FastAPI.

2. **Classe `RequestPost`** :
   - Définition d'un modèle de requête Pydantic pour valider le format des données d'entrée (`text`).

3. **Classe `CustomBert`** :
   - **Initialisation** : Charge un modèle BERT pré-entraîné et ajoute un classifieur linéaire pour 6 classes.
   - **Méthode `forward(input_ids, attention_mask)`** : Passe les inputs dans BERT et le classifieur pour obtenir les prédictions.

4. **Chargement du modèle** :
   - Crée une instance du modèle `CustomBert`.
   - Charge les poids du modèle sauvegardés à partir du fichier `my_custom_bert_kk.pth`.

5. **Fonction `classifier_fn(text: str)`** :
   - Définition des labels de classe.
   - Tokenise le texte d'entrée avec un tokenizer BERT.
   - Passe les inputs dans le modèle pour obtenir les prédictions.
   - Retourne le label de classe prédite.

6. **Endpoint FastAPI** :
   - Définit un endpoint POST `/predict` qui prend une requête de type `RequestPost`.
   - Utilise la fonction `classifier_fn` pour prédire la classe du texte et retourne le résultat.

7. **Exécution de l'application** :
   - Utilise Uvicorn pour lancer l'application FastAPI sur l'hôte `127.0.0.1` et le port `4445`.

### Exécution
- Le script démarre une API FastAPI qui permet de poster des textes pour obtenir des prédictions de classe via un modèle BERT et l'endpoint se trouve sur http://127.0.0.1:4445/predict.



