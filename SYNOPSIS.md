# FICHE DE CADRAGE DU PROJET MLOPS — 2026/2027

**Jalon 1 — Fin de semaine 2**

## 1. Informations équipe

- **Titre du projet :** CareReturn — Plateforme MLOps de prédiction du risque de réadmission hospitalière des patients diabétiques.
- **Configuration :** trinôme.
- **Groupe :** [1].
- **Dépôt public :** `mlops-project-2026-groupe1`.
- **URL :** `https://github.com/ghadakhchimi5-dot/mlops-project-2026-groupe1`.

| Membre | Responsabilités principales |
|---|---|
| [Ghada KHCHIMI] | Ingestion, exploration, qualité des données, splits par patient et DVC |
| [Wissem ZAOUGA] | Prétraitement, modèles, métriques, MLflow Tracking/Registry et analyse de drift |
| [Akrem ELOUER] | FastAPI, Docker, GitHub Actions/CML, Prefect et déploiement Blue/Green |

L’observabilité, les tests, les revues de code et la documentation sont partagés. Les contributions seront traçables dans les commits et les Pull Requests.

## 2. Problématique métier et cas d’usage

### Contexte et décision

À la sortie d’une hospitalisation, une équipe dispose d’une capacité limitée pour organiser des appels et consultations de suivi. CareReturn proposera un classement du risque de réadmission précoce afin de prioriser les dossiers à examiner.

**Question métier :** peut-on identifier une proportion utile des futures réadmissions à moins de 30 jours en limitant le nombre de dossiers signalés, tout en maintenant un service ML reproductible et surveillé ?

Le score sera calculé **au moment de la sortie** à partir des informations disponibles sur le séjour et les antécédents enregistrés. L’équipe soignante conservera la décision de suivi. Le projet ne prétend pas démontrer une réduction réelle des réadmissions, qui nécessiterait une évaluation prospective.

### Objectifs quantifiés proposés

Ces valeurs sont des objectifs de travail, **pas des résultats déjà obtenus**.

| Indicateur | Objectif initial |
|---|---|
| Priorisation métier | Capturer au moins 40 % des réadmissions positives parmi les 20 % de séjours les mieux classés (`recall@20% ≥ 0,40`) |
| Discrimination | ROC-AUC ≥ 0,70 et Average Precision supérieure à la prévalence positive |
| Inférence | Latence p95 < 200 ms pour une requête unitaire après chargement, sur machine documentée |
| Qualité logicielle | Couverture Pytest > 75 % au livrable final |

Le budget de suivi de 20 % est une hypothèse pédagogique. Sa pertinence et les cibles seront discutées à partir de la baseline de validation, avant l’évaluation finale sur test.

## 3. Formulation machine learning et choix du dataset

### Tâche et cible

- **Type :** classification binaire déséquilibrée.
- **Unité d’observation :** un séjour hospitalier.
- **Cible source :** `readmitted`.
- **Classe positive :** `<30`, réadmission enregistrée à moins de 30 jours.
- **Classe négative :** `>30` ou `NO`.

### Source, licence et dimensions

- **Nom :** Diabetes 130-US Hospitals for Years 1999–2008.
- **Source :** https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
- **DOI :** https://doi.org/10.24432/C5230J
- **Licence :** Creative Commons Attribution 4.0 International — attribution conservée dans le dépôt.
- **Fichier :** `diabetic_data.csv`, accompagné de `IDS_mapping.csv`.

La lecture du fichier source a confirmé **101 766 lignes, 50 colonnes et 71 518 patients distincts**. Les 50 colonnes comprennent 47 variables explicatives, deux identifiants et la cible. La classe `<30` compte 11 357 séjours, soit environ 11,16 % du total.

### Conformité aux critères du guide

| Critère | Vérification |
|---|---|
| Entre 2 000 et 200 000 observations | 101 766 séjours avant filtrage |
| Taille maîtrisée | CSV d’environ 18,3 Mo |
| Au moins cinq variables numériques | Huit proposées ci-dessous |
| Au moins deux variables catégorielles | Neuf proposées ci-dessous |
| Problématique réelle et cible explicite | Réadmission précoce et priorisation du suivi |
| Licence ouverte | CC BY 4.0 |
| Potentiel de dérive | Modification simulée de la composition de la population et des admissions |

**Variables numériques proposées :** `time_in_hospital`, `num_lab_procedures`, `num_procedures`, `num_medications`, `number_outpatient`, `number_emergency`, `number_inpatient`, `number_diagnoses`.

**Variables catégorielles proposées :** `age`, `admission_type_id`, `admission_source_id`, `discharge_disposition_id`, `A1Cresult`, `max_glu_serum`, `insulin`, `change`, `diabetesMed`. Les codes administratifs restent catégoriels malgré leur représentation numérique.

### Protocole de préparation et prévention des fuites

1. Vérifier le schéma, l’unicité des séjours, les valeurs de la cible et les plages admissibles.
2. Traiter `?` comme valeur manquante. Conserver la modalité littérale `None` des examens biologiques, qui indique un examen non réalisé.
3. Exclure de la cohorte les sorties correspondant à un décès ou à une prise en charge en hospice : codes 11, 13, 14, 19, 20 et 21 du dictionnaire fourni. Cette règle définit le périmètre pédagogique et devra être documentée.
4. Séparer les données par `patient_nbr`, avec des proportions proches de 60 % entraînement / 20 % validation / 20 % test et un équilibre approximatif des classes.
5. Retirer `patient_nbr` et `encounter_id` des variables explicatives ; conserver les identifiants uniquement pour les contrôles de séparation.
6. Ajuster l’imputation, l’encodage et la normalisation uniquement sur l’entraînement, dans une `Pipeline` scikit-learn.
7. Sélectionner modèle, hyperparamètres et seuil sur validation ; réserver le test à l’évaluation finale.

Le contrôle initial de faisabilité a produit une cohorte de **99 343 séjours** après les exclusions proposées. Une séparation déterministe a donné **59 603 / 19 870 / 19 870** séjours, avec **zéro patient commun** entre partitions. Aucun modèle n’a été entraîné à ce stade.

### Modèles et évaluation envisagés

- Baseline naïve : `DummyClassifier`.
- Premier modèle : régression logistique.
- Comparaison : Random Forest ; pondération des classes étudiée sur validation.
- Critère principal de classement des modèles : Average Precision, accompagnée du `recall@20%`.
- Métriques complémentaires : ROC-AUC, rappel, précision, F1, matrice de confusion et calibration du score.
- Analyse complémentaire : performances par tranche d’âge lorsque les effectifs sont suffisants.

Les données historiques américaines ne démontrent pas une généralisation à un hôpital actuel ou tunisien. La cible représente les réadmissions enregistrées dans la source ; elle n’est pas une mesure exhaustive de toutes les réadmissions possibles.

## 4. Architecture MLOps prévisionnelle

| Pilier | Réalisation prévue |
|---|---|
| Craftsmanship | Package `src/`, Poetry et fichier de verrouillage, typage, docstrings, Ruff et pre-commit |
| Données | DVC, `dvc.yaml`, données brutes immuables et remote partagé configuré sans secrets dans Git |
| Tracking et gouvernance | MLflow Tracking : paramètres, métriques, courbes, données et commit source ; enregistrement et promotion du modèle retenu dans le Registry |
| Serving | FastAPI/Pydantic ; `/health`, `/predict` et batch ; validation des entrées et version du modèle dans la réponse |
| Packaging | Docker multi-stage, utilisateur non-root, Docker Compose |
| CI/CD | GitHub Actions : lint et Pytest ; tests unitaires, schéma, intégration et inférence ; rapport CML avec courbes sur PR |
| Orchestration | Prefect 3.x : ingestion → validation → préparation → entraînement → évaluation → enregistrement ; retries, Work Pool et CRON |
| Déploiement trinôme | Blue/Green avec contrôles du nouveau service, bascule du trafic et retour à la version précédente |
| Observabilité trinôme | Simulateur de drift, dashboard HTML Evidently et export Prometheus : latence, erreurs, volume et indicateurs de dérive |

Le tronc commun, Prefect et CML sont complétés par les exigences du trinôme de la section 2.1 du guide : déploiement sans interruption, dashboard d’observabilité et métriques Prometheus. La fermeture entièrement autonome de la boucle de réentraînement est une extension, pas un engagement du périmètre initial.

## 5. Scénario de data drift simulé

### Scénario

Construire une référence à partir des données d’entraînement puis simuler des lots de production à partir de la validation, en augmentant progressivement la proportion de patients âgés de 70 ans et plus et d’admissions en urgence. Le rééchantillonnage conservera les lignes complètes pour préserver leurs relations internes.

Un lot témoin sans décalage sera comparé aux lots modifiés. Les paramètres et la graine de simulation seront enregistrés. Le dataset ne permettant pas ici une séparation temporelle fiable, ce scénario sera explicitement présenté comme une **simulation**, sans prétendre reconstituer une évolution historique.

### Détection et seuils initiaux

- **Numérique :** KS et PSI ; les variables retenues étant surtout des comptages discrets, interpréter les p-values KS avec prudence et compléter par les tailles d’effet.
- **Catégoriel :** χ², en regroupant les modalités rares lorsque nécessaire.
- **Alerte par variable :** PSI ≥ 0,20, ou test significatif après correction des comparaisons multiples accompagné d’un écart substantiel à calibrer sur les lots témoins.
- **Alerte critique globale :** au moins 30 % des variables surveillées déclarées en dérive.

Ces seuils constituent une configuration initiale à vérifier expérimentalement, pas des constantes universelles.

### Réaction prévue

Générer un rapport HTML Evidently, exposer les indicateurs à Prometheus et analyser les variables concernées. Un flow Prefect de réentraînement pourra être lancé après validation humaine si des données étiquetées appropriées sont disponibles. La promotion exigera des contrôles qualité et une comparaison au modèle déployé.

**Data drift et concept drift seront distingués :** une modification des entrées ne prouve pas une modification de la relation entre entrées et cible. Lorsque les résultats de réadmission seront disponibles, le suivi d’Average Precision, du rappel et de la calibration permettra d’identifier une dégradation nécessitant une investigation. Sans labels, aucune conclusion de concept drift ne sera annoncée.


**Note de préparation :** le template Google Drive séparé n’a pas pu être consulté ; ce document reprend le canevas officiel inclus dans le PDF. Le lien du formulaire et la date limite exacte restent à communiquer.
