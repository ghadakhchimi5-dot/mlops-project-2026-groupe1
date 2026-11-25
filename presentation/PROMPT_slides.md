# Prompt à coller dans Claude (joindre les images du dossier `images/`)

---

Tu es un designer de présentations techniques. Crée une présentation de **5 diapositives maximum** (format 16:9, en français) pour une séance de **suivi d'avancement noté** d'un projet MLOps universitaire. Elle doit être synthétique, visuelle et concrète : peu de texte, des schémas, des tableaux et des preuves de réalisation. **Distingue toujours ce qui est FAIT de ce qui est PRÉVU** avec un code visuel constant (par exemple ✅ Fait, 🟡 En cours, ⚪ Prévu, avec une légende). N'invente aucun chiffre : utilise uniquement les données ci-dessous.

**Livrable :** un fichier **.pptx** qui intègre les images jointes. Si tu ne peux produire qu'un artefact HTML, fais un diaporama 16:9 navigable et place, pour chaque image, un cadre réservé portant **le nom exact du fichier** pour que je puisse l'insérer. Ajoute des **notes d'orateur** (≈ 45 s par diapositive) et indique quel membre présente chaque diapositive.

**Style :** thème clinique et sobre. Bleu-vert foncé `#0F2A3A` dominant, sarcelle `#0E7C7B` en secondaire, corail `#E4572E` en accent (cibles et alertes). Fond blanc pour les diapositives de contenu, fond sombre pour la première. Titres en sans-serif gras, au moins 32 pt ; corps entre 14 et 16 pt. Pas de puces denses, pas de barres décoratives.

## Diapositive 1 — Équipe et présentation du projet

- **Titre :** CareReturn — Plateforme MLOps de prédiction du risque de réadmission hospitalière (< 30 jours) des patients diabétiques.
- **Groupe 1, trinôme.** Dépôt : `github.com/ghadakhchimi5-dot/mlops-project-2026-groupe1`.
- **Équipe et responsabilités prévues :**
  - **Ghada KHCHIMI** : ingestion, exploration, qualité des données, split par patient, DVC.
  - **Wissem ZAOUGA** : prétraitement, modèles, métriques, MLflow Tracking/Registry, analyse de drift.
  - **Akrem ELOUER** : FastAPI, Docker, GitHub Actions/CML, Prefect, déploiement Blue/Green.
  - Responsabilités partagées : observabilité, tests, revues de code et documentation.
- **Contexte :** après une hospitalisation, l'équipe soignante n'a qu'une capacité limitée d'appels et de consultations de suivi.
- **Problème :** quels dossiers examiner en priorité ? Environ **1 séjour sur 9** (11,2 %) est suivi d'une réadmission à moins de 30 jours.
- **Solution :** un score de risque calculé **au moment de la sortie** pour classer les séjours. Le budget de suivi est fixé à 20 % des dossiers. La décision reste à l'équipe soignante, et le service ML est reproductible et surveillé.
- **Objectifs fixés au cadrage (cibles, pas des résultats) :** recall@20 % ≥ 0,40 ; ROC-AUC ≥ 0,70 ; Average Precision supérieure à la prévalence ; latence p95 < 200 ms ; couverture des tests > 75 %.

## Diapositive 2 — Dataset et formulation du problème ML

- **Source :** Diabetes 130-US Hospitals for Years 1999–2008, UCI Machine Learning Repository, DOI 10.24432/C5230J, licence CC BY 4.0. Fichiers `diabetic_data.csv` (environ 18 Mo) et `IDS_mapping.csv`.
- **Volume (vérifié par notre étape de validation) :** **101 766 séjours**, **71 518 patients**, **50 colonnes** (47 variables explicatives, 2 identifiants, 1 cible). 37 colonnes textuelles, 13 numériques.
- **Tâche :** classification binaire déséquilibrée. Unité d'observation : le séjour. Cible `readmitted` : classe positive `<30` ; classe négative `>30` ou `NO`.
- **Répartition de la cible** (à montrer en graphique à barres, avec `<30` mis en évidence) : NO = 54 864 ; >30 = 35 545 ; **<30 = 11 357 (11,2 %)**.
- **Variables retenues :** 8 numériques (durée de séjour, actes de laboratoire, procédures, médicaments, consultations externes, urgences et hospitalisations antérieures, nombre de diagnostics) et 9 catégorielles (âge, type et source d'admission, mode de sortie, A1C, glycémie max, insuline, changement de traitement, traitement antidiabétique).
- **Premières observations, chacune avec la décision qui en découle :**
  - 23,5 % des patients ont plusieurs séjours (47 021 séjours concernés) → **split par patient**, pour éviter la fuite d'information.
  - Variables très incomplètes : `weight` 97 % manquant, `medical_specialty` 49 %, `payer_code` 40 % → non retenues.
  - « ? » signifie manquant, mais la valeur « None » des examens signifie « examen non réalisé » : c'est une vraie modalité, conservée.
  - Les codes administratifs sont numériques mais traités comme catégoriels.
  - Population âgée : 65 % des séjours concernent des patients de 60 à 90 ans.
  - Exclusion des sorties par décès ou en hospice (codes 11, 13, 14, 19, 20, 21) → cohorte de **99 343 séjours**.
- **Justification du choix :** problème réel, cible explicite, taille adaptée, licence ouverte, et possibilité de simuler une dérive (vieillissement de la population, hausse des admissions en urgence).

## Diapositive 3 — Architecture MLOps envisagée

Dessine **un schéma de flux de gauche à droite**, chaque bloc coloré selon son statut :

`Données UCI` → `DVC : validate → prepare → split` ✅ → `Entraînement : Pipeline scikit-learn` ✅ → `MLflow : Tracking + Model Registry (alias « champion »)` ✅ → `API FastAPI` ✅ → `Image Docker` ✅ (build vérifié localement) → `Déploiement Blue/Green` ⚪ → `Monitoring : Evidently (drift) + Prometheus` ⚪

- **Au-dessus du flux, en bandeau :** GitHub Actions CI (lint + tests + build Docker) 🟡 *workflow écrit, pas encore exécuté sur GitHub* ; rapport CML sur les Pull Requests ⚪.
- **En dessous, en bandeau :** orchestration Prefect 3 (ingestion → validation → préparation → entraînement → évaluation → enregistrement ; retries, Work Pool, CRON) ⚪.
- **Justification de chaque outil (une ligne courte par outil) :**
  - DVC : données et pipeline versionnés et reproductibles (`dvc repro`).
  - MLflow : paramètres, métriques, courbes et commit tracés ; registre avec promotion contrôlée.
  - FastAPI et Pydantic : validation stricte des entrées, documentation OpenAPI.
  - Docker : image multi-stage avec utilisateur non-root, donc portable et sûre.
  - GitHub Actions et CML : qualité vérifiée à chaque PR, résultats visibles dans la revue.
  - Prefect : planification, reprises sur erreur, réentraînement.
  - Evidently et Prometheus : détection de dérive, latence, erreurs et volume.
- **Périmètre trinôme :** tronc commun, Prefect, CML, déploiement sans interruption (Blue/Green), dashboard d'observabilité et métriques Prometheus. La boucle de réentraînement entièrement autonome est une extension, hors engagement initial.

## Diapositive 4 — Travail réalisé à ce stade

**Colonne gauche, « Réalisé » (✅) :**

- **Dépôt et qualité :** package `src/carereturn`, Poetry avec fichier de verrouillage, Ruff, pre-commit. **77 tests Pytest réussis, couverture de 88 %** (cible > 75 %).
- **Données (DVC, 5 étapes) :** 5 contrôles de validation réussis. Cohorte de 99 343 séjours. Split **59 603 / 19 870 / 19 870** (60/20/20), **0 patient commun** entre les partitions et 11,4 % de positifs dans chacune.
- **Modèles et MLflow :** Dummy, régression logistique et Random Forest suivis dans MLflow. Le Random Forest est sélectionné sur l'Average Precision en validation, puis enregistré dans le Registry (alias **@champion**) uniquement s'il bat la baseline. Le seuil de décision correspond au top 20 % des scores.
- **API :** `/health`, `/predict`, `/predict/batch` (≤ 1 000 séjours), réponses 422 sur entrée invalide, version du modèle renvoyée. Latence locale mesurée : **p95 ≈ 129 ms** sur 300 requêtes (MacBook, < 200 ms ✅).
- **Docker :** image multi-stage non-root, build vérifié localement.
- 🟡 **CI GitHub Actions :** workflow écrit (lint, tests, build), pas encore exécuté sur GitHub.
- **[À METTRE À JOUR AVANT LA SÉANCE] État du dépôt GitHub :** « code poussé, PR n° … » ou « code prêt localement, intégration par PR en cours ».

**Colonne droite, « Premiers résultats » :** image `01_val_model_comparison.png`, plus ce tableau de résultats sur le test :

| Test (19 870 séjours, Random Forest) | Résultat | Cible |
|---|---|---|
| Average Precision | 0,206 | > 0,114 (prévalence) ✅ |
| ROC-AUC | 0,660 | ≥ 0,70 ❌ |
| Recall@20 % | 0,370 | ≥ 0,40 ❌ |
| Precision@20 % | 0,211 | — |

Message à faire passer : le modèle apporte un vrai signal (environ 1,8 fois mieux que le hasard en Average Precision), mais il reste **sous les cibles**. C'est une baseline sans optimisation. La régression logistique est presque aussi performante (Average Precision en validation : 0,213 contre 0,217).

**Preuves (petites captures en bas de diapositive) :** `03b_mlflow_registry_crop.png` (Registry, version 2 @champion), `02b_mlflow_runs_crop.png` (6 runs suivis), `04b_api_swagger_crop.png` (API documentée).

**Bandeau « Difficultés rencontrées et solutions » :**

1. MLflow 2.22 est incompatible avec SQLAlchemy 2.1 → version épinglée à < 2.1.
2. Pandas lit « None » comme une valeur manquante → lecture explicite où seul « ? » compte comme manquant.
3. Patients avec plusieurs séjours → split groupé et stratifié (StratifiedGroupKFold).
4. Performances sous les cibles → plan d'amélioration (diapositive 5).

## Diapositive 5 — Planning et prochaines étapes

Montre une **frise chronologique en 4 jalons**. **[Remplacer « Semaines » par les vraies dates du calendrier du cours]**

1. **Semaines 1–2 : consolidation.** Pousser le code par PR, faire passer la CI au vert, configurer le remote DVC partagé (sans secrets dans Git), ajouter le rapport CML sur les PR.
2. **Semaines 3–4 : modèle.** Nouvelles variables (diagnostics, médicaments, antécédents), gradient boosting, pondération des classes, calibration. Optimisation **sur la validation uniquement**, le test reste gelé.
3. **Semaines 5–6 : orchestration et déploiement.** Flow Prefect (retries, Work Pool, CRON), Docker Compose, déploiement Blue/Green avec contrôle de santé et retour arrière.
4. **Semaines 7–8 : observabilité et livrable final.** Simulateur de dérive (part des 70 ans et plus et des urgences), rapport Evidently, métriques Prometheus (latence, erreurs, volume, dérive), documentation, démonstration.

**Tableau « Répartition immédiate » :**

| Membre | Priorités |
|---|---|
| Ghada | Remote DVC, nouvelles variables, données du simulateur de dérive |
| Wissem | Gradient boosting et calibration, règles de promotion dans le Registry, analyse Evidently |
| Akrem | Push et CI/CML, flow Prefect, Compose et Blue/Green, export Prometheus |

**Tableau « Risques → parades » :**

- Performances sous les cibles → enrichir les variables et tester d'autres modèles ; rediscuter les cibles à partir de la validation, jamais du test.
- Dépendances fragiles (cas MLflow/SQLAlchemy) → versions épinglées, fichier de verrouillage, CI à chaque PR.
- Remote DVC et secrets → identifiants hors de Git, accès partagé dans l'équipe.
- Complexité du Blue/Green → deux services Compose et un reverse proxy, bascule seulement après un `/health` réussi.
- Tests de dérive sur des comptages discrets (p-values KS peu fiables) → PSI et tailles d'effet, seuils calibrés sur des lots témoins.
- Données de 1999 à 2008, hôpitaux américains → limite de généralisation affichée ; la dérive est présentée comme une **simulation**.

---

## Images jointes

| Fichier | Contenu | Diapositive |
|---|---|---|
| `01_val_model_comparison.png` | Comparaison Dummy / régression logistique / Random Forest en validation (Average Precision et recall@20 %, avec la cible et la prévalence) | 4 |
| `02b_mlflow_runs_crop.png` (ou `02_mlflow_runs.png` en plein écran) | MLflow : 6 runs d'entraînement suivis | 4 |
| `03b_mlflow_registry_crop.png` (ou `03_…`) | MLflow Registry : `carereturn-readmission`, version 2, alias @champion | 4 |
| `04b_api_swagger_crop.png` (ou `04_…`) | Documentation OpenAPI de l'API (/health, /predict, /predict/batch) | 4 |
| `05_test_pr_curve.png` | Courbe Précision-Rappel sur le test (AP = 0,206 contre une prévalence de 0,114) | 4 (facultatif) |
| `06_test_confusion_matrix.png` | Matrice de confusion sur le test au seuil du top 20 % | 4 (facultatif) |
