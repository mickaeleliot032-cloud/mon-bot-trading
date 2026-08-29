# Pipeline Machine Learning — prédiction Top 3 CAC 40

Ce pipeline est volontairement séparé de la logique de trading. Il n'ouvre ni ne
ferme aucune position et ne modifie pas le score V4.3.

## Objectif

Le modèle apprend à estimer, à partir des informations disponibles au moment du
signal, la probabilité qu'une action termine dans le Top 3 du CAC 40 selon sa
performance maximale intraday.

La cible est :

`TOP3 = 1 si RANG_FIN_JOURNEE <= 3, sinon 0`.

Les colonnes de résultat de fin de journée ne sont pas utilisées comme variables
d'entrée afin d'éviter toute fuite d'information.

## Source des données

Le workflow lit directement les onglets `SIGNAUX` et `SUIVI` du Google Sheet,
puis les fusionne avec `ID_SIGNAL`.

Variables principales : scores, variation de séance, EMA, VWAP, volume relatif,
performance et surperformance CAC 40, momentum 15 min, RSI, ATR, gap, secteur,
contexte marché/secteur/news et heure du signal.

## Secrets GitHub nécessaires

Dans `Settings > Secrets and variables > Actions`, créer :

- `ML_GOOGLE_SHEET_ID` : identifiant du classeur Google Sheet ;
- `GOOGLE_SERVICE_ACCOUNT_JSON` : contenu JSON complet du compte de service
  Google utilisé en lecture seule.

Le Google Sheet doit être partagé avec l'adresse e-mail du compte de service,
avec un accès Lecteur au minimum.

Ne jamais ajouter le fichier JSON du compte de service dans le dépôt.

## Entraînement

Le workflow `Entraînement ML Top 3` peut être lancé manuellement depuis l'onglet
Actions. Il est également prévu une fois par semaine le dimanche soir.

Des garde-fous empêchent l'entraînement si la base est trop petite :

- au moins 60 lignes exploitables ;
- au moins 5 exemples Top 3 ;
- au moins 5 exemples hors Top 3 ;
- au moins 5 journées distinctes.

La validation respecte l'ordre temporel : environ 80 % des premières journées
servent à l'entraînement et les 20 % les plus récentes à la validation.

## Sorties

Chaque entraînement réussi publie un artifact GitHub conservé 90 jours :

- `top3_model.joblib` : pipeline complet de régression logistique ;
- `metrics.json` : métriques de validation ;
- `dataset.csv` : dataset construit pour audit.

Les métriques comprennent notamment précision, rappel, exactitude, Brier score,
taux de Top 3 de référence et ROC AUC lorsque la validation contient les deux
classes.

## Étape suivante

Le modèle reste pour l'instant hors ligne. Une future étape pourra charger le
modèle dans l'agent et écrire une colonne `PROBA_TOP3`, d'abord à titre purement
informatif, sans modifier les règles d'entrée/sortie.
