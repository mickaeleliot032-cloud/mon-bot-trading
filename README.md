# Agent de trading CAC 40 — V4

Agent intraday de **paper trading** : il analyse les 40 actions du CAC 40,
classe les opportunités et envoie les événements importants sur Telegram.
Il ne passe aucun ordre auprès d'un courtier.

## Ce que change la V4

- scan de collecte à **9 h 05**, premier classement à **9 h 15**, puis toutes
  les 15 minutes jusqu'à **16 h 00** (heure de Paris) ;
- une seule position et une seule entrée maximum par journée ;
- score progressif : `SURVEILLANCE ≥ 65`, `SIGNAL ≥ 72`, `FORT ≥ 80` ;
- classement fondé sur tendance, momentum, force relative au CAC 40, volumes,
  volatilité, secteur, contexte de marché et actualités ;
- confirmation sur deux scans pour un signal normal, mais entrée immédiate
  lorsqu'un score fort atteint 80 ;
- position surveillée chaque minute pendant les fenêtres GitHub Actions,
  indépendamment des scans de classement ;
- stop dynamique selon l'ATR, passage au point mort à +0,65 % ;
- TP à +1 %, étendu à +2 % lorsque +1 % est atteint avant 10 h, avec stop
  suiveur ;
- sortie forcée avant **17 h 20** ;
- frais aller-retour de 2 € inclus dans le capital fictif de 1 000 € ;
- état quotidien et journal CSV persistants ;
- aucun secret dans le dépôt.

La composition a été vérifiée avec la publication Euronext du
[30 juin 2026](https://live.euronext.com/sites/default/files/documentation/index-composition/CAC_40_Index_Composition.pdf).

## Installation

Python 3.11 ou 3.12 :

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Chargez ensuite les variables de `.env` dans votre environnement, puis :

```bash
python mon_bot.py
```

Sans variables Telegram, le moteur fonctionne et écrit les notifications dans
les logs.

## Exécution gratuite avec GitHub Actions

Le workflow `.github/workflows/trading-agent.yml` se déclenche du lundi au
vendredi toutes les cinq minutes entre 9 h et 17 h 55, heure de Paris. Le moteur
n'effectue toutefois les classements qu'à 9 h 05, 9 h 15, puis toutes les
15 minutes jusqu'à 16 h. Les autres passages servent à surveiller une éventuelle
position et à envoyer le bilan après la sortie forcée de 17 h 20.

L'état JSON et le journal CSV sont sauvegardés dans un cache GitHub privé au
workflow. Ils ne sont ni ajoutés au dépôt public, ni accessibles aux visiteurs.

Après fusion de la V4 dans `main`, ajoutez ces deux secrets dans
**Settings → Secrets and variables → Actions → New repository secret** :

- `TELEGRAM_BOT_TOKEN` : nouveau jeton généré par BotFather ;
- `TELEGRAM_CHAT_ID` : identifiant du chat destinataire.

Le workflow peut ensuite être testé manuellement depuis l'onglet **Actions**,
workflow **Agent de trading V4**, bouton **Run workflow**. Les exécutions
planifiées utilisent uniquement la version présente sur la branche par défaut.

## Variables obligatoires pour Telegram

- `TELEGRAM_BOT_TOKEN` : nouveau jeton généré par BotFather ;
- `TELEGRAM_CHAT_ID` : identifiant du chat destinataire.

L'ancien jeton figurait publiquement dans la première version du dépôt. Il doit
rester révoqué dans BotFather et ne jamais être recopié dans un fichier.

## Validation

```bash
pip install -r requirements-dev.txt
ruff check .
pytest -q
```

Les résultats restent des simulations. Yahoo Finance peut fournir des données
retardées ou incomplètes ; la V4 ne doit pas être utilisée pour exécuter
automatiquement des ordres réels.
