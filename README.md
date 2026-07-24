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
- position surveillée chaque minute, indépendamment des scans de classement ;
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

Chargez ensuite les variables de `.env` dans votre hébergeur, puis :

```bash
python mon_bot.py
```

Sans variables Telegram, le moteur fonctionne et écrit les notifications dans
les logs. Sur Render, utilisez `python mon_bot.py` comme commande de démarrage
et montez un disque persistant sur `/data`.

## Variables obligatoires pour Telegram

- `TELEGRAM_BOT_TOKEN` : nouveau jeton généré par BotFather ;
- `TELEGRAM_CHAT_ID` : identifiant du chat destinataire.

L'ancien jeton figurait publiquement dans la première version du dépôt. Il doit
être révoqué dans BotFather puis remplacé dans l'environnement Render.

## Validation

```bash
pip install -r requirements-dev.txt
ruff check .
pytest -q
```

Les résultats restent des simulations. Yahoo Finance peut fournir des données
retardées ou incomplètes ; la V4 ne doit pas être utilisée pour exécuter
automatiquement des ordres réels.
