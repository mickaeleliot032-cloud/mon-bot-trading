import datetime
import time
import feedparser
import pandas as pd
import requests
import yfinance as yf

# ==========================================
# CONFIGURATION TELEGRAM (FONCTIONNELLE)
# ==========================================
TELEGRAM_TOKEN = "8835351020:AAERKtAUihKC0RvkZaPtaEnu2vF1K0xPkPQ"
TELEGRAM_CHAT_ID = "8194258503"
TICKER_NAME = "^FCHI"  # Indice CAC 40


def envoyer_alerte_telegram(texte):
    """Envoie un message formaté sur votre application Telegram."""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": texte,
            "parse_mode": "Markdown",
        }
        response = requests.post(url, json=payload)
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Erreur réseau d'envoi Telegram : {e}")
        return False


# ==========================================
# FONCTIONS DE MARCHÉ & ALGORITHME
# ==========================================
def recuperer_donnees_cac40():
    """Récupère l'historique récent pour les calculs techniques."""
    try:
        ticker = yf.Ticker(TICKER_NAME)
        df = ticker.history(period="5d", interval="5m")
        return df
    except Exception as e:
        print(f"⚠️ Impossible de récupérer les données boursières : {e}")
        return pd.DataFrame()


def calculer_indicateurs(df):
    """Calcule manuellement les moyennes et indicateurs techniques de base."""
    if df.empty or len(df) < 20:
        return df
    df_analyse = df.copy()
    # Calcul d'une moyenne mobile exponentielle (EMA 50)
    df_analyse["EMA_50"] = df_analyse["Close"].ewm(span=50, adjust=False).mean()
    return df_analyse


def generer_signal_technique(df_indics):
    """Analyse la tendance pour dégager un signal d'achat ou d'attente."""
    if df_indics.empty or len(df_indics) < 2:
        return "ATTENTE"

    derniere_ligne = df_indics.iloc[-1]
    prix_actuel = derniere_ligne["Close"]
    ema_50 = derniere_ligne["EMA_50"]

    # Logique de trading : Achat si le prix repasse au-dessus de l'EMA 50
    if prix_actuel > ema_50:
        return "ACHAT"
    return "ATTENTE"


# ==========================================
# MOTEUR PRINCIPAL EN BOUCLE CONTINUE
# ==========================================
def executer_moteur_trading_permanent():
    """Gère la surveillance du marché en tâche de fond sur le Cloud."""
    print("🤖 Bot Cloud démarré...")

    # Notification immédiate de bon déploiement sur votre smartphone
    envoyer_alerte_telegram(
        "☁️ *Bot de Trading Déployé sur Render !*\nSurveillance automatique du CAC 40 initialisée."
    )

    position = None
    capital = 1000.0  # Capital fictif de départ pour le Paper Trading

    while True:
        try:
            maintenant = datetime.datetime.now()
            jour_semaine = maintenant.weekday()  # 0 = Lundi, 6 = Dimanche
            heure_actuelle = maintenant.time()

            # 1. Gestion des périodes de fermeture (Week-end et Nuit)
            est_weekend = jour_semaine in [5, 6]
            est_hors_horaires = (
                heure_actuelle < datetime.time(9, 0)
            ) or (heure_actuelle > datetime.time(17, 35))

            if est_weekend or est_hors_horaires:
                timestamp = maintenant.strftime("%H:%M:%S")
                print(f"[{timestamp}] Marché fermé. Veille active...")
                time.sleep(900)  # Sommeil de 15 minutes pendant la fermeture
                continue

            # 2. Cycle d'analyse en période d'ouverture (Toutes les 5 minutes)
            df_brut = recuperer_donnees_cac40()
            df_indics = calculer_indicateurs(df_brut)

            if not df_indics.empty:
                prix_actuel = float(df_indics.iloc[-1]["Close"])
                timestamp = maintenant.strftime("%H:%M:%S")
                print(
                    f"[{timestamp}] Scan en cours... Prix actuel : {prix_actuel:.2f} €"
                )

                signal = generer_signal_technique(df_indics)

                # Logique de prise de position (Achat)
                if signal == "ACHAT" and position is None:
                    position = {
                        "prix_entree": prix_actuel,
                        "tp": prix_actuel * 1.01,  # Objectif +1%
                        "sl": prix_actuel * 0.995,  # Stop de sécurité -0.5%
                        "parts": capital / prix_actuel,
                    }
                    envoyer_alerte_telegram(
                        f"🚀 *SIGNAL D'ACHAT VALIDÉ*\n"
                        f"📈 Prix d'entrée : {prix_actuel:.2f} €\n"
                        f"🎯 Objectif (TP) : {position['tp']:.2f} €\n"
                        f"🛡️ Stop Loss (SL) : {position['sl']:.2f} €"
                    )

                # Suivi de la position ouverte (Vente)
                elif position is not None:
                    if prix_actuel >= position["tp"]:
                        capital = position["parts"] * position["tp"]
                        envoyer_alerte_telegram(
                            f"🎯 *OBJECTIF ATTEINT (+1%) !*\nNouveau capital : {capital:.2f} €"
                        )
                        position = None
                    elif prix_actuel <= position["sl"]:
                        capital = position["parts"] * position["sl"]
                        envoyer_alerte_telegram(
                            f"🛡️ *STOP LOSS DÉCLENCHÉ.*\nCapital actuel : {capital:.2f} €"
                        )
                        position = None

            time.sleep(300)  # Pause de 5 minutes entre chaque scan boursier

        except Exception as e:
            print(f"⚠️ Une erreur est survenue dans la boucle : {e}")
            time.sleep(300)


if __name__ == "__main__":
    executer_moteur_trading_permanent()
  
