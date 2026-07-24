"""Univers CAC 40 et regroupement sectoriel.

Composition vérifiée à partir des publications Euronext du 30 juin 2026.
Les symboles sont ceux utilisés par Yahoo Finance.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Instrument:
    ticker: str
    name: str
    sector: str


CAC40: tuple[Instrument, ...] = (
    Instrument("AC.PA", "Accor", "Consommation"),
    Instrument("AI.PA", "Air Liquide", "Matériaux"),
    Instrument("AIR.PA", "Airbus", "Industrie"),
    Instrument("ALO.PA", "Alstom", "Industrie"),
    Instrument("MT.AS", "ArcelorMittal", "Matériaux"),
    Instrument("CS.PA", "AXA", "Finance"),
    Instrument("BNP.PA", "BNP Paribas", "Finance"),
    Instrument("EN.PA", "Bouygues", "Industrie"),
    Instrument("CAP.PA", "Capgemini", "Technologie"),
    Instrument("CA.PA", "Carrefour", "Consommation de base"),
    Instrument("ACA.PA", "Crédit Agricole", "Finance"),
    Instrument("BN.PA", "Danone", "Consommation de base"),
    Instrument("DSY.PA", "Dassault Systèmes", "Technologie"),
    Instrument("FGR.PA", "Eiffage", "Industrie"),
    Instrument("ENGI.PA", "Engie", "Services collectifs"),
    Instrument("EL.PA", "EssilorLuxottica", "Santé"),
    Instrument("ENX.PA", "Euronext", "Finance"),
    Instrument("ERF.PA", "Eurofins Scientific", "Santé"),
    Instrument("RMS.PA", "Hermès", "Consommation"),
    Instrument("KER.PA", "Kering", "Consommation"),
    Instrument("LR.PA", "Legrand", "Industrie"),
    Instrument("OR.PA", "L'Oréal", "Consommation"),
    Instrument("MC.PA", "LVMH", "Consommation"),
    Instrument("ML.PA", "Michelin", "Consommation"),
    Instrument("ORA.PA", "Orange", "Télécommunications"),
    Instrument("RI.PA", "Pernod Ricard", "Consommation de base"),
    Instrument("PUB.PA", "Publicis Groupe", "Communication"),
    Instrument("RNO.PA", "Renault", "Consommation"),
    Instrument("SAF.PA", "Safran", "Industrie"),
    Instrument("SGO.PA", "Saint-Gobain", "Industrie"),
    Instrument("SAN.PA", "Sanofi", "Santé"),
    Instrument("SU.PA", "Schneider Electric", "Industrie"),
    Instrument("GLE.PA", "Société Générale", "Finance"),
    Instrument("STLAP.PA", "Stellantis", "Consommation"),
    Instrument("STMPA.PA", "STMicroelectronics", "Technologie"),
    Instrument("HO.PA", "Thales", "Industrie"),
    Instrument("TTE.PA", "TotalEnergies", "Énergie"),
    Instrument("URW.PA", "Unibail-Rodamco-Westfield", "Immobilier"),
    Instrument("VIE.PA", "Veolia", "Services collectifs"),
    Instrument("DG.PA", "Vinci", "Industrie"),
)

BY_TICKER = {instrument.ticker: instrument for instrument in CAC40}
