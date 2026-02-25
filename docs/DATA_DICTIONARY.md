# Data Dictionary

## Rohdaten: `data/raw/MMI_Daten/Proband_*.csv`
Trennzeichen: `;`

### Spalten
- `Frame`: fortlaufender Frame-Index innerhalb einer Messung.
- `Winkel`: gemessener Kniewinkel in Grad (gerundet, ganzzahlig exportiert).

### Beispiel
```csv
Frame;Winkel
0;93
1;97
2;106
```

## Ergebnisdaten: `data/processed/results_out/*.csv`

### `screen_absdev.csv`
- `var`: Merkmal/Variable.
- `type`: Variablentyp (z. B. `binary`).
- `effect`: geschaetzter Effekt.
- `ci_lo`: untere Grenze des Konfidenzintervalls.
- `ci_hi`: obere Grenze des Konfidenzintervalls.
- `n1`: Anzahl Beobachtungen in Gruppe 1.
- `n0`: Anzahl Beobachtungen in Gruppe 0.
- `abs_effect`: Betrag des Effekts.

### `screen_residual.csv`
Gleiche Spaltenbedeutung wie `screen_absdev.csv` fuer Residual-basierte Kennzahlen.

### `logreg_inwindow_coefs.csv`
- `feature`: Feature-Name.
- `coef`: logistischer Regressionskoeffizient.

### `ridge_absdev_coefs.csv`
- `feature`: Feature-Name.
- `coef`: Ridge-Regressionskoeffizient.

## Datenqualitaet und Transparenz
- Rohdaten bleiben getrennt von abgeleiteten Ergebnisdateien.
- Dokumentierte Spalten sind Mindeststandard fuer zugaengliche Datennutzung.
- Bei neuen Dateien dieses Dokument erweitern.
