# MMI Tool

Dieses Repository stellt Mess- und Auswertungsdaten zur Kniewinkel-Erfassung strukturiert und nachvollziehbar bereit.

## Ziel
- Transparente Bereitstellung von Rohdaten und Auswertungsergebnissen.
- Reproduzierbare Ausführung der Mess-Tools.
- Klar getrennte Bereiche fuer Quellcode, Daten, Notebooks und Medien.

## Repository-Struktur
- `src/`: Python-Quellcode der Anwendungen.
- `data/raw/MMI_Daten/`: Rohdaten pro Proband (`Frame;Winkel`).
- `data/processed/results_out/`: aufbereitete Ergebnis-CSV-Dateien.
- `notebooks/`: explorative Auswertungen (`.ipynb`) und begleitende Tabellen.
- `assets/images/`: erzeugte Diagramme und Abbildungen.
- `archive/`: Alt-/Ablagebereich (nicht fuer den normalen Workflow).
- `docs/`: Daten- und Strukturdokumentation.

Details: siehe [docs/REPOSITORY_STRUCTURE.md](docs/REPOSITORY_STRUCTURE.md) und [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md).

## Schnellstart
### 1. Umgebung erstellen
```powershell
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
```

### 2. Hauptanwendung starten
```powershell
python src/mmi_tool_video.py
```

Alternative (einfachere Variante):
```powershell
python src/mmi_tool.py
```

## Daten-Transparenz
- Anzahl Rohdaten-Dateien: 25 (`data/raw/MMI_Daten/Proband_*.csv`).
- Rohdatenformat: Semikolon-separiert mit `Frame` und `Winkel`.
- Ergebnisdateien liegen getrennt als CSV in `data/processed/results_out/`.

## Reproduzierbarkeit
1. Python-Umgebung wie oben erstellen.
2. Anwendung starten und Winkelverlauf aufnehmen.
3. CSV-Export der App fuer neue Datensaetze verwenden.
4. Notebooks in `notebooks/` fuer Auswertung und Visualisierung nutzen.

## Hinweise fuer oeffentliche Veroeffentlichung
- Keine lokalen Build-Artefakte committen (`env/`, `build/`, `dist/`, `.exe`).
- Bei neuen Daten die Dokumentation in `docs/DATA_DICTIONARY.md` aktualisieren.
- Fuer Releases bevorzugt GitHub Releases statt Binardateien im Repository-Root nutzen.
