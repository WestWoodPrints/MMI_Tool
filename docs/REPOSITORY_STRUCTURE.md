# Repository-Struktur

## Verzeichnisbaum
```text
.
|- src/
|- data/
|  |- raw/MMI_Daten/
|  |- processed/results_out/
|- notebooks/
|- assets/images/
|- docs/
|- archive/
```

## Verantwortlichkeiten
- `src/`: produktiver Code (Messung, GUI, Export).
- `data/raw/`: unveraenderte Rohmessdaten.
- `data/processed/`: aggregierte oder modellierte Auswertungsergebnisse.
- `notebooks/`: explorative Analysen und Nachvollziehbarkeit der Auswertungsschritte.
- `assets/images/`: Visualisierungen fuer Berichte/README.
- `archive/`: Altlasten, nicht Teil des aktiven Daten-/Code-Flows.

## Konventionen
- Rohdaten-Dateien nicht ueberschreiben, nur neue Versionen hinzufuegen.
- Ergebnisdateien sprechend benennen (`screen_absdev.csv`, etc.).
- Skripte in `src/` in snake_case halten.
