# README – Einrichtung einer Python-Umgebung für Pose-Erkennung

README ist mit ChatGPT erstellt als Anleitung für euch! 
---

## 1. Voraussetzungen

- Python 3.8 oder neuer  
- pip (meist automatisch in Python enthalten)

---

## 2. Virtuelle Umgebung "env" erstellen

Öffne ein Terminal im Projektordner und führe folgenden Befehl aus:

```bash
python3 -m venv env
```

Dadurch entsteht ein Ordner `env/`, der alle Projektabhängigkeiten getrennt vom System-Python verwaltet.

---

## 3. Virtuelle Umgebung aktivieren

### Linux / macOS

```bash
source env/bin/activate
```

### Windows (PowerShell)

```powershell
.\env\Scripts\Activate
```

Wenn die Umgebung aktiv ist, wird vorne im Terminal `(env)` angezeigt.

---

## 4. Benötigte Pakete installieren

Installiere die notwendigen Python-Pakete:

```bash
pip install opencv-python mediapipe
```

Falls du eine schlankere Version von OpenCV bevorzugst, kannst du alternativ folgendes installieren:

```bash
pip install opencv-python-headless mediapipe
```

---

## 5. Programm starten

Führe das Pose-Erkennungs-Skript aus:

```bash
python pose_stickman.py
```

Das Fenster kann mit `q` geschlossen werden.

---

## 6. Virtuelle Umgebung deaktivieren

Nach der Arbeit kannst du die Umgebung mit folgendem Befehl verlassen:

```bash
deactivate
```

---

## 7. Optional: Abhängigkeiten speichern

Falls du eine `requirements.txt` erzeugen möchtest:

```bash
pip freeze > requirements.txt
```

---
