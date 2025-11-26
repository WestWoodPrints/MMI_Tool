# README – Anleitung zur Einrichtung der Python-Umgebung für das Pose-Erkennungsprojekt

Diese README wurde mit ChatGPT erstellt und ist speziell dafür gedacht, euch – auch wenn ihr wenig Erfahrung mit Git, Python oder virtuellen Umgebungen habt – Schritt für Schritt durch die Einrichtung zu führen.

Das Ziel:  
Ihr sollt das Projekt **problemlos starten können**, ohne dass Vorwissen nötig ist.

---

## 1. Voraussetzungen

Bevor wir starten, benötigt ihr Folgendes:

- **Python 3.8 oder neuer**  
  (Mit `python3 --version` im Terminal könnt ihr eure Version prüfen.)
- **pip**  
  Wird normalerweise automatisch mit Python installiert.
- Einen beliebigen Editor (z. B. VS Code, aber nicht zwingend).

Mehr braucht ihr nicht.

---

## 2. Virtuelle Umgebung ("env") erstellen

Eine *virtuelle Umgebung* sorgt dafür, dass alle benötigten Python-Pakete nur für dieses Projekt installiert werden.  
Das verhindert Chaos auf eurem System.

Öffnet ein Terminal im Projektordner (dort, wo diese README liegt) und führt Folgendes aus:

```bash
python3 -m venv env
```

Danach entsteht ein neuer Ordner namens **env**.  
Darin befindet sich eine isolierte Python-Installation nur für dieses Projekt.

---

## 3. Virtuelle Umgebung aktivieren

Bevor ihr Pakete installiert oder das Projekt startet, muss die Umgebung aktiviert werden.

### Linux / macOS

```bash
source env/bin/activate
```

### Windows (PowerShell)

```powershell
.\env\Scripts\Activate
```

Wenn alles richtig funktioniert hat, steht am Anfang jeder Terminalzeile:

```
(env)
```

Das bedeutet: Die Umgebung ist aktiv.

---

## 4. Notwendige Python-Pakete installieren

Nun installieren wir die Bibliotheken, die das Programm benötigt:

```bash
pip install opencv-python mediapipe
```

Falls ihr Probleme mit OpenCV habt (kommt selten vor), könnt ihr stattdessen die schlankere Version installieren:

```bash
pip install opencv-python-headless mediapipe
```

---

## 5. Installation über eine requirements.txt

Falls ihr die Datei `requirements.txt` im Projekt habt (z. B. weil sie mit im Repository liegt), könnt ihr *alle* benötigten Pakete automatisch installieren.

Dazu einfach:

```bash
pip install -r requirements.txt
```

Wichtig:  
Die virtuelle Umgebung **muss vorher aktiviert sein** (siehe Schritt 3).

---

## 6. Projekt starten (Pose-Erkennung)

Um das Programm zu starten, führt Folgendes in der aktiven Umgebung aus:

```bash
python dummy.py
```

Danach öffnet sich ein Fenster mit eurem Kamerabild und einem Strichmännchen, das eure Pose verfolgt.

Beenden könnt ihr das Programm jederzeit mit der Taste:

```
q
```

---

## 7. Virtuelle Umgebung deaktivieren

Wenn ihr fertig seid, könnt ihr die Umgebung wieder verlassen:

```bash
deactivate
```

Damit wechselt ihr zurück in das normale System-Python.

---

## 8. Optional: requirements.txt selbst erzeugen

Falls ihr Pakete ergänzt habt und die aktuelle Paketliste speichern wollt:

```bash
pip freeze > requirements.txt
```

Diese Datei kann dann von anderen genutzt werden (siehe Schritt 5).

---

## 9. Optional: requirements.txt selbst erzeugen

EXE kann mit folgendem befehl erzeugt werden

```bash
pyinstaller --onefile --windowed `
--add-data "env\Lib\site-packages\mediapipe/modules/pose_detection/pose_detection.tflite;mediapipe/modules/pose_detection" `
--add-data "env\Lib\site-packages\mediapipe/modules/pose_landmark/pose_landmark_full.tflite;mediapipe/modules/pose_landmark" `
--add-data "env\Lib\site-packages\mediapipe/modules/pose_landmark/pose_landmark_cpu.binarypb;mediapipe/modules/pose_landmark" `
MMI_Tool.py
```

---
## 10. Hilfe und Fehlerbehebung (Basics)

### "python3: command not found"
→ Unter Windows heißt Python oft einfach `python`.

### Kamera funktioniert nicht
→ Prüft, ob sie in anderen Programmen blockiert ist.

### ModuleNotFoundError
→ Prüft, ob ihr die Umgebung aktiviert habt (steht `(env)` vorne?).

### Paket lässt sich nicht installieren
→ Internetverbindung prüfen  
→ Notfalls alternative OpenCV-Version nutzen (siehe Schritt 4)

---


