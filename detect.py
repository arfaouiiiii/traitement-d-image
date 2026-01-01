import cv2
import os

# === Chemins vers les classifieurs Haar ===
base_path = "/usr/share/opencv4/haarcascades"

face_cascade_path = os.path.join(base_path, "haarcascade_frontalface_default.xml")
eye_cascade_path  = os.path.join(base_path, "haarcascade_eye.xml")
smile_cascade_path = os.path.join(base_path, "haarcascade_smile.xml")

# Charger les classifieurs
face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade  = cv2.CascadeClassifier(eye_cascade_path)
smile_cascade = cv2.CascadeClassifier(smile_cascade_path)

# Vérifications
if face_cascade.empty():
    print("Erreur : impossible de charger le classifieur de visage.")
    print("Chemin :", face_cascade_path)
    exit()

if eye_cascade.empty():
    print("Erreur : impossible de charger le classifieur d'yeux.")
    print("Chemin :", eye_cascade_path)
    exit()

if smile_cascade.empty():
    print("Erreur : impossible de charger le classifieur de sourire/bouche.")
    print("Chemin :", smile_cascade_path)
    exit()

# Ouvrir la caméra
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Impossible d'ouvrir la caméra")
    exit()

print("Appuie sur 'q' pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Impossible de lire l'image depuis la caméra")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # === Détection des visages ===
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:
        # Rectangle visage
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(frame, "Visage", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # ROI = région du visage
        roi_gray  = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # === Détection des yeux dans le visage ===
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(20, 20)
        )

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
            cv2.putText(roi_color, "Oeil", (ex, ey-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # === Détection du sourire / bouche dans le visage ===
        smiles = smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.7,
            minNeighbors=22,
            minSize=(25, 25)
        )

        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
            cv2.putText(roi_color, "Bouche / Sourire", (sx, sy+sh+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Afficher le nombre de visages
    cv2.putText(frame, f"Visages detectes : {len(faces)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Afficher la fenêtre
    cv2.imshow("Detection des composants du visage", frame)

    # Quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()