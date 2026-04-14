import cv2
import time
import mediapipe as mp
import numpy as np

# --- MediaPipe Ayarları ---
mpHands = mp.solutions.hands
# Tracking'i en yüksek hassasiyete çekiyoruz (Model 1 = Daha doğru konumlandırma)
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=1, 
    model_complexity=1, 
    min_detection_confidence=0.6, 
    min_tracking_confidence=0.6
)
mpDraw = mp.solutions.drawing_utils
##selam
frameWidth = 1280
frameHeight = 720
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

draw_points = []
pTime = 0

# --- HASSASİYET VE DOĞRULUK AYARLARI ---
# smooth_factor 0.65: Gecikmeyi (delay) yok eder, anlık takip sağlar.
smooth_factor = 0.65  
px, py = 0, 0 
lost_frames = 0
stability_threshold = 5 # Çizginin kopmaması için kısa süreli tolerans

while True:
    success, img = cap.read()
    if not success:
        break
        
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    drawing_active = False
    cx, cy = 0, 0

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Eli sadece ince çizgilerle göster (Görüntü kirliliğini önle)
            mpDraw.draw_landmarks(
                img, handLms, mpHands.HAND_CONNECTIONS,
                mpDraw.DrawingSpec(color=(255, 0, 255), thickness=1, circle_radius=1),
                mpDraw.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
            )

            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                # Landmark koordinatlarını doğrudan kullan (Kayma/Shift olmaması için)
                cx_raw, cy_raw = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx_raw, cy_raw])

            if len(lmList) >= 21:
                # İşaret parmağı ucu (8) ve orta parmak ucu (12) kontrolü
                # Daha doğru bir "Pointing" tespiti için eklem farkları
                index_tip = lmList[8][2]
                index_pip = lmList[6][2]
                middle_tip = lmList[12][2]
                middle_pip = lmList[10][2]

                if index_tip < index_pip and middle_tip > middle_pip:
                    drawing_active = True
                    lost_frames = 0
                    cx, cy = lmList[8][1], lmList[8][2]
                else:
                    drawing_active = False

    # --- ANLIK TAKİP VE HIZLI YUMUŞATMA ---
    if drawing_active:
        if px == 0 and py == 0:
            px, py = cx, cy
        else:
            # 0.65 katsayısı ile parmağınızı çok daha yakından ve hızlı takip eder
            px = int(px + (cx - px) * smooth_factor)
            py = int(py + (cy - py) * smooth_factor)
        
        draw_points.append((px, py))
        # Kalem ucu parmağın tam üzerinde olmalı
        cv2.circle(img, (px, py), 5, (255, 255, 255), cv2.FILLED)
    else:
        lost_frames += 1
        if lost_frames > stability_threshold:
            if len(draw_points) > 0 and draw_points[-1] is not None:
                draw_points.append(None)
            px, py = 0, 0

    # --- NET VE KESKİN ÇİZGİLER ---
    for i in range(1, len(draw_points)):
        if draw_points[i - 1] is None or draw_points[i] is None:
            continue
        # Çizgiyi incelttik ve beyaz çekirdeği belirginleştirdik (Daha isabetli görünüm)
        cv2.line(img, draw_points[i - 1], draw_points[i], (255, 255, 0), 4)
        cv2.line(img, draw_points[i - 1], draw_points[i], (255, 255, 255), 1)

    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime
    
    cv2.putText(img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.putText(img, "C: Temizle | Q: Cikis", (10, 70), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

    cv2.imshow("Hassas Hava Cizim", img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        draw_points.clear()

cap.release()
cv2.destroyAllWindows()