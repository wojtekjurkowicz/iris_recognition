import cv2
import numpy as np
from config import IMG_SIZE
import os
os.makedirs("fallbacks", exist_ok=True)

fallback_count = 0


def get_fallback_count():
    global fallback_count
    return fallback_count


def segment_iris(img):
    img_small = cv2.resize(img, (80, 80))
    img_blur = cv2.medianBlur(img_small, 5)
    circles = cv2.HoughCircles(
        img_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=30,
        maxRadius=80
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0][0]

        # Przeskaluj współrzędne z powrotem do oryginalnego rozmiaru obrazu
        scale_x = img.shape[1] / 80
        scale_y = img.shape[0] / 80
        x = int(x * scale_x)
        y = int(y * scale_y)
        r = int(r * (scale_x + scale_y) / 2)

        # Zabezpieczenie przed wyjściem poza obraz
        h, w = img.shape
        y1 = max(y - r, 0)
        y2 = min(y + r, h)
        x1 = max(x - r, 0)
        x2 = min(x + r, w)

        mask = np.zeros_like(img)
        cv2.circle(mask, (x, y), r, 255, -1)

        result = cv2.bitwise_and(img, mask)
        cropped = result[y1:y2, x1:x2]

        # Jeszcze jedna kontrola: upewnij się, że coś zostało wycięte
        if cropped.size == 0:
            return cv2.resize(img, IMG_SIZE)

        cropped = cv2.resize(cropped, IMG_SIZE)
        return cropped
    else:
        global fallback_count
        fallback_count += 1
        cv2.imwrite(f"fallbacks/fallback_{fallback_count}.png", img)
        return cv2.resize(img, IMG_SIZE)
