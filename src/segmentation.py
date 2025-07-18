import cv2
import numpy as np
from src.config import IMG_SIZE
import os
os.makedirs("fallbacks", exist_ok=True)


def segment_iris(img, disable_fallback_write=False, debug_path=None):
    assert img is not None and img.size > 0, "Błąd: wejściowy obraz jest pusty lub None."
    if len(img.shape) != 2:
        raise ValueError("Oczekiwano obrazu w skali szarości (2D ndarray).")

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

        # DEBUG: zapisz wykrycie jeśli ścieżka podana
        if debug_path:
            debug_img = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
            cv2.circle(debug_img, (x, y), r, (0, 255, 0), 2)
            cv2.imwrite(debug_path, debug_img)

        return cropped
    else:
        if not disable_fallback_write:
            fallback_index = len(os.listdir("fallbacks"))
            fallback_path = f"fallbacks/fallback_{fallback_index}.png"
            cv2.imwrite(fallback_path, img)
            print(f"[FALLBACK] Zapisano fallback nr {fallback_index}: {fallback_path}")
        return cv2.resize(img, IMG_SIZE)
