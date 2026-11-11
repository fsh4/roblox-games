import sys
import numpy as np, cv2
out = sys.argv[1]
H, W = 600, 800
yy, xx = np.mgrid[0:H, 0:W]
def backdrop():
    g = 120 + 25 * (yy / H)
    return np.dstack([g, g, g]).astype(np.float32)
def shade(mask, base):
    d = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    d = d / max(d.max(), 1)
    light = 0.55 + 0.45 * np.sqrt(d)
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0) ; gy = cv2.Sobel(d, cv2.CV_32F, 0, 1)
    light += 0.25 * np.clip(-gx - gy, -1, 1)
    return np.clip(np.asarray(base, np.float32)[None, None] * light[..., None], 0, 255)
def compose(mask, base, name, shadow=True):
    img = backdrop()
    if shadow:
        sh = cv2.GaussianBlur(np.roll(mask.astype(np.float32), (14, 30), (0, 1)), (0, 0), 12)
        img *= (1 - 0.35 * sh)[..., None]
    col = shade(mask, base)
    img = np.where(mask[..., None], col, img)
    noise = np.random.default_rng(1).normal(0, 2.5, img.shape)
    cv2.imwrite(f"{out}/{name}", cv2.cvtColor(np.clip(img + noise, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
m = np.zeros((H, W), np.uint8)
cv2.circle(m, (400, 300), 200, 1, -1); cv2.circle(m, (400, 300), 85, 0, -1)
compose(m > 0, (230, 160, 90), "donut.jpg")
m = np.zeros((H, W), np.uint8)
cv2.rectangle(m, (260, 150), (500, 520), 1, -1); cv2.ellipse(m, (500, 330), (110, 120), 0, -90, 90, 1, 38)
compose(m > 0, (200, 70, 60), "mug.jpg")
m = np.zeros((H, W), np.uint8)
cv2.ellipse(m, (400, 210), (230, 130), 0, 180, 360, 1, -1); cv2.rectangle(m, (170, 205), (630, 225), 1, -1); cv2.rectangle(m, (350, 220), (450, 520), 1, -1)
compose(m > 0, (200, 60, 50), "mushroom.jpg")
m = np.zeros((H, W), np.uint8)
cv2.circle(m, (400, 170), 95, 1, -1); cv2.ellipse(m, (400, 400), (150, 170), 0, 0, 360, 1, -1)
cv2.ellipse(m, (250, 360), (40, 110), 25, 0, 360, 1, -1); cv2.ellipse(m, (550, 360), (40, 110), -25, 0, 360, 1, -1)
compose(m > 0, (120, 170, 220), "character.jpg")
m = np.zeros((H, W), np.uint8)
pts = np.array([[390, 40], [410, 40], [420, 420], [380, 420]]); cv2.fillPoly(m, [pts], 1); cv2.rectangle(m, (320, 420), (480, 445), 1, -1); cv2.rectangle(m, (388, 445), (412, 560), 1, -1)
compose(m > 0, (190, 195, 205), "sword.jpg")
print("ok")
