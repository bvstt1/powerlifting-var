import cv2
import time
import tensorflow as tf
from keypoints import Keypoint, Skeleton

# ===========================
# 1. CARGAR MOVENET THUNDER
# ===========================
print("Cargando MoveNet Thunder...")
model = tf.saved_model.load("models/movenet_thunder")
movenet = model.signatures['serving_default']


def detect_keypoints_movenet(frame_bgr):
    """
    Recibe frame BGR y retorna keypoints [17, 3] de MoveNet Thunder.
    Thunder pide input 256x256 INT32.
    """
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (256, 256))
    img_resized = tf.cast(img_resized, tf.int32)
    input_tensor = tf.expand_dims(img_resized, 0)

    outputs = movenet(input_tensor)
    kps = outputs['output_0'].numpy()[0, 0, :, :]  # [17, 3]

    return kps


def kp_to_point(kp, W, H):
    """Convierte keypoint normalizado a píxeles."""
    y, x, score = kp
    return int(x * W), int(y * H), score


# Conexiones del esqueleto
SKELETON_EDGES = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6),
    (11, 12),
    (5, 11), (6, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

# ====================================
# 3. INICIAR CAPTURA DE WEBCAM
# ====================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

print("Webcam iniciada correctamente")

prev_time = time.time()

# ====================================
# 4. LOOP DE DEMOSTRACIÓN
# ====================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al leer frame")
        break

    H, W, _ = frame.shape

    # Optimización para Thunder
    small = cv2.resize(frame, (320, 320))
    kps = detect_keypoints_movenet(small)

    scale_x = W / 320
    scale_y = H / 320

    points_px = []
    for kp in kps:
        y_norm, x_norm, score = kp
        x = int((x_norm * 320) * scale_x)
        y = int((y_norm * 320) * scale_y)
        points_px.append((x, y, score))

    # EXTRA: calcular ángulo de rodilla solo como demo
    hip = Keypoint(*points_px[11][:2])
    knee = Keypoint(*points_px[13][:2])
    ankle = Keypoint(*points_px[15][:2])
    skeleton = Skeleton({"hip": hip, "knee": knee, "ankle": ankle})
    angle = skeleton.angle("hip", "knee", "ankle")

    # Dibujar puntos
    for x, y, score in points_px:
        if score > 0.3:
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    # Dibujar líneas
    for a, b in SKELETON_EDGES:
        x1, y1, s1 = points_px[a]
        x2, y2, s2 = points_px[b]

        if s1 > 0.3 and s2 > 0.3:
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Mostrar ángulo y FPS
    now = time.time()
    fps = 1 / (now - prev_time)
    prev_time = now

    cv2.putText(frame, f"Rodilla: {angle:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("MoveNet Thunder Demo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
