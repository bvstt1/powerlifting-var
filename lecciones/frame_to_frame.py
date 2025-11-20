import cv2
import tensorflow as tf

from keypoints import Keypoint, Skeleton   # si los tienes en archivo separado

# ==== Cargar MoveNet Thunder ====
model = tf.saved_model.load("../models/movenet_thunder")
movenet = model.signatures['serving_default']

def detect_keypoints_movenet(image_bgr):
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    input_size = 256
    img_resized = tf.image.resize(img_rgb, (input_size, input_size))
    img_resized = tf.cast(img_resized, dtype=tf.int32)
    input_tensor = tf.expand_dims(img_resized, axis=0)

    outputs = movenet(input_tensor)
    keypoints = outputs['output_0'].numpy()[0, 0, :, :]
    return keypoints

def kp_to_keypoint(kp_row, img_w, img_h):
    y_norm, x_norm, score = kp_row
    x = x_norm * img_w
    y = y_norm * img_h
    return Keypoint(x, y, score)

# ==== LÓGICA EN TIEMPO REAL ====

cap = cv2.VideoCapture(0)  # webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    kps = detect_keypoints_movenet(frame)

    # cadera, rodilla, tobillo izquierdos
    hip    = kp_to_keypoint(kps[11], frame.shape[1], frame.shape[0])
    knee   = kp_to_keypoint(kps[13], frame.shape[1], frame.shape[0])
    ankle  = kp_to_keypoint(kps[15], frame.shape[1], frame.shape[0])

    s = Skeleton({"hip": hip, "knee": knee, "ankle": ankle})
    angle = s.angle("hip", "knee", "ankle")

    # Dibujar keypoints
    for (y_norm, x_norm, score) in kps:
        x = int(x_norm * frame.shape[1])
        y = int(y_norm * frame.shape[0])
        cv2.circle(frame, (x, y), 4, (0,255,0), -1)

    # Escribir ángulo en pantalla
    cv2.putText(frame, f"Rodilla: {angle:.1f}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("MoveNet Thunder - Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
