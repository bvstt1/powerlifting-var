import cv2
import mediapipe as mp
import numpy as np

class Keypoint:
    def __init__(self, x, y, conf=1.0):
        self.x = float(x)
        self.y = float(y)
        self.conf = float(conf)

    def as_array(self):
        """Devuelve el punto como un vector NumPy [x, y]."""
        return np.array([self.x, self.y], dtype=float)

    def distance_to(self, other):
        """Distancia euclidiana a otro Keypoint."""
        return np.linalg.norm(self.as_array() - other.as_array())

    def normalize(self, width, height):
        """Normaliza coordenadas a rango [0,1]."""
        self.x /= width
        self.y /= height
        return self

    def __repr__(self):
        return f"Keypoint(x={self.x:.3f}, y={self.y:.3f}, conf={self.conf:.2f})"


class Skeleton:
    def __init__(self, keypoints_dict):
        """
        keypoints_dict: diccionario con nombres y objetos Keypoint.
        Ejemplo:
        {
            'hip': Keypoint(...),
            'knee': Keypoint(...),
            'ankle': Keypoint(...),
        }
        """
        self.kp = keypoints_dict

    def angle(self, p1_name, p2_name, p3_name):
        """Calcula el ángulo en p2 formado por p1 - p2 - p3."""
        p1 = self.kp[p1_name].as_array()
        p2 = self.kp[p2_name].as_array()
        p3 = self.kp[p3_name].as_array()

        v1 = p1 - p2
        v2 = p3 - p2

        dot = np.dot(v1, v2)
        mag1 = np.linalg.norm(v1)
        mag2 = np.linalg.norm(v2)

        if mag1 == 0 or mag2 == 0:
            return 0.0

        cos_theta = dot / (mag1 * mag2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        return np.degrees(np.arccos(cos_theta))

    def __repr__(self):
        return f"Skeleton({list(self.kp.keys())})"

def to_keypoint(landmark, img_width, img_height):
    x = landmark.x * img_width
    y = landmark.y * img_height
    conf = landmark.visibility
    return Keypoint(x, y, conf)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
img = cv2.imread("sentadilla.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

result = pose.process(img_rgb)
landmarks = result.pose_landmarks.landmark

h, w, _ = img.shape
hip = to_keypoint(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], w, h)
knee = to_keypoint(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value], w, h)
ankle = to_keypoint(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value], w, h)

s = Skeleton({
    "hip": hip,
    "knee": knee,
    "ankle": ankle
})

angulo_rodilla = s.angle("hip", "knee", "ankle")
print("Ángulo de rodilla:", angulo_rodilla)


for lm in landmarks:
    x = int(lm.x * w)
    y = int(lm.y * h)
    cv2.circle(img, (x, y), 4, (0,255,0), -1)

if angulo_rodilla < 90:
    print("Sentadilla PROFUNDA (válida)")
else:
    print("Sentadilla NO profunda (posible roja)")

# puntos en pixeles
hip_pt   = (int(hip.x), int(hip.y))
knee_pt  = (int(knee.x), int(knee.y))
ankle_pt = (int(ankle.x), int(ankle.y))

# dibujar segmentos
cv2.line(img, hip_pt, knee_pt, (255, 0, 0), 2)
cv2.line(img, ankle_pt, knee_pt, (255, 0, 0), 2)

# escribir ángulo cerca de la rodilla
text = f"{angulo_rodilla:.1f} deg"
cv2.putText(img, text, (knee_pt[0] + 10, knee_pt[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

cv2.imwrite("sentadilla_mediapipe.png", img)


