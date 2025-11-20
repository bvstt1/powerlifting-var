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


hip    = Keypoint(300, 200)
knee   = Keypoint(310, 300)
ankle  = Keypoint(315, 400)


s = Skeleton({
    "hip": hip,
    "knee": knee,
    "ankle": ankle
})

p1 = Keypoint(324, 406, 0.96)

print(p1)

print("Ángulo de rodilla:", s.angle("hip", "knee", "ankle"))