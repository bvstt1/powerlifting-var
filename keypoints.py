import numpy as np

class Keypoint:
    def __init__(self, x, y, conf=1.0):
        self.x = float(x)
        self.y = float(y)
        self.conf = float(conf)

    def as_array(self):
        return np.array([self.x, self.y], dtype=float)

    def distance_to(self, other):
        return np.linalg.norm(self.as_array() - other.as_array())


class Skeleton:
    def __init__(self, keypoints_dict):
        self.kp = keypoints_dict

    def angle(self, p1_name, p2_name, p3_name):
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
