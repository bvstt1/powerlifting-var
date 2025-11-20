import numpy as np


def angulo(p1, p2, p3):
    """
    Calcula el ángulo formado en p2 por los puntos p1 - p2 - p3.
    p1, p2, p3 deben ser arrays de NumPy: [x, y]
    """
    v1 = p1 - p2  # Vector desde p2 hacia p1
    v2 = p3 - p2  # Vector desde p2 hacia p3

    # Producto punto
    dot = np.dot(v1, v2)

    # Magnitudes
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)

    # Evitar división por cero
    if mag1 == 0 or mag2 == 0:
        return 0.0

    # Coseno del ángulo
    cos_theta = dot / (mag1 * mag2)

    # Evitar problemas numéricos (clamp)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Convertir a grados
    return np.degrees(np.arccos(cos_theta))


# Ejmeplo Powerlifting

cadera  = np.array([300, 200])
rodilla = np.array([310, 300])
tobillo = np.array([315, 400])

print("Ángulo de rodilla:", angulo(cadera, rodilla, tobillo))

p1 = np.array([20, 50])
p2 = np.array([70, 60])
p3 = np.array([100, 100])

print(angulo(p1, p2, p3))
