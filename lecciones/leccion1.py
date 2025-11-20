# 1.1 Listas

def distancia(a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5

x = [1,2]
y = [4,3]
print(distancia(x,y))

def normalizar_punto(x, y, width=640, height=480):
    return x / width, y / height

#x = 10
#y = 5
#print(normalizar_punto(x,y))

# 1.2 List Comprehensions y lambdas
puntos = [(10, 20), (30, 40), (50, 60)]
xs = [x for x, y in puntos]   # [10, 30, 50]
ys = [y for x, y in puntos]   # [20, 40, 60]

print(xs, ys)

dist = lambda a, b: ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5
print(dist(x, y))

# 1.3 Clases y Objetos – la base antes de Keypoint


class NombreDeClase:
    def __init__(self, parametros):
        # se ejecuta al crear el objeto
        self.algo = parametros

    def metodo(self, otros_parametros):
        # hace algo con self
        ...
# 1.4 Primera versión de la clase Keypoint

class Keypoint:
    def __init__(self, x, y, conf=1.0):
        self.x = x
        self.y = y
        self.conf = conf

    def as_array(self):
        """Devuelve las coordenadas como un array de NumPy."""
        import numpy as np
        return np.array([self.x, self.y], dtype=float)

    def normalize(self, width, height):
        """Normaliza las coordenadas a [0, 1]. Modifica el objeto y lo devuelve."""
        self.x = self.x / width
        self.y = self.y / height
        return self

print("\n")
kp = Keypoint(320, 240, conf=0.95)
kp.normalize(640, 480)
print(kp.x, kp.y)      # 0.5, 0.5
print(kp.as_array())   # [0.5 0.5]
