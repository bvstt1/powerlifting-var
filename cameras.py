import cv2

def scan_cameras(max_index=10):
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            print(f"CAMARA ENCONTRADA: {i}")
            cap.release()
        else:
            print(f"NO disponible: {i}")

scan_cameras()