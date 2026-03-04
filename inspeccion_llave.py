import cv2
import numpy as np

# 1. Cargar la imagen de la llave de ajuste
# Asegúrate de que el archivo 'RLLAVE.jpg' esté en la misma carpeta
imagen = cv2.imread("RLLAVE.jpg.jpeg")

if imagen is None:
    print("Error: No se pudo cargar la imagen. Verifica el nombre y la ruta.")
    exit()

# 2. Preprocesamiento: Convertir a escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# 3. Reducir ruido mediante desenfoque Gaussiano
# Ayuda a que la segmentación sea más limpia y no detecte granos de ruido
suave = cv2.GaussianBlur(gris, (5, 5), 0)

# 4. Segmentación por umbralización óptima (Método de Otsu)
# Separa automáticamente la llave del fondo creando una imagen de blanco y negro
ret, binaria = cv2.threshold(
    suave,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# 5. Limpieza morfológica (Cierre)
# Rellena pequeños huecos internos y suaviza los bordes de la silueta
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
binaria_limpia = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)

# 6. Encontrar contornos de los objetos detectados
# Buscamos solo el contorno externo de la pieza
contornos, _ = cv2.findContours(
    binaria_limpia,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Validamos que se haya detectado al menos un objeto
if len(contornos) > 0:
    # Seleccionamos el contorno con el área más grande (la llave principal)
    contorno_llave = max(contornos, key=cv2.contourArea)

    # 7. Calcular el área real detectada en píxeles
    area_detectada = cv2.contourArea(contorno_llave)

    # Definir el área nominal de referencia (estándar de calidad)
    area_nominal = 25000 

    # 8. Evaluación del criterio de calidad (Umbral del 95%)
    porcentaje_area = (area_detectada / area_nominal) * 100

    if porcentaje_area >= 95:
        estado = "APTA"
        color = (0, 255, 0)  # Verde para piezas aprobadas
    else:
        estado = "NO APTA"
        color = (0, 0, 255)  # Rojo para piezas rechazadas

    # 9. Dibujar resultados visuales sobre la imagen original
    # Dibujamos el perímetro de la llave
    cv2.drawContours(imagen, [contorno_llave], -1, color, 2)

    # Escribimos el diagnóstico (Estado y Porcentaje de área)
    cv2.putText(
        imagen,
        f"Estado: {estado}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.putText(
        imagen,
        f"Area: {int(porcentaje_area)}%",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

# 10. Desplegar los resultados en pantalla
cv2.imshow("Resultado inspeccion de calidad", imagen)
cv2.imshow("Segmentacion binaria (Mascara)", binaria_limpia)

# El programa espera a que presiones cualquier tecla para cerrarse
cv2.waitKey(0)
cv2.destroyAllWindows()