import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

# Cargar el modelo preentrenado de detección de objetos desde TensorFlow Hub
model_url = "https://tfhub.dev/tensorflow/efficientdet/d0/1"  # Modelo preentrenado EfficientDet
detector = hub.load(model_url)

# Función para realizar detección de objetos
def run_inference_for_single_image(model, image):
    # Convertir imagen a tensor y agregar batch dimension
    image_np = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis,...]  # Añadir una dimensión para el batch
    
    # Ejecutar la detección
    output_dict = model(input_tensor)
    
    # Filtrar las detecciones para obtener solo las más relevantes (con probabilidad > 0.5)
    output_dict = {key:value.numpy() for key,value in output_dict.items()}
    return output_dict

# Abrir la cámara web
cap = cv2.VideoCapture(0)  # '0' es el índice de la cámara web predeterminada

if not cap.isOpened():
    print("No se pudo acceder a la cámara.")
    exit()

while True:
    # Capturar el fotograma actual de la cámara
    ret, frame = cap.read()
    
    if not ret:
        print("No se pudo leer el fotograma de la cámara.")
        break

    # Convertir la imagen BGR a RGB para la detección de objetos
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Ejecutar la detección de objetos
    output_dict = run_inference_for_single_image(detector, image_rgb)

    # Extraer la información de las detecciones
    boxes = output_dict['detection_boxes']
    class_ids = output_dict['detection_classes']
    scores = output_dict['detection_scores']

    # Filtrar detecciones con una probabilidad mayor al umbral (0.5)
    threshold = 0.5
    boxes_filtered = boxes[scores > threshold]
    class_ids_filtered = class_ids[scores > threshold]
    scores_filtered = scores[scores > threshold]

    # Dibujar los resultados (cajas y texto) sobre el fotograma
    for box, class_id, score in zip(boxes_filtered, class_ids_filtered, scores_filtered):
        ymin, xmin, ymax, xmax = box
        xmin = int(xmin * frame.shape[1])
        xmax = int(xmax * frame.shape[1])
        ymin = int(ymin * frame.shape[0])
        ymax = int(ymax * frame.shape[0])

        # Dibujar un rectángulo alrededor del objeto
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.putText(frame, f"ID: {int(class_id)}, Score: {score:.2f}", 
                    (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Mostrar el fotograma con las detecciones
    cv2.imshow("Detección de Objetos - Cámara Web", frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
