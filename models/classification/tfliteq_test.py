import csv
import cv2
import numpy as np
import tensorflow as tf

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="models/tsc2-tbo_quant.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get the input shape and use it for preprocessing frames
input_shape = input_details[0]['shape']
input_height, input_width = input_shape[1], input_shape[2]

# Initialize an empty dictionary
class_labels = {}

# Load the labels from the CSV file
with open('../../data/Classification/tsc2_labels.csv', mode='r') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)  # Skip the header row
    for row in csvreader:
        class_id, label_name = row
        class_labels[int(class_id)] = label_name

cap = cv2.VideoCapture(0)  # '0' is usually the default ID for the primary webcam.

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    processed_frame = cv2.resize(frame, (input_width, input_height))
    processed_frame = np.expand_dims(processed_frame, axis=0)  # Add batch dimension

    # Quantize the input data (since the input zero-point is 0, we don't need to subtract it)
    input_scale = input_details[0]['quantization'][0]
    processed_frame = np.round(processed_frame / 255.0 / input_scale).astype(np.int8)

    # Make prediction
    interpreter.set_tensor(input_details[0]['index'], processed_frame)
    interpreter.invoke()

    # De-quantize the output data
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_scale = output_details[0]['quantization'][0]
    output_zero_point = output_details[0]['quantization'][1]
    output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale

    # Now interpret the model's output
    top_prediction = np.argmax(output_data[0])
    confidence = np.max(output_data[0])
    label = class_labels[top_prediction]

    # Format the confidence to be a percentage between 0 and 100
    confidence = max(0, min(confidence, 1)) * 100  # Clamp the value to [0, 100] just in case

    # Display the label and confidence on the frame
    label_text = f"{label}: {confidence:.2f}%"
    cv2.putText(frame, label_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()