import cv2
import joblib
import numpy as np
from sift_features import extract_sift
from build_histogram import build_histogram

# Load trained model and vocabulary
model = joblib.load("model.pkl")
kmeans = joblib.load("vocabulary.pkl")

# Open camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not accessible")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame not received")
        break

    # Define ROI
    x1, y1 = 300, 100
    x2, y2 = 600, 400
    roi = frame[y1:y2, x1:x2]

    # Convert ROI to grayscale and enhance
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.equalizeHist(roi_gray)

    cv2.imshow("ROI", roi_gray)

    # Extract SIFT descriptors
    desc = extract_sift(roi)

    if desc is not None:
        hist = build_histogram(desc, kmeans)

        pred = model.predict([hist])[0]

        # Multi-class safe confidence
        scores = model.decision_function([hist])

        if len(scores.shape) > 1:
            confidence = np.max(scores)
        else:
            confidence = scores[0]

        confidence = round(float(abs(confidence)), 2)

        text = f"{pred} ({confidence})"
    else:
        text = "No Gesture"

    # Display prediction
    cv2.putText(frame, text, (50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,255,0), 2)

    # Draw rectangle
    cv2.rectangle(frame, (x1,y1), (x2,y2),
                  (0,255,0), 2)

    cv2.imshow("Gesture Recognition", frame)

    # Quit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()