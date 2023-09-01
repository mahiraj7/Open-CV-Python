import cv2

# Load the pre-trained Haarcascades classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load an image from file
image_path = 'path/to/your/image.jpg'
image = cv2.imread(image_path)

# Convert the image to grayscale (required for the face detection algorithm)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)

# Display the image with detected faces
cv2.imshow('Face Detection', image)

# Wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()