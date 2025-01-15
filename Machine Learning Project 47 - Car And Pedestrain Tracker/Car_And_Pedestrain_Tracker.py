import cv2  

# Our Image
img_file = r"C:\Users\jgaur\Downloads\Project\Car And Pedestrain Tracker\Car_Image.jpg"
video_file = cv2.VideoCapture(r"C:\Users\jgaur\Downloads\Project\Car And Pedestrain Tracker\pedestrain.mp4")

# Our pre-trained car and pedestrain Classifier
car_traker_file= r"C:\Users\jgaur\Downloads\Project\Car And Pedestrain Tracker\cars.xml"
pedestrain_tracker_file = r"C:\Users\jgaur\Downloads\Project\Car And Pedestrain Tracker\haarcascade_fullbody.xml"

# create car and pedestrain classifier
car_tracker = cv2.CascadeClassifier(car_traker_file)
pedestrain_tracker = cv2.CascadeClassifier(pedestrain_tracker_file)

# Run forever until car stops or something
while True:

    # Read the current frame
    (read_sucessful, frame) = video_file.read()

    if read_sucessful:
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # detect cars 
    cars =  car_tracker.detectMultiScale(grayscale_frame)
    pedestrains = pedestrain_tracker.detectMultiScale(grayscale_frame)

    # Draw rectangles around the cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    # Draw rectangles around the cars
    for (x, y, w, h) in pedestrains:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)    


    cv2.imshow("Car Detector", frame)
    key = cv2.waitKey(1)

    # Stop if Q key is pressed
    if key == 81 or key == 113:
        break

# Release the VideoCapture Object
video_file.release()

# create opencv image
img = cv2.imread(img_file)

# create care classifire
car_tracker = cv2.CascadeClassifier(classifire_file)

# convert to grayscale (needed for haar cascade)
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect cars
car = car_tracker.detectMultiScale(black_n_white)
# print(car)

# # Draw rectangles around the cars
for (x, y, w, h) in car:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

# Display the image with the faces spotted
cv2.imshow("Car Detector", img)

# Don't autoclose (Wait here in the code and litsten for a key press)
cv2.waitKey()

print("Code Completed") 