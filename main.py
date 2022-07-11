import cv2
import dlib
import numpy as np

# Function for converting the facial landmarks to a np array
def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords


def resize_image(image_name, image, procent):
    width = int(image.shape[1] * procent)
    height = int(image.shape[0] * procent)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow(image_name, image)



def eye_mask(mask, side):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask


def pupil_mask(mask, frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret2, pupil = cv2.threshold(frame, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pupil = cv2.bitwise_not(pupil, mask=mask)
    pupil = cv2.dilate(pupil, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    pupil = cv2.morphologyEx(pupil, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    return pupil


def eye_pos(side, pupils, frame):
    try:
        # finding the pupil position
        cnt, _ = cv2.findContours(pupils, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #cv2.drawContours(frame, cnt, -1, (255, 10, 10), 1)
        for cnt in cnt:
            x,y,w,h = cv2.boundingRect(cnt)
            pupil_cx, pupil_cy = int(x+w/2), int(y+h/2)
            cv2.circle(frame, (pupil_cx, pupil_cy), radius=0, color=(0, 0, 255), thickness=-1)

        points = [shape[i] for i in side]
        points = np.array(points, dtype=np.int32)
        x,y,w,h = cv2.boundingRect(points)
        eye_cx, eye_cy = int(x + w / 2), int(y + h / 2)
        cv2.circle(frame, (eye_cx, eye_cy), radius=0, color=(255, 0, 0), thickness=-1)
        return (pupil_cx-eye_cx, pupil_cy-eye_cy)
    except:
         print('hey')
         return 0

# for trackbars
def nothing(x):
    pass


# Import learned facial recognition
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68.dat')

# These are the numbers assigned to the left and right eye
left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

cap = cv2.VideoCapture(0)
ret, img = cap.read()
thresh = img.copy()

cv2.namedWindow('result')
kernel = np.ones((9, 9), np.uint8)


cv2.createTrackbar('threshold', 'image', 0, 255, nothing)

while (True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the facial points on current frame
    rects = detector(gray, 1)
    # skip if there was nothing
    if len(rects) == 0:
        continue

    for rect in rects:
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        # Colouring in the eyes for thresholding
        left_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        left_mask = eye_mask(left_mask, left)
        left_mask = cv2.dilate(left_mask, kernel, 5)
        right_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        right_mask = eye_mask(right_mask, right)
        right_mask = cv2.dilate(right_mask, kernel, 5)
        mask = cv2.add(right_mask, left_mask)

        eyes = cv2.bitwise_and(img, img, mask=mask)
        cv2.imshow('before', eyes)

        left_pupil = pupil_mask(left_mask, eyes)
        right_pupil = pupil_mask(right_mask, eyes)

        draw_frame = img.copy()
        left_pos = eye_pos(left, left_pupil, draw_frame)
        right_pos = eye_pos(right, right_pupil, draw_frame)

        print(right_pos)




    #     mask = (eyes == [0, 0, 0]).all(axis=2)
    #     eyes[mask] = [255, 255, 255]
    #     mid = (shape[42][0] + shape[39][0]) // 2
    #     eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
    #     threshold = cv2.getTrackbarPos('threshold', 'image')
    #     _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
    #     thresh = cv2.erode(thresh, None, iterations=2)  # 1
    #     thresh = cv2.dilate(thresh, None, iterations=4)  # 2
    #     thresh = cv2.medianBlur(thresh, 3)  # 3
    #     thresh = cv2.bitwise_not(thresh)
    #     contouring(thresh[:, 0:mid], mid, img)
    #     contouring(thresh[:, mid:], mid, img, True)
    #     # for (x, y) in shape[36:48]:
    #     #     cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
    # # show the image with the face detections + facial landmarks
    cv2.imshow("pupil1", left_pupil)
    cv2.imshow("pupil2", right_pupil)
    resize_image('result', draw_frame, 2)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()