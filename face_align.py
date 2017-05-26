from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from matplotlib import pyplot as plt

def rect_center(coords):
    x1, y1, w, h = coords
    center = [int(x1 + w/2), int(y1  + (h / 2))]
    return np.array(center)


def affine_transfer(img_example_name, img_name, coords_example, coords, size, show):
    img_example = cv2.imread(img_example_name)
    img = cv2.imread(img_name)
    img = imutils.resize(img, width=size)
    img_example = imutils.resize(img_example, width=size)

    pts_example = []
    pts = []
    for i in range(3):
        temp = coords_example[i] # + coords_example[0][:2]
        pts_example.append(temp)
        temp = coords[i] #+ coords[0][:2]
        pts.append(temp)
    pts_example = np.float32(pts_example)
    pts = np.float32(pts)

    if show:
        for i in [0, 1, 2]:
            cv2.circle(img, (pts[i][0], pts[i][1]), 4, (0,255,0), -1)
            cv2.imshow('bitch', img)
            cv2.waitKey(0)

        for i in [0, 1, 2]:
            cv2.circle(img_example, (pts_example[i][0], pts_example[i][1]), 4, (0,255,0), -1)
            cv2.imshow('exmpl', img_example)
            cv2.waitKey(0)

    rows,cols,ch = img_example.shape
    M = cv2.getAffineTransform(pts,pts_example)
    dst = cv2.warpAffine(img,M,(cols,rows))

    if show:
        a = plt.subplot(132),plt.imshow(img),plt.title('Input')
        b = plt.subplot(131),plt.imshow(img_example),plt.title('Example')
        c = plt.subplot(133),plt.imshow(dst),plt.title('Output')
        plt.show()
    return dst


def get_facial_features(img_name, size, show=False):
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(img_name)
    image = imutils.resize(image, width=size)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    # 1 face in the image
    rect = rects[0] 
    # determine the facial landmarks for the face region, then
    # convert the landmark (x, y)-coordinates to a NumPy array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    feature_points = []
    center_coords = []
    # loop over the face parts individually
    for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
        face_part_points = []
        # clone the original image so we can draw on it, then
        # display the name of the face part on the image
        clone = image.copy()
        cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 255), 2)

        # loop over the subset of facial landmarks, drawing the
        # specific face part
        for (x, y) in shape[i:j]:
            cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
            face_part_points.append((x, y))

        feature_points.append(face_part_points)
        # extract the ROI of the face region as a separate image
        (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
        if name in ['mouth', 'left_eye', 'right_eye']:
            center_coords.append(rect_center((x, y, w, h)))
        roi = image[y:y + h, x:x + w]
        roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

        if show:
            # show the particular face part
            cv2.imshow("ROI", roi)
            cv2.imshow("Image", clone)
            cv2.waitKey(0)

    # visualize all facial landmarks with a transparent overlay
    output = face_utils.visualize_facial_landmarks(image, shape)
    cv2.imshow("Image", output)
    cv2.waitKey(0)
    return feature_points, center_coords

def pose_transfer(img_example_name, img_name, size, show):
    features_example, coords_example = get_facial_features(img_example_name, size, show)
    features, coords = get_facial_features(img_name, size)
    dst = affine_transfer(img_example_name, img_name, coords_example, coords, size, show)
    return dst


# f1 is example
img_example_name = './images/example_1.png'
x = cv2.imread(img_example_name)
img_name = './images/face_1.jpg' 
##img_example_name = 'face_1.jpg'
#img_name = 'test_face.jpg' 
dst = pose_transfer(img_example_name, img_name, 500, False)
cv2.imshow('d', dst)
cv2.waitKey(0)

