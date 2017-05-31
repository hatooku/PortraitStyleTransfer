import numpy as np
from imutils import face_utils
from PIL import Image
import imutils
import dlib
import cv2
import time
from matplotlib import pyplot as plt

""" Given the bounding box coordinates in the form of x1, y1, w, h.
    Calculates the center of the rectangle.

    """
def rect_center(coords):
    x1, y1, w, h = coords
    center = [int(x1 + w/2), int(y1  + (h / 2))]
    return np.array(center)

""" converts images to a color scheme that cv2 can display 
   
    """
def cv2_display(image):
    im2 = image.copy()
    im2[:, :, 0] = image[:, :, 2]
    im2[:, :, 2] = image[:, :, 0]
    return im2

""" Prevent morphing from trying to copy a pixel out of range

    """
def trim_x_morph(X_morph, width, height):
    x1 = X_morph
    if x1[0] >= width:
        x1[0] = width - 1
    elif x1[0] < 0:
        x1[0] = 0
    if x1[1] >= height:
        x1[1] = height - 1
    elif x1[1] < 0:
        x1[1] = 0
    return x1

""" Return minimum distance between line segment pq and point

    """
def minimum_distance(p, q, point):
    x1 = p[0]
    y1 = p[1]
    x2 = q[0]
    y2 = q[1]
    num = 1. * (y2 - y1) * point[0] - (x2 - x1) * point[1]

    num += x2 * y1 - y2 * x1
    num = np.abs(num)
    denom = (y2 - y1)**2 + (x2 - x1)**2
    denom = np.sqrt(denom)
    return num / denom 

""" Performs an affine transformation between two images

    args:
    img_example (image): the image to be warped
    img (image):    the image the other images is being warped to match
    coords_example (points array): contains locations of 3 features in example
    coords (points array): contains locations of 3 features in base image
    show (bool): If true, plots the points used for the transformation and shows result

    """
def affine_transfer(img_example, img, coords_example, coords, show):
    pts_example = []
    pts = []
    for i in range(3):
        temp = np.array(coords_example[i]) # + coords_example[0][:2]
        pts_example.append(temp)
        temp = np.array(coords[i]) #+ coords[0][:2]
        pts.append(temp)
    pts_example = np.float32(pts_example)
    pts = np.float32(pts)

    if show:
        clone = img.copy()
        clone_exmp = img_example.copy()
        for i in [0, 1, 2]:
            cv2.circle(clone, (pts[i][0], pts[i][1]), 4, (0,255,0), -1)
            cv2.imshow('bitch', cv2_display(clone))
            cv2.waitKey(0)

        for i in [0, 1, 2]:
            cv2.circle(clone_exmp, (pts_example[i][0], pts_example[i][1]), 4, (0,255,0), -1)
            cv2.imshow('exmpl', cv2_display(clone_exmp))
            cv2.waitKey(0)

    rows,cols,ch = img.shape
    M = cv2.getAffineTransform(pts_example, pts)
    dst = cv2.warpAffine(img_example,M,(cols,rows))

    if show:
        plt.title("Transformation after Affine Transfer")
        a = plt.subplot(132),plt.imshow(img),plt.title('Input')
        b = plt.subplot(131),plt.imshow(img_example),plt.title('Example')
        c = plt.subplot(133),plt.imshow(dst),plt.title('Output')
        plt.show()
    return dst

""" Finds facial features in an image. 

    Uses a pretrained detector to find facial features.
    args: 
        image (image) the image to identify
        show (bool) if true, display the features found

    Returns:

    feature points: a list of 7 arrays. The arrays contain feature
    points for the mouth, left_eyebrow, right_eyebrow, left_eye, right_eye, nose
    and jawline in order.

    center coords (array):
        contains an array of 3 points: the centers of the bounding boxes for the 
        mouth and the eyes 

    """
def get_facial_features(image, show=False):
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # load the input image, resize it, and convert it to grayscale

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
            cv2.circle(clone, (x, y), 2, (255, 0, 0), -1)
            face_part_points.append(np.array([x, y]))

        feature_points.append(face_part_points)
        # extract the ROI of the face region as a separate image
        (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
        if name in ['mouth', 'left_eye', 'right_eye']:
            center_coords.append(rect_center((x, y, w, h)))
        roi = image[y:y + h, x:x + w]
        roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

        if show:
            # show the particular face part
            cv2.imshow("ROI", cv2_display(roi))
            cv2.imshow("Image", cv2_display(clone))
            cv2.waitKey(0)

    if show:
        # visualize all facial landmarks with a transparent overlay
        output = face_utils.visualize_facial_landmarks(image, shape)
        plt.imshow(output)
        plt.show()
    return feature_points, center_coords

""" Calculates the u parameter for morphing 
    
    """
def get_u(p, q, x):
    u = (x - p) * (q - p) / (np.linalg.norm(q - p) ** 2)
    return u

""" Calculates the v parameter for morphing 
    
    """    
def get_v(p, q, x):
    num_term = q - p
    num_term = [num_term[1], -num_term[0]] # perpendicular vector
    v = (x - p) * num_term / np.linalg.norm(q - p)
    return v

""" Calculates the x1 parameter for morphing 
    
    """
def get_x1(u, v, p1, q1):
    num_term = q1 - p1
    num_term = [num_term[1], -num_term[0]] # perpendicular vector
    x1 = p1 + u * (q1 - p1) + (v * (num_term) / np.linalg.norm(q1 - p1))
    return x1

""" morphs one image to match the shape of the other.

    Using lines that define facial features of two images, creates the morphed
    version of the example image by writing 1 pixel at the time. Each pixel in
    the result is placed by sampling a pixel from the starting example image. 
    Which pixel to sample is given by the morph which is calculated using the 
    lines of both set 

    args:
    img_example (image): the image to be warped
    img (image):    the image the other images is being warped to match
    features_example (points double array): contains locations of 3 features in example
    features (points double array): contains locations of 3 features in base image
    A (double): larger A = smoother warp but less precise
    B (double): larger B, the more points will only be warped based on nearby
                lines and less from far ones
    p (double) a larger p gives longer lines more weight than short ones

    """

def morph_image(img_example, img, fts_example, fts, a, b, p_const, use_all_features=True, show=False):   
    clone = img.copy()
    clone_two = img_example.copy()
    res = img_example.copy()

    width = len(img_example)
    height = len(img_example[0])
    if use_all_features:
        features_example = fts_example
        features = fts
    else:
        features_example = get_points_for_morph(fts_example)
        features = get_points_for_morph(fts)

    # for each pixel in the image
    for i in range(width):
        if (i % 50 == 0) and i > 0:
            print ("Done with %f percent" % (100. * i / width))
        for j in range(height):
            x = np.array([i, j])
            Dsum = np.array([0., 0.])
            weight_sum = 0
            # for each feature point
            for feat in range(7): # not using eyes and mouth here
                n = len(features_example[feat])
                if n == 0:
                    continue
                assert n == len(features[feat])
                # using consecutive points of features rn
                for feat_pt in range(0, n):
                    p = features[feat][feat_pt - 1]
                    q = features[feat][feat_pt]        

                    # temp = (np.array(img_example.shape[:2], dtype='float32') / np.array(img.shape[:2], dtype='float32')) 
                    # temp = np.array(temp, dtype='int')
                    # p *= temp   
                    # q *= temp              

                    p1 = features_example[feat][feat_pt - 1]
                    q1 = features_example[feat][feat_pt]
                    
                    u = get_u(p, q, x)
                    v = get_v(p, q, x)
                    x1 = get_x1(u, v, p1, q1)
                    
                    Di = x1 - x
                    dist = minimum_distance(p, q, x)
                    length = np.linalg.norm(p - q)
                    weight = (length**p_const / (a + dist))**b

                    Dsum += Di * weight
                    weight_sum += weight
                    if show and i ==0 and j == 0:
                        cv2.line(clone, tuple(p), tuple(q),(0,255,0),2)
                        cv2.line(clone_two, tuple(p1), tuple(q1),(0,255,0),2)
                if show and i == 0 and j == 0:
                    cv2.imshow("Image", cv2_display(clone))
                    cv2.imshow("Image_two", cv2_display(clone_two))
                    cv2.waitKey(0)
                    
            X_morph = x + Dsum / weight_sum
            X_morph = trim_x_morph(X_morph, width, height)
            res[i][j] = img_example[int(X_morph[0]), int(X_morph[1])]
    return res

""" Performs a morph like above but only using the line of the bridges of the
    nose.

    """
def morph_one_line(img_example, img, features_example, features):
    dst = img_example.copy()
    p = features[5][0]
    q = features[5][1]
    p1 = features_example[5][0]
    q1 = features_example[5][1]

    width = len(img)
    height = len(img[0])

    for i in range(width):
        for j in range(height):
            x = np.array([i, j])
            u = get_u(p, q, x)
            v = get_v(p, q, x)
            x1 = get_x1(u, v, p1, q1)

            x1 = trim_x_morph(x1, width, height)
            dst[i][j] = img_example[int(x1[0])][int(x1[1])]
    return dst

""" Function to allow for only a subset of the facial features to be used as 
    lines in the morph. Using fewer lines will make the morphing faster but it
    may make the morph look a bit different. 

    """
def get_points_for_morph(features):
    points_to_take = {
        0: range(13), # mouth
        1: range(0, 5, 2), # eye brows (l, r)
        2: range(0, 5, 2),
        3: range(6), # eyes, (l , r)
        4: range(6),
        5: [0, 3, 4, 6, 8], # nose
        6: range(0, 17, 2) # jawline
    } 
    features_to_use = [1, 2, 5, 6] # not using eyes and mouth here
    points = []
    for feat in range(7): 
        temp = []
        if feat in features_to_use:
            n = len(features[feat])
            for feat_pt in range(0, n):
                if feat_pt in points_to_take[feat]:
                    temp.append(features[feat][feat_pt])
        points.append(temp)
    return points

""" Matches the pose between two images """
def pose_transfer(img_example_pil, img_pil, show=True, a=0.1, b=1.25, p=0.5):
    img = np.asarray(img_pil, dtype='uint8')
    img_example = content_array = np.asarray(img_example_pil, dtype='uint8')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_example = cv2.cvtColor(img_example, cv2.COLOR_BGR2GRAY)

    print ('finding features in the example face')
    _, coords_example = get_facial_features(img_example, show)
    print ('finding features in the input face')
    features, coords = get_facial_features(img, show)
    print ('performing affine transfer')
    dst = affine_transfer(img_example, img, coords_example, coords, False)
    gray_example = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    #sift_flow(gray_example, gray)
    features_example, coords_example = get_facial_features(dst, show)

    print ('morphing')
    t  = time.time()
    morph = morph_image(dst, img, features_example, features, a, b, p, use_all_features=False, show=show)
    print ("time spent morphing: %f seconds" % (time.time() - t))

    
    plt.title("Transformation after Pose Transfer")
    a = plt.subplot(142),plt.imshow(cv2_display(img)), plt.title('Input')
    b = plt.subplot(141),plt.imshow(cv2_display(img_example)),plt.title('Example')
    c = plt.subplot(143),plt.imshow(cv2_display(dst)),plt.title('Affine Transfer')
    d = plt.subplot(144),plt.imshow(cv2_display(morph)),plt.title('Morph')
    plt.show()
    return morph 

if __name__ == "__main__":
    height = 512
    width = 512

    img_name = './images/example_1.png'
    #img_name = './images/face_3.jpg' 
    #img_name = './images/styles/portrait1.jpg' 
    img_example_name = './images/face_1.jpg'
    #img_example_name = './images/example_2.png'
    #img_example_name = './images/styles/portrait1.jpg' 

    a = 100 # larger A = smoother warp but less precise
    b = 1.25 # the larger B, the more points will only be warped based on nearby lines and less from far ones
    p = 0.5  # # larger p gives longer lines more weight than short ones

    content_image = Image.open(img_name)
    content_image = content_image.resize((height, width))

    style_image = Image.open(img_example_name)
    style_image = style_image.resize((height, width))

    res = pose_transfer(style_image, content_image, True, a, b, p)
    # Code to crop the image if needed:
    # gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
    # _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    # contours= cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # cnt = contours[0]
    # x,y,w,h = cv2.boundingRect(cnt)
    # res = res[y:y+h,x:x+w]
    cv2.imshow('result', cv2_display(res))
    cv2.waitKey(0)

