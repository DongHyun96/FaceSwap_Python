import dlib
import cv2
import numpy as np

# create list for landmarks
ALL = list(range(0, 68))
RIGHT_EYEBROW = list(range(17, 22))
LEFT_EYEBROW = list(range(22, 27))
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
NOSE = list(range(27, 36))
MOUTH_OUTLINE = list(range(48, 61))
MOUTH_INNER = list(range(61, 68))
JAWLINE = list(range(0, 17))


def resize(img):
    print(f'before resize = [{img.shape[1]}, {img.shape[0]}]')
    if img.shape[0] < 1000 or img.shape[1] < 1000:
        return img
    elif 1000 < img.shape[0] < 2000 or 1000 < img.shape[1] < 2000:
        return cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    else:
        return cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))


def face_detection(src_dir):
    # create face detector, predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')

    image_o = cv2.imread(src_dir)

    image = resize(image_o)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get faces (up-sampling=1)
    face_detector = detector(img_gray, 1)

    # the number of face detected
    print("The number of faces detected : {}".format(len(face_detector)))

    return image, face_detector, predictor


def add_landmarks(img, detector, predictor):
    # loop as the number of face
    landmark_list = []
    for face in detector:
        landmarks = predictor(img, face)  # 얼굴에서 68개 점 찾기
        for p in landmarks.parts():
            landmark_list.append((p.x, p.y))
            #cv2.circle(img, (p.x, p.y), 3, (0, 0, 255), -1)
        #cv2.namedWindow("img", cv2.WINDOW_KEEPRATIO)
        #cv2.imshow("img", img)
        #cv2.waitKey()
    return landmark_list
