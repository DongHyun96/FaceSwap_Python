import numpy as np
import cv2
import dlib
from Face_Swap import face_swap_origin as fs
from Face_Landmark import image_detect as i_d


def calculate_landmarks(face_detector, predictor, image):
    # create list to contain landmarks
    landmark_list = []
    for face in face_detector:
        landmarks = predictor(image, face)  # 얼굴에서 68개 점 찾기

        # append (x, y) in landmark_list
        for p in landmarks.parts():
            landmark_list.append((p.x, p.y))

    return landmark_list


def swapping(src_image, src_points, current_image, current_landmarks):
    img1Warped = np.copy(current_image)
    # Find convex hull
    hull1 = []
    hull2 = []

    hullIndex = cv2.convexHull(np.array(current_landmarks), returnPoints=False)

    for i in range(0, len(hullIndex)):
        hull1.append(src_points[int(hullIndex[i])])
        hull2.append(current_landmarks[int(hullIndex[i])])

    # Find delanauy traingulation for convex hull points
    sizeImg2 = current_image.shape
    rect = (0, 0, sizeImg2[1], sizeImg2[0])
    try:  # dt can be out of current frame range
        dt = fs.calculateDelaunayTriangles(rect, hull2)

        if len(dt) == 0:
            quit()

        # Apply affine transformation to Delaunay triangles
        for i in range(0, len(dt)):
            t1 = []
            t2 = []

            # get points for img1, img2 corresponding to the triangles
            for j in range(0, 3):
                t1.append(hull1[dt[i][j]])
                t2.append(hull2[dt[i][j]])

            fs.warpTriangle(src_image, img1Warped, t1, t2)

        # Calculate Mask
        hull8U = []
        for i in range(0, len(hull2)):
            hull8U.append((hull2[i][0], hull2[i][1]))

        mask = np.zeros(current_image.shape, dtype=current_image.dtype)

        cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

        r = cv2.boundingRect(np.float32([hull2]))

        center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))

        # Clone seamlessly.
        output = cv2.seamlessClone(np.uint8(img1Warped), current_image, mask, center, cv2.NORMAL_CLONE)
        return output
    except:
        return current_image


def realtime_loop(src_directory, detector, predictor):
    cv2.namedWindow('result', cv2.WINDOW_KEEPRATIO)
    vid_in = cv2.VideoCapture(0)

    # Read src image
    src_image, Face_Detector1, Predictor1 = i_d.face_detection(src_directory)

    #src_points = fs.readPoints(src_directory + '.txt')
    src_points = i_d.add_landmarks(src_image, Face_Detector1, Predictor1)

    prevtime = 0

    while True:
        ret, image_o = vid_in.read()
        image = cv2.flip(image_o, 1)

        # resize the video
        #image = cv2.resize(image, dsize=(640, 480), interpolation=cv2.INTER_AREA)
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get faces (up-sampling=1)
        face_detector = detector(img_gray, 1)

        # the number of face detected
        print("The number of faces detected : {}".format(len(face_detector)))

        output = image
        if len(face_detector) == 1:
            landmark_list = calculate_landmarks(face_detector, predictor, image)
            output = swapping(src_image, src_points, image, landmark_list)

        cv2.imshow('result', output)

        # wait for keyboard input
        key = cv2.waitKey(1)

        # if esc,
        if key == 27:
            break

    vid_in.release()


if __name__ == '__main__':
    # create face detector, predictor
    Detector = dlib.get_frontal_face_detector()
    Predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')

    # Real-time Loop
    src_dir = "C:/Users/wadan/Desktop/cv_image/face_swap/iu.jpg"
    realtime_loop(src_dir, Detector, Predictor)
