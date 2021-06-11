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


def swapping(current_image, current_landmarks):
    print(len(current_landmarks))
    imgWarped = np.copy(current_image)
    # Find convex hull
    hull1 = []
    hull2 = []

    hullIndex = cv2.convexHull(np.array(current_landmarks[68:]), returnPoints=False)

    for i in range(0, len(hullIndex)):
        hull1.append(current_landmarks[:68][int(hullIndex[i])])
        hull2.append(current_landmarks[68:][int(hullIndex[i])])

    # Find delanauy traingulation for convex hull points
    sizeImg = current_image.shape
    rect = (0, 0, sizeImg[1], sizeImg[0])

    try:  # dt can be out of current frame range
        dt1 = fs.calculateDelaunayTriangles(rect, hull1)
        dt2 = fs.calculateDelaunayTriangles(rect, hull2)

        if len(dt1) == 0 or len(dt2) == 0:
            quit()

        # Apply affine transformation to Delaunay triangles
        for i in range(0, len(dt1)):
            t1 = []
            t2 = []

            # get points for img1, img2 corresponding to the triangles
            for j in range(0, 3):
                t1.append(hull1[dt1[i][j]])
                t2.append(hull2[dt1[i][j]])

            fs.warpTriangle(current_image, imgWarped, t2, t1)

        #######################################################################
        for i in range(0, len(dt2)):
            t1 = []
            t2 = []

            # get points for img1, img2 corresponding to the triangles
            for j in range(0, 3):
                t1.append(hull1[dt2[i][j]])
                t2.append(hull2[dt2[i][j]])

            fs.warpTriangle(current_image, imgWarped, t1, t2)

        # Calculate Mask
        hull8U_1 = []
        hull8U_2 = []
        for i in range(0, len(hull2)):
            hull8U_1.append((hull1[i][0], hull1[i][1]))
            hull8U_2.append((hull2[i][0], hull2[i][1]))

        mask1 = np.zeros(current_image.shape, dtype=current_image.dtype)
        mask2 = np.zeros(current_image.shape, dtype=current_image.dtype)

        cv2.fillConvexPoly(mask1, np.int32(hull8U_1), (255, 255, 255))
        cv2.fillConvexPoly(mask2, np.int32(hull8U_2), (255, 255, 255))

        r1 = cv2.boundingRect(np.float32([hull1]))
        r2 = cv2.boundingRect(np.float32([hull2]))

        center1 = ((r1[0] + int(r1[2] / 2), r1[1] + int(r1[3] / 2)))
        center2 = ((r2[0] + int(r2[2] / 2), r2[1] + int(r2[3] / 2)))

        # Clone seamlessly.
        first = cv2.seamlessClone(np.uint8(imgWarped), current_image, mask1, center1, cv2.NORMAL_CLONE)
        second = cv2.seamlessClone(np.uint8(imgWarped), first, mask2, center2, cv2.NORMAL_CLONE)
        return second
    except:
        return current_image


def realtime_loop(detector, predictor):
    vid_in = cv2.VideoCapture(0)
    while True:
        ret, image_o = vid_in.read()
        image = cv2.flip(image_o, 1)

        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get faces (up-sampling=1)
        face_detector = detector(img_gray, 1)

        # the number of face detected
        #print(f"The number of faces detected : {len(face_detector)}")

        output = image
        if len(face_detector) == 2:
            landmark_list = calculate_landmarks(face_detector, predictor, image)
            output = swapping(image, landmark_list)

        cv2.imshow('result', output)

        # wait for keyboard input
        key = cv2.waitKey(1)

        # if esc,
        if key == 27:
            break

    vid_in.release()


if __name__ == '__main__':
    """
    Input -> one frame(image) two face
    processing -> simultaneous face swap
    Output -> One frame(image)
    """
    # create face detector, predictor
    Detector = dlib.get_frontal_face_detector()
    Predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')

    # Real-time Loop
    realtime_loop(Detector, Predictor)
