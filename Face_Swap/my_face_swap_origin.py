import sys
import numpy as np
import cv2
import Face_Swap.face_swap_origin as fs
import Face_Landmark.image_detect as i_d


if __name__ == '__main__':

    # Make sure OpenCV is version 3.0 or above
    (major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')

    if int(major_ver) < 3:
        print(sys.stderr, 'ERROR: Script needs OpenCV 3.0 or higher')
        sys.exit(1)

    # Read images
    filename1 = "img/ted_cruz.jpg"
    filename2 = "img/brad.jpg"

    image, Face_Detector, Predictor = i_d.face_detection(filename2)
    landmark_list = i_d.add_landmarks(image, Face_Detector, Predictor)

    img1 = cv2.imread(filename1)
    img2 = image

    img1Warped = np.copy(img2)

    # Read array of corresponding points
    points1 = fs.readPoints(filename1 + '.txt')
    points2 = landmark_list

    # Find convex hull
    hull1 = []
    hull2 = []

    hullIndex = cv2.convexHull(np.array(points2), returnPoints=False)

    for i in range(0, len(hullIndex)):
        hull1.append(points1[int(hullIndex[i])])
        hull2.append(points2[int(hullIndex[i])])

    # Find delanauy traingulation for convex hull points
    sizeImg2 = img2.shape
    rect = (0, 0, sizeImg2[1], sizeImg2[0])

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

        fs.warpTriangle(img1, img1Warped, t1, t2)

    # Calculate Mask
    hull8U = []
    for i in range(0, len(hull2)):
        hull8U.append((hull2[i][0], hull2[i][1]))

    mask = np.zeros(img2.shape, dtype=img2.dtype)

    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

    r = cv2.boundingRect(np.float32([hull2]))

    center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))

    # Clone seamlessly.
    output = cv2.seamlessClone(np.uint8(img1Warped), img2, mask, center, cv2.NORMAL_CLONE)

    cv2.namedWindow("Face Swapped", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Face Swapped", output)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

