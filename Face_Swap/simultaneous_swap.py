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
    filename1 = "img/iu.jpg"
    filename2 = "img/onebin.jpg"

    image1, Face_Detector1, Predictor1 = i_d.face_detection(filename1)
    image2, Face_Detector2, Predictor2 = i_d.face_detection(filename2)

    # Read array of corresponding points
    points1 = i_d.add_landmarks(image1, Face_Detector1, Predictor1)
    points2 = i_d.add_landmarks(image2, Face_Detector2, Predictor2)

    img1Warped = np.copy(image2)
    img2Warped = np.copy(image1)

    # Find convex hull
    hull1 = []
    hull2 = []

    hullIndex = cv2.convexHull(np.array(points2), returnPoints=False)

    for i in range(0, len(hullIndex)):
        hull1.append(points1[int(hullIndex[i])])
        hull2.append(points2[int(hullIndex[i])])


    # Find delanauy traingulation for convex hull points
    sizeImg1 = image1.shape
    sizeImg2 = image2.shape
    rect1 = (0, 0, sizeImg1[1], sizeImg1[0])
    rect2 = (0, 0, sizeImg2[1], sizeImg2[0])

    dt1 = fs.calculateDelaunayTriangles(rect1, hull1)
    dt2 = fs.calculateDelaunayTriangles(rect2, hull2)

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

        fs.warpTriangle(image2, img2Warped, t2, t1)

    #########################################################################################

    for i in range(0, len(dt2)):
        t1 = []
        t2 = []

        # get points for img1, img2 corresponding to the triangles
        for j in range(0, 3):
            t1.append(hull1[dt2[i][j]])
            t2.append(hull2[dt2[i][j]])

        fs.warpTriangle(image1, img1Warped, t1, t2)

    # Calculate Mask
    hull8U_1 = []
    hull8U_2 = []
    print(f'wow = {len(hull2)}')
    print(f'wow = {len(hull1)}')

    for i in range(0, len(hull2)):
        hull8U_1.append((hull1[i][0], hull1[i][1]))
        hull8U_2.append((hull2[i][0], hull2[i][1]))

    mask1 = np.zeros(image1.shape, dtype=image1.dtype)
    mask2 = np.zeros(image2.shape, dtype=image2.dtype)

    cv2.fillConvexPoly(mask1, np.int32(hull8U_1), (255, 255, 255))
    cv2.fillConvexPoly(mask2, np.int32(hull8U_2), (255, 255, 255))

    r1 = cv2.boundingRect(np.float32([hull1]))
    r2 = cv2.boundingRect(np.float32([hull2]))

    center1 = ((r1[0] + int(r1[2] / 2), r1[1] + int(r1[3] / 2)))
    center2 = ((r2[0] + int(r2[2] / 2), r2[1] + int(r2[3] / 2)))

    # Clone seamlessly.
    output1 = cv2.seamlessClone(np.uint8(img2Warped), image1, mask1, center1, cv2.NORMAL_CLONE)
    output2 = cv2.seamlessClone(np.uint8(img1Warped), image2, mask2, center2, cv2.NORMAL_CLONE)

    cv2.imshow("Face Swapped1", output1)
    cv2.imshow("Face Swapped2", output2)

    cv2.waitKey(0)

    cv2.destroyAllWindows()
