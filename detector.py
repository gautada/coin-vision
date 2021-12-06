import cv2

# https://www.geeksforgeeks.org/reading-image-opencv-using-python/
if "__main__" == __name__:
    img = cv2.imread("test01.png", cv2.IMREAD_COLOR)
    cv2.imshow("Cute Kitens", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
