import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

def display(img, name='', mode='bgr'):
    if mode == 'bgr':
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    elif mode == 'rgb':
        plt.imshow(img)
    elif mode == 'gray':
        plt.imshow(img, 'gray')
    elif mode == 'rainbow': # for 0-1 img
        plt.imshow(img, cmap='rainbow')
    else:
        raise ValueError('unkown mode')
    plt.title(name)
    plt.show()

def applyAffineTransform(src, srcTri, dstTri, size) :
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    return cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )


# Warps triangular regions from img1 and img2 to img
def morphTriangle(img1, img2, img, t1, t2, t, alpha) :
    
    # Offset points by left top corner of the respective rectangles
    t1Rect, t2Rect, tRect = [], [], []
    
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, (r[2], r[3]))
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, (r[2], r[3]))

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask

