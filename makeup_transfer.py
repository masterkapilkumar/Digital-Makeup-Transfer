import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import stasm

import wls_filter

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

def main():
    
    img = cv2.imread(sys.argv[1])
    
    img_gray = cv2.imread(sys.argv[1], cv2.CV_LOAD_IMAGE_GRAYSCALE)
    img_LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    landmarks = stasm.search_single(img_gray)
    points = stasm.force_points_into_image(landmarks, img_gray)
    
    lightness_layer = img_LAB[..., 0]
    Ic_A = img_LAB[..., 1]
    Ic_B = img_LAB[..., 2]
    face_structure_layer, skin_detail_layer = wls_filter.wlsfilter_layer(lightness_layer)
    Is, Id = face_structure_layer, skin_detail_layer
    
    #example
    img2 = cv2.imread(sys.argv[2])
    
    img_gray2 = cv2.imread(sys.argv[2], cv2.CV_LOAD_IMAGE_GRAYSCALE)
    img_LAB2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
    
    landmarks2 = stasm.search_single(img_gray2)
    points2 = stasm.force_points_into_image(landmarks2, img_gray2)
    
    lightness_layer2 = img_LAB2[..., 0]
    Ec_A = img_LAB2[..., 1]
    Ec_B = img_LAB2[..., 2]
    face_structure_layer2, skin_detail_layer2 = wls_filter.wlsfilter_layer(lightness_layer2)
    Es, Ed = face_structure_layer2, skin_detail_layer2
    
    #skin detail transfer
    del_I = 0    #original skin details weight
    del_E = 0    #example skin details weight
    # Rd = del_I * Id + del_E * Ed
    
    #color transfer
    gamma = 0.8
    
    img_points = img.copy()
    
    for point in points:
        img_points[int(round(point[1]))][int(round(point[0]))] = 255

    display(np.hstack([img, img_points]))



if __name__=='__main__':
    main()

