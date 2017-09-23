import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import stasm

import wls_filter
import face_morphing

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

def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

def main():
    
    img1 = cv2.imread(sys.argv[1])
    
    img_gray = cv2.imread(sys.argv[1], cv2.CV_LOAD_IMAGE_GRAYSCALE)
    img_LAB = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
    
    print "Getting feature points of base image"
    landmarks = stasm.search_single(img_gray)
    points1 = stasm.force_points_into_image(landmarks, img_gray)
    points1 = list(map(tuple, points1))
    
    lightness_layer = img_LAB[..., 0]
    Ic_A = img_LAB[..., 1]
    Ic_B = img_LAB[..., 2]
    print "Applying WLS filter on base image\n"
    face_structure_layer, skin_detail_layer = wls_filter.wlsfilter_layer(lightness_layer)
    Is, Id = face_structure_layer, skin_detail_layer
    
    #example
    img2 = cv2.imread(sys.argv[2])
    
    img_gray2 = cv2.imread(sys.argv[2], cv2.CV_LOAD_IMAGE_GRAYSCALE)
    
    print "Getting feature points of example image"
    landmarks2 = stasm.search_single(img_gray2)
    points2 = stasm.force_points_into_image(landmarks2, img_gray2)
    points2 = list(map(tuple, points2))
    
    img_points = img1.copy()
    img_points2 = img2.copy()
    
    for point in points1:
        img_points[int(round(point[1]))][int(round(point[0]))] = 255
    for point in points2:
        img_points2[int(round(point[1]))][int(round(point[0]))] = 255
    
    print "Number of feature points in base image:",len(points1)
    print "Number of feature points in example image:",len(points2)
    
    if( len(points1) == len(points2) ):
        # img1 = np.float32(img1)
        # img2 = np.float32(img2)
        
        points = []
        alpha = 0.5             #TODO - tune alpha
        
        # Compute weighted average point coordinates
        for i in range(0, len(points1)):
            x = ( 1 - alpha ) * points1[i][0] + alpha * points2[i][0]
            y = ( 1 - alpha ) * points1[i][1] + alpha * points2[i][1]
            points.append((x,y))
        
        
        ##denaulay triangulation
        size = img1.shape
        rect = (0, 0, size[1], size[0])
        subdiv = cv2.Subdiv2D(rect)
        for p in points1:
            subdiv.insert(p)
        triangle_list = subdiv.getTriangleList()
        
        points_map = {}
        for i in range(len(points1)):
            points_map[ str(points1[i]) ] = i
        
        imgMorph = np.zeros(img1.shape, dtype = img1.dtype)
        
        #morphing triangle one by one
        for p in triangle_list:
            p1, p2, p3 = (p[0],p[1]), (p[2],p[3]), (p[4],p[5])
            if rect_contains(rect, p1) and rect_contains(rect, p2) and rect_contains(rect, p3) :
                index1 = points_map[str(p1)]
                index2 = points_map[str(p2)]
                index3 = points_map[str(p3)]
                
                t1 = [p1, p2, p3]
                t2 = [points2[index1], points2[index2], points2[index3]]
                t = [points[index1], points[index2], points[index3]]
                
                face_morphing.morphTriangle(img1, img2, imgMorph, t1, t2, t, alpha)
    
    img_LAB2 = cv2.cvtColor(imgMorph, cv2.COLOR_BGR2LAB)
    lightness_layer2 = img_LAB2[..., 0]
    Ec_A = img_LAB2[..., 1]
    Ec_B = img_LAB2[..., 2]
    print "Applying WLS filter on example image"
    face_structure_layer2, skin_detail_layer2 = wls_filter.wlsfilter_layer(lightness_layer2)
    Es, Ed = face_structure_layer2, skin_detail_layer2
    
    #skin detail transfer
    del_I = 0    #original skin details weight
    del_E = 1    #example skin details weight
    Rd = del_I * Id + del_E * Ed
    
    #color transfer
    #TODO - include example image weight in eyes region
    gamma = 0.8
    Rc_A = Ic_A.copy()
    Rc_B = Ic_B.copy()
        
    
    # display(np.hstack([img1, img_points]))
    # display(np.hstack([img2, img_points2]))
    # display(imgMorph)
    
    result_LAB = img_LAB.copy()
    result_LAB[..., 0] = Is + Rd
    result_LAB[..., 1] = Rc_A
    result_LAB[..., 2] = Rc_B
    
    result = cv2.cvtColor(result_LAB, cv2.COLOR_LAB2BGR)
    cv2.imwrite('morphed.jpg', imgMorph)
    cv2.imwrite('result.jpg', result)
    display(result)


if __name__=='__main__':
    main()

