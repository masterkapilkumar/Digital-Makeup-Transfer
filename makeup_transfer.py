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

def PointInsideTriangle2(pt,poly):
	hull = cv2.convexHull(np.array(poly))
	dist = cv2.pointPolygonTest(hull,(pt[0], pt[1]),False)
	if dist>=0:
		return True
	else:
		return False
	        

def cpartition(points1, size0, size1):

	cmat = np.zeros((size0,size1))

	left_eye = points1[30:38] 
	right_eye = points1[40:48]
	mouth = points1[66:72]
	lips = points1[59:66] + points1[72:77]
	skin = points1[0:16]

	for y in xrange(size0):
		for x in xrange(size1):
			if PointInsideTriangle2((x,y),skin):
				if PointInsideTriangle2((x,y),left_eye):
					cmat[y][x] = 3
				elif PointInsideTriangle2((x,y),right_eye):
					cmat[y][x] = 3
				elif PointInsideTriangle2((x,y),lips):
					if PointInsideTriangle2((x,y),mouth):
						cmat[y][x] = 3
					else:
						cmat[y][x] = 2
				else:
					cmat[y][x] = 1

	return cmat



def main():
    
    print "Reading subject image"
    img1 = cv2.imread(sys.argv[1])
    img_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img_LAB = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
    size = img1.shape

    print "Getting feature points of subject image"
    landmarks = stasm.search_single(img_gray)
    points1 = stasm.force_points_into_image(landmarks, img_gray)
    points1 = list(map(tuple, points1))

    print "Reading example image"
    img2 = cv2.imread(sys.argv[2])
    img_gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    print "Getting feature points of example image"
    landmarks2 = stasm.search_single(img_gray2)
    points2 = stasm.force_points_into_image(landmarks2, img_gray2)
    points2 = list(map(tuple, points2))

    #printing the feature points
    img_points = img1.copy()
    img_points2 = img2.copy()
    
    for point in points1:
        img_points[int(round(point[1]))][int(round(point[0]))] = 255
    for point in points2:
        img_points2[int(round(point[1]))][int(round(point[0]))] = 255

    #display(img_points)
    #display(img_points2)

    print "Face alignment through warping"
    if( len(points1) == len(points2) ):

        points = []
        alpha = 0 
        
        # Compute weighted average point coordinates
        for i in range(0, len(points1)):
            x = ( 1 - alpha ) * points1[i][0] + alpha * points2[i][0]
            y = ( 1 - alpha ) * points1[i][1] + alpha * points2[i][1]
            points.append((x,y))
        
        imgMorph = np.zeros(img1.shape, dtype = img1.dtype)

        ##denaulay triangulation
        
        rect = (0, 0, size[1], size[0])
        subdiv = cv2.Subdiv2D(rect)
        for p in points1:
            subdiv.insert(p)
        triangle_list = subdiv.getTriangleList()
        
        points_map = {}
        for i in range(len(points1)):
            points_map[ str(points1[i]) ] = i
        
        
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
                
                face_morphing.morphTriangle(img1, img2, imgMorph, t1, t2, t, 1)
    

    distance = 8
    sigmacolor = 85
    #display(imgMorph)

    print "Partitioning the image into C1, C2, C3"
    cmat = cpartition(points1,size[0],size[1])

    print "Layer decomposition"
    lightness_layer = img_LAB[..., 0]
    Ic_A = img_LAB[..., 1]
    Ic_B = img_LAB[..., 2]

    img_LAB2 = cv2.cvtColor(imgMorph, cv2.COLOR_BGR2LAB)
    lightness_layer2 = img_LAB2[..., 0]
    Ec_A = img_LAB2[..., 1]
    Ec_B = img_LAB2[..., 2]
    
    print "Applying bilateral filter on base image"
    #face_structure_layer = cv2.bilateralFilter(lightness_layer,distance,sigmacolor,0)
    face_structure_layer, skin_detail_layer = wls_filter.wlsfilter_layer(lightness_layer,cmat) 
    #display(face_structure_layer,mode='gray')
    #skin_detail_layer = lightness_layer - face_structure_layer
    Is, Id = face_structure_layer, skin_detail_layer

    print "Applying bilateral filter on example image"
    #face_structure_layer2 = cv2.bilateralFilter(lightness_layer2,distance,sigmacolor,0)
    face_structure_layer2, skin_detail_layer2 = wls_filter.wlsfilter_layer(lightness_layer2,cmat) 
    #skin_detail_layer2 = lightness_layer2 - face_structure_layer2
    #display(skin_detail_layer2,mode='gray')
    Es, Ed = face_structure_layer2, skin_detail_layer2


    #skin detail transfer
    print "Skin detail transfer"
    del_I = 0    #original skin details weight
    del_E = 1    #example skin details weight

    Rd = del_I * Id + del_E * Ed
    
    #color transfer
    print "Color transfer"
    
    #printing the C3 points
    #img_gray = np.zeros((size[0],size[1]))
    #lrev = points1[30:33]
    #left_eye = points1[33:38] + lrev

    #for point in left_eye:
    #   img_gray[int(round(point[1]))][int(round(point[0]))] = 255
       #display(img_gray,mode='gray')

    #temp = 71
    #img_gray[int(round(points1[temp][1]))][int(round(points1[temp][0]))] = 0

    

    gamma = 0.8
    Rc_A = (1-gamma)*Ic_A + (gamma)*Ec_A
    Rc_B = (1-gamma)*Ic_B + (gamma)*Ec_B

    for y in xrange(size[0]):
    	for x in xrange(size[1]):
    		if cmat[y][x]==0 or cmat[y][x]==3:
    			#Rc_A[y][x] = 255
    			#Rc_B[y][x] = 255
    			Rc_A[y][x] = Ic_A[y][x]
    			Rc_B[y][x] = Ic_B[y][x]
    			Rd[y][x] = Id[y][x]


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

