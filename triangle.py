from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

def PointInsideTriangle2(pt,poly):

	point = Point(pt[0], pt[1])
	polygon = Polygon(poly)
	return polygon.contains(point)		

print PointInsideTriangle2((0.5,0.5),[(0,0),(2,0),(0,2),(2,2),(0,0)])