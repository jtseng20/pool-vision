import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

DEBUG = False
DIST_THRESHOLD = 10
TEMPLATE_SCALE = 1.5
AREA_THRESHOLD = 0.2
table_template_points = (TEMPLATE_SCALE*np.array([[0,0],[515,0],[515,295],[0,295]])).reshape(-1,1,2)

def createLineIterator(P1, P2, img):
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    #difference and absolute difference between points
    #used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    #predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa,dXa),3),dtype=np.float32)
    itbuffer.fill(np.nan)

    #Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X: #vertical line segment
        itbuffer[:,0] = P1X
        if negY:
            itbuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)
        else:
            itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)              
    elif P1Y == P2Y: #horizontal line segment
        itbuffer[:,1] = P1Y
        if negX:
            itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
        else:
            itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
    else: #diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = dX/dY
            if negY:
                itbuffer[:,1] = np.arange(P1Y-1,P1Y-dYa-1,-1)
            else:
                itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
                itbuffer[:,0] = (slope*(itbuffer[:,1]-P1Y)).astype(np.int) + P1X
        else:
            slope = dY/dX
            if negX:
                itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
            else:
                itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
            itbuffer[:,1] = (slope*(itbuffer[:,0]-P1X)).astype(np.int) + P1Y

    #Remove points outside of image
    colX = itbuffer[:,0]
    colY = itbuffer[:,1]
    itbuffer = itbuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]

    #Get intensities from img ndarray
    itbuffer[:,2] = img[itbuffer[:,1].astype(np.uint),itbuffer[:,0].astype(np.uint)]

    return itbuffer


def dist(x1,y1,x2,y2):
    return ((x2-x1)**2 + (y2-y1)**2)**0.5

def isBetween(ix,iy,x1,y1,x2,y2):
    return ((x1 <= ix <= x2) or (x1 >= ix >= x2)) and ((y1 <= iy <= y2) or (y1 >= iy >= y2))
    
def isClose(slope1, slope2):
    if abs(slope1) > 10: # for steep slopes 
        if abs(1/slope1 - 1/slope2) < 0.1:
            return True
    elif abs(slope1) > 1: # for medium slopes
        if abs(slope1/slope2 - 1) < 0.2:
            return True
    else: # for shallow slopes
        if abs(slope1 - slope2) < 0.1:
            return True
    return False

def isInLine(L1,L2):
    x1,y1,x2,y2 = L1
    x3,y3,x4,y4 = L2
    
    slope1 = (y2-y1 + 1e-8)/(x2-x1+1e-8)
    slope2 = (y4-y3 + 1e-8)/(x4-x3+1e-8)
    
    mx1, my1 = (x1+x2)/2, (y1+y2)/2
    mx2, my2 = (x3+x4)/2, (y3+y4)/2
    
    # Slopes gotta be similar
    if not isClose(slope1, slope2):
        return False
    
    # return early if midpoints are close
    if dist(mx1,my1,mx2,my2) < DIST_THRESHOLD:
        return True
    
    # return whether the second line has a point that's close to the midpoint of the first
    ix1, iy1 = line_intersection((mx1,my1,mx1+1,my1-(1/slope1)), L2)
    ix2, iy2 = line_intersection((mx2,my2,mx2+1,my2-(1/slope2)), L1)
    
    return dist(ix1,iy1, mx1,my1) < DIST_THRESHOLD or dist(ix2,iy2, mx2,my2) < DIST_THRESHOLD

def allFalse(l, ll):
    for u in ll:
        if isInLine(l, u):
            return False
    return True

def combineLines(L1, L2):
    x1,y1,x2,y2 = L1
    x3,y3,x4,y4 = L2
    slope = (y2-y1)/(x2-x1+1e-8)
    pointsArray = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
    
    # steep line, minimize/maximize y
    if abs(slope) > 1:
        minStart = pointsArray[np.argmin(pointsArray[:,1])]
        maxEnd = pointsArray[np.argmax(pointsArray[:,1])]
        return np.hstack((minStart,maxEnd))
    
    minStart = pointsArray[np.argmin(pointsArray[:,0])]
    maxEnd = pointsArray[np.argmax(pointsArray[:,0])]
    return np.hstack((minStart,maxEnd))

def getAbsSlope(L):
    x1,y1,x2,y2 = L
    return abs((y2-y1)/(x2-x1+1e-8))
    
def filterLines(lines):
    uniqueLines = []
    for line in lines:
        if allFalse(line[0], uniqueLines):
            uniqueLines.append(line[0])
            
    for i in range(len(uniqueLines)):
        for l in lines:
            if isInLine(l[0], uniqueLines[i]):
                #combine these two
                uniqueLines[i] = combineLines(uniqueLines[i], l[0])
    
    # Sort by absolute slope to make sure parallel pairs are together
    uniqueLines.sort(key = getAbsSlope)
    return uniqueLines
        

def line_intersection(A, B):
    # Intersection point between lines
    out = np.zeros(2)
    
    x1, y1, x2, y2 = A
    x3, y3, x4, y4 = B
    
    out[0] = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
    out[1] = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
    
    return int(out[0]),int(out[1])

def getCorners(lines):
    l = np.array([line_intersection(lines[0], lines[2]),line_intersection(lines[0], lines[3]),line_intersection(lines[1], lines[2]),line_intersection(lines[1], lines[3])])
    # process to order points in clockwise order >:D
    # first, get point closest to origin
    startingPoint = np.argmin(np.linalg.norm(l,axis=1))
    s = l[startingPoint]
    l = np.delete(l,startingPoint,0)
    shifted = l - s
    # then, sort by angle to the first point
    sc = np.vstack((s, l[np.argsort(np.arctan2(shifted[:,1], shifted[:,0]))]))
    
    # and then ... maybe rotate once to make sure first edge is longest
    if np.linalg.norm(sc[1] - sc[0]) < np.linalg.norm(sc[2] - sc[1]):
        sc = np.roll(sc, -1, axis=0)
    return sc

def drawCorners(img, corners, color = (0,255,0), radius = 5):
    for corner in corners:
        cv.circle(img, (corner[0],corner[1]), radius, color, 5)
        
def getTableMask(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    low_green = np.array([25, 52, 72])
    high_green = np.array([102, 255, 255])
    mask = cv.inRange(hsv, low_green, high_green)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.erode(mask, kernel, iterations=1)
    return mask
    
def findCornersAndTransform(img, template):
    outputRect = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    totalArea = img.shape[0]*img.shape[1]
    
    mask = getTableMask(img)
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv.contourArea(cnt)
        approx = cv.convexHull(cnt)

        if area > totalArea*AREA_THRESHOLD:
            cv.drawContours(outputRect, [approx], 0, 255, 2)
            #cv.fillPoly(outputRect, [approx], 255)

    lines = cv.HoughLinesP(outputRect,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
    filteredLines = filterLines(lines)
    
    if len(filteredLines) != 4 or DEBUG:
        print("Filtered {} lines down to {}".format(len(lines), len(filteredLines)))
        out = np.zeros(img.shape, dtype=np.uint8)
        print(filteredLines)
        for line in filteredLines:
            x1,y1,x2,y2 = line
            cv.line(out,(x1,y1),(x2,y2),(0,255,0),1)
            
        print("Found {} edges".format(len(filteredLines)))
        return out
    
    corners = np.float32(getCorners(filteredLines)).reshape(-1,1,2)
    M, _ = cv.findHomography(corners, template, 0)
    
    return corners, M
    
#############################################################

for i in range(8,14):
    img = cv.imread('testFrames/table'+str(i)+'.jpg')
    worldSpace = np.zeros((int(295*TEMPLATE_SCALE),int(515*TEMPLATE_SCALE),3), dtype=np.uint8)
    output = findCornersAndTransform(img, table_template_points)

    if len(output) == 2:
        corners, M = output
        t_corners = cv.perspectiveTransform(corners, M)

        worldSpace = cv.polylines(worldSpace,[np.int32(t_corners)],True,(0,255,0),10, cv.LINE_AA)
        img = cv.polylines(img,[np.int32(corners)],True,(0,255,255),10, cv.LINE_AA)

        plt.subplot(121),plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB))
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(worldSpace)
        plt.title('Transformed Image'), plt.xticks([]), plt.yticks([])
        plt.show()
    else:
        plt.imshow(cv.cvtColor(output,cv.COLOR_BGR2RGB))
        plt.title('Bugged Output'), plt.xticks([]), plt.yticks([])
        plt.show()
