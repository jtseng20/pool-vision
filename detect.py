import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

DEBUG = False
DIST_THRESHOLD = 10
TEMPLATE_SCALE = 7.5
AREA_THRESHOLD = 0.2
PYRAMID_SCALE = 0.6
BALL_SIZE = int(2.25 * 2 * TEMPLATE_SCALE)
table_template_points = (TEMPLATE_SCALE*np.array([[0,0],[88,0],[88,44],[0,44]])).reshape(-1,1,2)
cueBallTemplate = cv.resize(cv.imread('testFrames/templateBallBlank.png'), (BALL_SIZE, BALL_SIZE), interpolation = cv.INTER_AREA)

# Marked for removal
def createPyramid(img, min_size = 10):
    out = []
    currentScale = 1.0
    while True:
        currentScale *= PYRAMID_SCALE
        if img.shape[0] * currentScale <= min_size or img.shape[1] * currentScale <= min_size:
            break
            
        width = int(img.shape[1] * currentScale)
        height = int(img.shape[0] * currentScale)
        dim = (width, height)
        rescaled_img = cv.resize(img,dim,interpolation = cv.INTER_AREA)
        out.append(rescaled_img)
        
    return out

# Marked for removal
def returnPyramidMatch(img, template):
    pyramid = createPyramid(template)
    res = np.zeros((img.shape[0],img.shape[1]),dtype=np.float32)
    for t in pyramid:
        match = cv.matchTemplate(img,t,3)
        res[:match.shape[0], :match.shape[1]] += match
        
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    return max_loc

def locateBall(img, template):
    res = cv.matchTemplate(img,template,5)
    # Convolve with box filter to encourage finding the point with 
    # sustained average high matching probability, instead of just a single high value.
    # Box filter chosen over gaussian / median to emphasize averaging
    smoothingKernel = np.ones((5,5), np.float32)/25
    res = cv.filter2D(res, -1, smoothingKernel)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    return res, max_loc, max_val


def dist(x1,y1,x2,y2):
    return ((x2-x1)**2 + (y2-y1)**2)**0.5
    
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
    # Determines if two line segments are aligned and should be combined
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

# Helper function for filtering lines.
# Returns true if l is not aligned with anything in ll.
def allFalse(l, ll):
    for u in ll:
        if isInLine(l, u):
            return False
    return True

def combineLines(L1, L2):
    # Take two lines, ***assumed to be aligned***, and combines them
    x1,y1,x2,y2 = L1
    x3,y3,x4,y4 = L2
    slope = (y2-y1)/(x2-x1+1e-8)
    pointsArray = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
    
    # steep line, minimize/maximize y ( make the combined line as long as possible )
    if abs(slope) > 1:
        minStart = pointsArray[np.argmin(pointsArray[:,1])]
        maxEnd = pointsArray[np.argmax(pointsArray[:,1])]
        return np.hstack((minStart,maxEnd))
    
    # shallow line, minimize/maximize x
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

# Marked for removal
def drawCorners(img, corners, color = (0,255,0), radius = 5):
    for corner in corners:
        cv.circle(img, (corner[0],corner[1]), radius, color, 5)
        
def getTableMask(img):
    # Masks out the green stuff in the scene. Maybe upgrade to use feature matching?
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    low_green = np.array([25, 52, 72])
    high_green = np.array([102, 255, 255])
    mask = cv.inRange(hsv, low_green, high_green)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.erode(mask, kernel, iterations=1)
    return mask
    
def findCornersAndTransform(img, template):
    # Finds the table, calculates the corner points, then finds the camera perspective
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
    
################### TESTING CODE ###############################

for i in range(4,14):
    img = cv.imread('testFrames/table'+str(i)+'.jpg')
    output = findCornersAndTransform(img, table_template_points)

    if len(output) == 2:
        corners, M = output
        worldSpace = cv.warpPerspective(img,M, (int(88*TEMPLATE_SCALE),int(44*TEMPLATE_SCALE)))
        img = cv.polylines(img,[np.int32(corners)],True,(0,255,255),10, cv.LINE_AA)
        
        ballLoc, (ballX, ballY), ballVal = locateBall(worldSpace, cueBallTemplate)
        if ballVal > 0.5:
            cv.circle(worldSpace, (ballX + BALL_SIZE // 2, ballY + BALL_SIZE // 2), BALL_SIZE, (0,0,255), 5)

        plt.subplot(131),plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB))
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(132),plt.imshow(cv.cvtColor(worldSpace,cv.COLOR_BGR2RGB))
        plt.title('Transformed Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(133),plt.imshow(ballLoc, cmap = "gray")
        plt.title('Ball Location Image'), plt.xticks([]), plt.yticks([])
        plt.show()
    else:
        plt.imshow(cv.cvtColor(output,cv.COLOR_BGR2RGB))
        plt.title('Bugged Output'), plt.xticks([]), plt.yticks([])
        plt.show()
