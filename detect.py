import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

WALLWIDTH = 2
SHORTSIDE = 44 + 2*WALLWIDTH
LONGSIDE = 88 + 2*WALLWIDTH

DEBUG = False
FORCEROTATE = False
DIST_THRESHOLD = 10
TEMPLATE_SCALE = 7.5
AREA_THRESHOLD = 0.1
PYRAMID_SCALE = 0.6
BALL_SIZE = int(2.25 * 2 * TEMPLATE_SCALE)
table_template_points = (TEMPLATE_SCALE*np.array([[0,0],[LONGSIDE,0],[LONGSIDE,SHORTSIDE],[0,SHORTSIDE]])).reshape(-1,1,2)
cueBallTemplate = cv.resize(cv.imread('testFrames/templateBallBlank.png'), (BALL_SIZE, BALL_SIZE), interpolation = cv.INTER_AREA)
BOUNCE_COUNT = 3
DOCONFIG = False
CORRELATION_FLOOR = 0.8
WINDOW = 10

low_green = np.array([25, 52, 72])
high_green = np.array([102, 255, 255])

def nothing(x):
    pass

if DOCONFIG:
    cv.namedWindow('config')
    cv.createTrackbar('H_low','config',0,255,nothing)
    cv.createTrackbar('S_low','config',0,255,nothing)
    cv.createTrackbar('V_low','config',0,255,nothing)
    cv.createTrackbar('H_high','config',0,255,nothing)
    cv.createTrackbar('S_high','config',0,255,nothing)
    cv.createTrackbar('V_high','config',0,255,nothing)

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
    if lines is None:
        return None
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
    if np.linalg.norm(sc[1] - sc[0]) < np.linalg.norm(sc[2] - sc[1]) or FORCEROTATE:
        sc = np.roll(sc, -1, axis=0)
    return sc

# Marked for removal
def drawCorners(img, corners, color = (0,255,0), radius = 5):
    for corner in corners:
        cv.circle(img, (corner[0],corner[1]), radius, color, 5)
        
def getTableMask(img):
    # Configure the bounds
    if DOCONFIG:
        low_green[0] = cv.getTrackbarPos('H_low','config')
        low_green[1] = cv.getTrackbarPos('S_low','config')
        low_green[2] = cv.getTrackbarPos('V_low','config')
        high_green[0] = cv.getTrackbarPos('H_high','config')
        high_green[1] = cv.getTrackbarPos('S_high','config')
        high_green[2] = cv.getTrackbarPos('V_high','config')
    # Masks out the green stuff in the scene. Maybe upgrade to use feature matching?
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, low_green, high_green)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.erode(mask, kernel, iterations=1)
    return mask
    
def findCornersAndTransform(img, template):
    # Finds the table, calculates the corner points, then finds the camera perspective
    outputRect = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    totalArea = img.shape[0]*img.shape[1]
    
    mask = getTableMask(img)
    if DOCONFIG:
        cv.imshow("maskDebug", mask)
        return []
    else:
        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv.contourArea(cnt)
            approx = cv.convexHull(cnt)

            if area > totalArea*AREA_THRESHOLD:
                cv.drawContours(outputRect, [approx], 0, 255, 2)

        lines = cv.HoughLinesP(outputRect,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
        filteredLines = filterLines(lines)

        if filteredLines is None:
            return []

        if len(filteredLines) != 4 or DEBUG:
            #print("Filtered {} lines down to {}".format(len(lines), len(filteredLines)))
            out = np.zeros(img.shape, dtype=np.uint8)
            for line in filteredLines:
                x1,y1,x2,y2 = line
                cv.line(out,(x1,y1),(x2,y2),(0,255,0),1)

            #print("Found {} edges".format(len(filteredLines)))
            return out

        corners = np.float32(getCorners(filteredLines)).reshape(-1,1,2)
        M, _ = cv.findHomography(corners, template, 0)

        return corners, M

def getLength(L):
    a,b,c,d = L
    return dist(a,b,c,d)

def findCue(img, bx, by):
    edge = cv.Canny(img, 100, 200)
    lines = cv.HoughLinesP(edge,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
    if lines is None:
        return None
    validLines = []
    for line in lines:
        x1,y1,x2,y2 = line[0]
        slope = (y2-y1+1e-8) / (x2-x1+1e-8)
        referenceLine = bx,by,bx+1,by-1/slope
        ix, iy = line_intersection(referenceLine, line[0])
        
        if dist(ix,iy,bx,by) < DIST_THRESHOLD:
            # Make sure the second point is the one closer to the ball
            if dist(x2,y2,bx,by) < dist(x1,y1,bx,by):
                validLines.append((x1,y1,x2,y2))
            else:
                validLines.append((x2,y2,x1,y1))
            
    if len(validLines) == 0:
        return None
    validLines.sort(key = getLength, reverse = True)
    
    return validLines[0]
    
def calculateBounce(bx,by,dx,dy):
    slope = dy/(dx+1e-8)
    sidesToBounce = []
    if dy < 0:
        sidesToBounce.append(WALLWIDTH * TEMPLATE_SCALE)
    else:
        sidesToBounce.append((SHORTSIDE - WALLWIDTH) * TEMPLATE_SCALE)
        
    if dx < 0:
        sidesToBounce.append(WALLWIDTH * TEMPLATE_SCALE)
    else:
        sidesToBounce.append((LONGSIDE - WALLWIDTH) * TEMPLATE_SCALE)
        
    # calculate the intersection with the top/bottom edge
    yDist = sidesToBounce[0] - by
    xDist = yDist * dx / (dy + 1e-8)
    ix1, iy1 = bx + xDist, sidesToBounce[0]
    dist1 = (yDist**2 + xDist**2)**0.5
    
    # calculate the intersection with the left/right edge
    xDist = sidesToBounce[1] - bx
    yDist = xDist * dy / (dx + 1e-8)
    ix2, iy2 = sidesToBounce[1], by + yDist
    dist2 = (yDist**2 + xDist**2)**0.5
    
    return (int(ix1),int(iy1), dx, -dy) if dist1 < dist2 else (int(ix2), int(iy2), -dx, dy)

# inputStream names the camera feed
def runProcess(inputStream):
    stage = 0
    calibrationOutput = []
    transform = None
    corners = None
    
    calibrationFrame = np.zeros((int(SHORTSIDE * TEMPLATE_SCALE), int(LONGSIDE * TEMPLATE_SCALE), 3))
    for point in table_template_points:
        x,y = point[0]
        cv.circle(calibrationFrame, (int(x),int(y)), 20, (255,0,0), 20)
                
    
    capture = cv.VideoCapture(inputStream)
    if not capture.isOpened:
        print('Unable to open input stream')
        return
    
    lastSureX, lastSureY = None, None
    lastDX, lastDY = None, None
    counter = 0
    while True:
        ret, img = capture.read()
        if img is None:
            break
            
        if stage == 0: # Finding the table transform
            if len(calibrationOutput) == 2:
                corners, transform = calibrationOutput
                img = cv.polylines(img,[np.int32(corners)],True,(0,255,255),10, cv.LINE_AA)
            else:
                calibrationOutput = findCornersAndTransform(img, table_template_points)
            cv.imshow("",img)
        elif stage == 1: # Calibrating the projector (serves only a semi-decorative purpose)
            cv.imshow("",calibrationFrame)
            # Project this image
        elif stage >= 2: # Main routine
            worldSpace = cv.warpPerspective(img,transform, (int(88*TEMPLATE_SCALE),int(44*TEMPLATE_SCALE)))
            ballLoc, (ballX, ballY), ballVal = locateBall(worldSpace, cueBallTemplate)
            if ballVal > CORRELATION_FLOOR:
                counter = 0
                ballX, ballY = ballX + BALL_SIZE // 2, ballY + BALL_SIZE // 2
                if lastSureX is not None:
                    lastDX, lastDY = ballX - lastSureX, ballY - lastSureY
                lastSureX, lastSureY = ballX, ballY
                cueLine = findCue(worldSpace, ballX, ballY)
                cv.circle(worldSpace, (ballX, ballY), BALL_SIZE, (0,0,255), 5)

                if cueLine is not None:
                    x1,y1,x2,y2 = cueLine
                    cv.line(worldSpace, (x1,y1), (x2,y2), (0,0,255), 10)
                    dx, dy = x2-x1,y2-y1

                    for bounce in range(BOUNCE_COUNT):
                        ix, iy, dx, dy = calculateBounce(ballX, ballY, dx, dy)
                        cv.line(worldSpace, (ballX,ballY), (ix,iy), (0,255,0), 2)
                        ballX, ballY = ix, iy
            elif lastSureX is not None and counter < 30:
                #_, max_val, _, (unsureX, unsureY) = cv.minMaxLoc(ballLoc[lastSureX-WINDOW:lastSureX+WINDOW, lastSureY-WINDOW:lastSureY+WINDOW])
                #unsureX, unsureY = unsureX + lastSureX + BALL_SIZE // 2, unsureY + lastSureY + BALL_SIZE // 2
                cv.circle(worldSpace, (lastSureX, lastSureY), BALL_SIZE, (255,255,0), 5)
                lastSureX, lastSureY = lastSureX + lastDX, lastSureY + lastDY
                if lastDX != 0 or lastDY != 0:
                    counter += 1
                
            cv.imshow("",worldSpace)
        else:
            break
            
        key = cv.waitKey(3)
        if key == 13: # press ENTER to advance stage
            stage += 1
    
    
########################  MAIN  ########################

if __name__ == "__main__":
    # Identify the name of the camera and insert it as the inputStream
    INPUT_STREAM = "testFrames/output1.mp4"
    runProcess(INPUT_STREAM)
