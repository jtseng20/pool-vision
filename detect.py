import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

DIST_THRESHOLD = 10
TEMPLATE_SCALE = 1.5

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

def isInLine(L1,L2):
    x1,y1,x2,y2 = L1
    x3,y3,x4,y4 = L2
    
    mx1, my1 = (x1+x2)/2, (y1+y2)/2
    mx2, my2 = (x3+x4)/2, (y3+y4)/2
    
    slope1 = (y2-y1)/(x2-x1+1e-8)
    slope2 = (my2-my1)/(mx2-mx1+1e-8)
    
    if abs(slope2/slope1 - 1) < 0.2 or abs(slope2 - slope1) < 0.05:
        return True
    
    slope3 = (y4-y3)/(x4-x3+1e-8)
    return dist(mx2,my2,mx1,my1) < DIST_THRESHOLD and abs(slope3/slope1 - 1) < 0.2

def allFalse(l, ll):
    for u in ll:
        if isInLine(l, u):
            return False
    return True

def combineLines(L1, L2):
    x1,y1,x2,y2 = L1
    x3,y3,x4,y4 = L2
    slope = (y2-y1)/(x2-x1+1e-8)
    
    # steep line, minimize/maximize y
    if abs(slope) > 1:
        minStart = (x1,y1) if y1 < y3 else (x3,y3)
        maxEnd = (x2,y2) if y2 > y4 else (x4,y4)
        return np.array([minStart[0],minStart[1], maxEnd[0],maxEnd[1]])
    
    minStart = (x1,y1) if x1 < x3 else (x3,y3)
    maxEnd = (x2,y2) if x2 > x4 else (x4,y4)
    return np.array([minStart[0],minStart[1], maxEnd[0],maxEnd[1]])

def getAbsSlope(L):
    x1,y1,x2,y2 = L
    return abs((y2-y1)/(x2-x1))
    
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

def line_intersection(A: np.ndarray, B: np.ndarray) -> np.ndarray:

    # Intersection point between lines
    out = np.zeros(2)
    
    x1, y1, x2, y2 = A
    x3, y3, x4, y4 = B
    
    out[0] = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
    out[1] = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))

    assert out.shape == (2,)
    assert out.dtype == np.float

    return int(out[0]),int(out[1])

def getCorners(lines):
    return np.array([line_intersection(lines[0], lines[2]),line_intersection(lines[0], lines[3]),line_intersection(lines[1], lines[2]),line_intersection(lines[1], lines[3])])

def drawCorners(img, corners, color = (0,255,0), radius = 5):
    for corner in corners:
        cv.circle(img, (corner[0],corner[1]), radius, color, 5)
        

#############################################################

img4 = cv.imread('testFrames/table9.jpg')
outputRect = np.zeros((img4.shape[0], img4.shape[1]), dtype=np.uint8)
hsv = cv.cvtColor(img4, cv.COLOR_BGR2HSV)

low_green = np.array([25, 52, 72])
high_green = np.array([102, 255, 255])
green_mask = cv.inRange(hsv, low_green, high_green)
green = cv.bitwise_and(img4, img4, mask=green_mask)

mask = cv.inRange(hsv, low_green, high_green)
kernel = np.ones((5, 5), np.uint8)
mask = cv.erode(mask, kernel, iterations=1)


contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv.contourArea(cnt)
    approx = cv.convexHull(cnt)#cv.approxPolyDP(cv.convexHull(cnt), 0.05*cv.arcLength(cnt, True), True)

    if area > 10000:
        cv.drawContours(outputRect, [approx], 0, 255, 1)
        
#smallkernel = np.ones((3,3))
#outputRect = cv.erode(outputRect, smallkernel, iterations=3)
#edges = cv.Canny(outputRect,50,150,apertureSize = 3)

lines = cv.HoughLinesP(outputRect,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
filteredLines = filterLines(lines)

print("Filtered {} lines down to {}".format(len(lines), len(filteredLines)))
assert(len(filteredLines) >= 4), "That ain't enough sides!!!"
    
for line in filteredLines:
    x1,y1,x2,y2 = line
    cv.line(img4,(x1,y1),(x2,y2),(0,255,0),2)
    
corners = getCorners(filteredLines)
drawCorners(img4, corners)

#table_template_points = np.transpose(TEMPLATE_SCALE*np.array([[0,0,1],[515,0,1],[0,295,1],[515,295,1]]))
#homogeneous_corners = np.transpose(np.hstack((corners, np.ones((corners.shape[0],1)))))

table_template_points = np.transpose(TEMPLATE_SCALE*np.array([[0,0],[515,0],[0,295],[515,295]]))
drawCorners(img4, np.transpose(table_template_points.astype(int)), (255,0,0))
affine_corners = np.transpose(corners)

affine_transform = np.linalg.lstsq(affine_corners, table_template_points, rcond=None)[0]
transformedPoints = np.transpose(affine_corners @ affine_transform).astype(int)
drawCorners(img4, transformedPoints, (0,0,255), 10)

plt.imshow(cv.cvtColor(img4,cv.COLOR_BGR2RGB))
plt.title('Image'), plt.xticks([]), plt.yticks([])
plt.show()

'''
img2 = cv.imread('table.png', 0)
img1 = cv.imread("template.jpg", 0)
MIN_MATCH_COUNT = 100


# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)


FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

search_params = dict(checks = 50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)
# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if True:#m.distance < 0.7*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
    print(dst)
    img2 = cv.polylines(img2,[np.int32(dst)],True,(0,0,0),3, cv.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None
    
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

#img3=cv.drawKeypoints(img1,kp1,img1,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(img3),plt.show()
'''
