import cv2 
import numpy as np 
from sklearn.metrics import pairwise

# set some global variables

background = None 

accumulated_weight = 0.5

roi_top = 100
roi_bottom = 600
roi_right = 300
roi_left = 900


def calc_accum_avg(frame, accumulated_weight):

    global background
    
    # for the first time set the background to the frame
    if background is None:
        background = frame.copy().astype('float')
        return None
    
    cv2.accumulateWeighted(frame, background, accumulated_weight)
    
    

def segment(frame, threshold_min=25):
    ''' 
    a function that will segment the hand region in the frame, by calculating the absolute difference, 
    calculating the threshhold, then grabbing the contours, and then grabbing the largest contour, treating that as the hand segment 
    then passing back the threshholded image and the hand segment 
    '''
    # calculate the absulote difference between the background and the passed in frame
    diff = cv2.absdiff(background.astype('uint8'), frame)
    
    
    # apply a threshhold to this image
    ret, thresholded = cv2.threshold(diff, threshold_min, 255, cv2.THRESH_BINARY)
    
    # grab the external contours from the image
    image, contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # quick check
    if len(contours) == 0:
        return None
    
    else:
        # assumin the largest extarnal contour is the hand
        hand_segment = max(contours, key=cv2.contourArea)
        
        return (thresholded, hand_segment)
    


def count_fingers(thresholded, hand_segment):
    '''
    after casting the thresholded image as well as the hand segment contour, we create the convex hull,c
    alculate the most extreme four points, calculate the center, calculate the distance from the center to all those extreme points,
    find the one with the max distance, create a circle of of that max distance, and then using that circle were gonna end up putting
    contours around everything that's outside of a certain region of that circle, and then were gonna start counting with some
    limitations (making sure that its not the wrist or noise) 
    and then retun the count of the fingers
    '''
    conv_hull = cv2.convexHull(hand_segment)
    
    # grab the extreme points 
    
    # TOP
    top    = tuple(conv_hull[conv_hull[:, :, 1].argmin()][0])
    bottom = tuple(conv_hull[conv_hull[:, :, 1].argmax()][0])
    left   = tuple(conv_hull[conv_hull[:, :, 0].argmin()][0])
    right  = tuple(conv_hull[conv_hull[:, :, 0].argmax()][0])
    
    cX = (left[0] + right[0]) // 2
    cY = (top[1] + bottom[1]) // 2
    
    distance = pairwise.euclidean_distances([(cX, cY)], Y=[left, right, top, bottom])[0]
    
    max_distance = distance.max()
    
    radius = int(0.9 * max_distance)
    circumfrence = (2*np.pi*radius)
    
    circular_roi = np.zeros(thresholded.shape[:2], dtype='uint8')
    
    cv2.circle(circular_roi, (cX, cY), radius, 255,10)
    
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)
    
    image, contours, hierarchy = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0
    
    for cnt in contours:
        
        (x,y,w,h) = cv2.boundingRect(cnt)

        out_of_wrist = (cY + (cY*0.1)) > (y+h)
        
        limit_points = ((circumfrence*0.1)) > cnt.shape[0]
        
        if out_of_wrist and limit_points:
            count += 1
            
    return count


# cam = cv2.VideoCapture(0)

# num_frames = 0

# while True:
    
#     ret, frame = cam.read()
    
#     frame_copy = frame.copy()
    
#     roi = frame[roi_top:roi_bottom, roi_right:roi_left]
    
#     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
#     gray = cv2.GaussianBlur(gray, (7,7), 0)
    
#     if num_frames < 60:
#         calc_accum_avg(gray, accumulated_weight)
        
#         if num_frames <= 59:
#             cv2.putText(frame_copy, 'Wait Getting Backgrounf', (200,300), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1,(0,0,255), 2)
#             cv2.imshow('Finger Count', frame_copy)
            
#     else:
#         hand = segment(gray)
#         if hand is not None:
#             thresholded, hand_segment = hand
#             # draws conrours around real hand in the live stream
#             cv2.drawContours(frame_copy, [hand_segment+(roi_right, roi_top)], -1, (255,0,0), 5)
            
#             fingers = count_fingers(thresholded, hand_segment)
            
#             cv2.putText(frame_copy, str(fingers), (70,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
#             cv2.imshow('Thresholded', thresholded)
            
            
#     cv2.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0,0,255), 5)
    
#     num_frames += 1
    
#     cv2.imshow('Finger counts', frame_copy)
    
#     k = cv2.waitKey(1) & 0xFF
    
#     if k == 27:
#         break
        
# cam.release()
# cv2.destroyAllWindows()