from general import *


cam = cv2.VideoCapture(0)

num_frames = 0

while True:
    
    ret, frame = cam.read()
    
    frame_copy = frame.copy()
    
    roi = frame[roi_top:roi_bottom, roi_right:roi_left]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    gray = cv2.GaussianBlur(gray, (7,7), 0)
    
    if num_frames < 60:
        calc_accum_avg(gray, accumulated_weight)
        
        if num_frames <= 59:
            cv2.putText(frame_copy, 'Wait Getting Backgrounf', (200,300), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1,(0,0,255), 2)
            cv2.imshow('Finger Count', frame_copy)
            
    else:
        hand = segment(gray)
        if hand is not None:
            thresholded, hand_segment = hand
            # draws conrours around real hand in the live stream
            cv2.drawContours(frame_copy, [hand_segment+(roi_right, roi_top)], -1, (255,0,0), 5)
            
            fingers = count_fingers(thresholded, hand_segment)
            
            cv2.putText(frame_copy, str(fingers), (70,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            cv2.imshow('Thresholded', thresholded)
            
            
    cv2.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0,0,255), 5)
    
    num_frames += 1
    
    cv2.imshow('Finger counts', frame_copy)
    
    k = cv2.waitKey(1) & 0xFF
    
    if k == 27:
        break
        
cam.release()
cv2.destroyAllWindows()
    