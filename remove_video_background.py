import numpy as np 
import cv2 
from memory_profiler import profile
  
cap = cv2.VideoCapture('video.mp4') 
fgbg = cv2.createBackgroundSubtractorMOG2() 
  
@profile
def main(): 
    while True: 
        ret, frame = cap.read() 
        fgmask = fgbg.apply(frame) 
    
        cv2.imshow('fgmask', fgmask) 
        cv2.imshow('frame',frame ) 
    
        k = cv2.waitKey(30) & 0xff
        if k == 27: 
            break
        
    cap.release() 
    cv2.destroyAllWindows() 

if __name__ == '__main__':
    main()