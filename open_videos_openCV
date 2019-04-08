import cv2
import numpy as np

cap = cv2.VideoCapture('Desktop/Rachel/CeVICHE/sample_analysed_video.mp4')
cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame', 800,800)

frames = []

while(cap.isOpened()):
    ret, frame = cap.read()
    #print(frame.shape)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frames = np.append(frames, frame)

cap.release()
cv2.destroyAllWindows()
