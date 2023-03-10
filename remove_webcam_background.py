import cv2
from memory_profiler import profile

@profile
def main():
    cap = cv2.VideoCapture(0)
    bgsub = cv2.createBackgroundSubtractorMOG2()

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame,180)
        fgmask = bgsub.apply(frame)

        cv2.imshow('Frame', frame)
        cv2.imshow('Foreground Mask', fgmask)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()






