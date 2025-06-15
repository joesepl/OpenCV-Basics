import cv2 as cv

def display_webcam():
    camera = cv.VideoCapture(0)         # 0 is the default camera if you have multiple try 1,2, or 3 etc.
    print("Web Cam Displaying. Press q to quit.")
    while True:
        isTrue, frame = camera.read()       # Read frame.
        if isTrue:                          # If frame retreived succesfully.
            cv.imshow('Web Cam', frame)         # Show each frame. 
            if (cv.waitKey(1) == ord('q')):     # Wait for q to be pressed for 1ms then move on through loop.
                break
        else:
            print("Frame couldn't be retreived.")
            break
        
    camera.release()                    # Release camera resource for other programs to use. 
    cv.destroyAllWindows()              # Close the windows.


def main():
    display_webcam()


if __name__ == '__main__':
    main()