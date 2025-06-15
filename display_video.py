import cv2 as cv

def display_video(window_title, filename):
    try:
        capture = cv.VideoCapture(filename)    # Here you can provide an integer for a camera with the default being 0. Or you can provide a full path to a video file. 
    except:
        print("Error getting camera or video file.")
    print("Video playing press q to quit.")
    while True:
        isTrue, frame = capture.read()      # Returns the actual frame of the video, and whether it captured it succesffuly (boolean). Iterates through each frame of the video.
        if (isTrue):
            cv.imshow(window_title, frame)
            if cv.waitKey(20) == ord('q'):      # Waits for 20 ms each loop for a key then returns -1 if no key pressed or the ascii number for pressed key. In this case q will quit the program.
                break
        else:
            break

    capture.release()                       # Release camera resource.
    cv.destroyAllWindows()                  # Close opencv windows.


def main():
    display_video('Video', 'Photos/dog_vid.mp4')


if __name__ == "__main__":
    main()
