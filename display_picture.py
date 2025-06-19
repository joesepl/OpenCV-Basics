import cv2 as cv


def display_picture(window_name, file_name, imread_mode=cv.IMREAD_COLOR_BGR):
    try:
        img = cv.imread(file_name, imread_mode)      # Read a picture taking absolute path of file as input and outputting a numpy like array. imread mode is a color code you want to use for the array.
    except:
        print("Couldn't get image file.")
        quit()

    cv.imshow(window_name, img)                      # Display the image. 
    cv.waitKey(0)                                    # Wait for a key press to resume. 


def main():
    display_picture("Guts", "Photos/Guts.jpg")


if __name__ == "__main__":
    main()