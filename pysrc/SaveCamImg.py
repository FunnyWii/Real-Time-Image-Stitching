import cv2
import numpy

image_path = "./imgs/"

def SaveImage():
    width = 1920
    height = 1080
    img_count = 0

    cap = cv2.VideoCapture("/dev/video0")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('m','j','p','g'))
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)

    if not cap.isOpened():
        print("Cap is not opened")
        exit()
  
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Not frame !")
            break
        print("Frame Height : ", height)
        cv2.imshow("Cam", frame)

        key = cv2.waitKey(1)  & 0xFF
        if key == ord( ' '):
            img_name = image_path +  f"{img_count}.jpg"
            cv2.imwrite(img_name, frame)
            print(f"Saved : {img_name}")
            img_count += 1
        if key == ord('q'):
            break       

if __name__ == '__main__':
    SaveImage()