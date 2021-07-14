import cv2
import mediapipe as mp
import numpy as np
import itertools

# blurType == 0 => GaussianBlur
# blurType == 1 => white blur


class BlurFaces:
    def __init__(self, min_detection_confidence=0.5, draw_dots=False, blurType=0):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.min_detection_confidence = min_detection_confidence
        self.draw_dots = draw_dots
        self.blurType = blurType
        self.w = 0
        self.h = 0
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5)

    def create_color_blur(img):
        row, col, _ = img.shape
        mask = np.random.randint(0, 3, (row, col))
        blur_img = np.zeros((row, col, 3)) + 255.0
        for x, y in list(itertools.product(list(range(row)), list(range(col)))):
            blur_img[x][y][mask[x][y]] = 0
        return blur_img

    def findFaces(self, image, h, w):
        self.h = h
        self.w = w
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = self.face_detection.process(image)

        # Draw the face detection annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def blurFaces(self, image, results):
        if results.detections:
            for detection in results.detections:

                bb_data = detection.location_data.relative_bounding_box
                x_min = int(bb_data.xmin * self.w)
                y_min = int(bb_data.ymin * self.h)
                bb_width = int(bb_data.width * self.w)
                bb_hight = int(bb_data.height * self.h)
                col_start = x_min
                col_end = x_min+bb_width
                row_start = y_min
                row_end = y_min+bb_hight

                roi = image[row_start:row_end, col_start:col_end, :]
                h, w, _ = roi.shape
                if h > 0 and w > 0:
                    if self.blurType == 0:
                        # GaussianBlur:
                        factor = 4.0
                        kh = int(w/factor)-1 if int(w /
                                                    factor) % 2 == 0 else int(w/factor)
                        kw = int(h/factor)-1 if int(h /
                                                    factor) % 2 == 0 else int(h/factor)
                        roi = cv2.GaussianBlur(roi, (kw, kh), 0)
                        image[row_start:row_end, col_start:col_end, :] = roi
                    else:
                        # white blur:
                        roi = self.create_color_blur(roi)
                        print(roi.shape)
                    image[row_start:row_end, col_start:col_end,
                          :] = roi
                if self.draw_dots:
                    # write on the image the fave signs
                    self.mp_drawing.draw_detection(image, detection)
        return image


def main():
    # For webcam input:
    cap = cv2.VideoCapture(0)
    blFace = BlurFaces()
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        h, w, _ = image.shape
        image, results = blFace.findFaces(image, h, w)
        image = blFace.blurFaces(image, results)
        cv2.imshow('MediaPipe Face Detection', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break


if __name__ == "__main__":
    main()
