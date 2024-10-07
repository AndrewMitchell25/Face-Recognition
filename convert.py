import cv2
import os

cropping = False
start_x, start_y = 0, 0
fixed_width, fixed_height = 384, 286 
image = None
image_copy = None
name = "Andrew"
counter = 65

def crop(event, x, y, flags, param):
    global start_x, start_y, cropping, image_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        start_x, start_y = x, y
        cropping = True

        end_x = start_x + fixed_width
        end_y = start_y + fixed_height

        end_x = min(end_x, image.shape[1])
        end_y = min(end_y, image.shape[0])

        image_copy = image.copy()
        cv2.rectangle(image_copy, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        cv2.imshow("Image", image_copy)


image_dir = 'data'  

for filename in os.listdir(image_dir):
    if filename.endswith((".jpg", ".JPEG")):
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Unable to read {filename}")
            continue

        image_copy = image.copy()

        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", crop)

        while True:
            cv2.imshow("Image", image_copy)

            key = cv2.waitKey(1) & 0xFF

            if key == 13:
                end_x = min(start_x + fixed_width, image.shape[1])
                end_y = min(start_y + fixed_height, image.shape[0])
                if start_x != end_x and start_y != end_y:
                    cropped_image = image[start_y:end_y, start_x:end_x]
                    cv2.imshow("Cropped", cropped_image)

                    cropped_image_path = os.path.join(image_dir, name + str(counter) + ".pgm")
                    counter += 1

                    with open(cropped_image_path, 'wb') as f:
                        f.write(f"P5\n{fixed_width}\n{fixed_height}\n255\n".encode('ascii'))  
                        cropped_image.tofile(f)  

                    print(f"Cropped image saved as {cropped_image_path}")
                    break

            elif key == ord('q'):  
                cv2.destroyAllWindows()
                exit()  
