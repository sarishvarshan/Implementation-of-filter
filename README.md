# Implementation-of-filter
## Aim:
To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:
Anaconda - Python 3.7

## Algorithm:

1.Import the required libraries.

2.Convert the image from BGR to RGB.

3.Apply the required filters for the image separately.

4.Plot the original and filtered image by using matplotlib.pyplot.

5.End of Program

## Program:
### Developed By   : SARISH VARSHAN V
### Register Number: 212223230196
</br>

### 1. Smoothing Filters

i) Using Averaging Filter
```Python

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

img_path = r"C:\Users\admin\Downloads\bird.jpg"  # Change this to your correct path

if not os.path.exists(img_path):
    print(" Image not found. Check the file path.")
else:
    image1 = cv2.imread(img_path)
    if image1 is None:
        print(" Image could not be loaded (possibly corrupted or unsupported format).")
    else:
        image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        kernel = np.ones((11, 11), np.float32) / 169
        image3 = cv2.filter2D(image2, -1, kernel)

        plt.figure(figsize=(9, 9))
        plt.subplot(1, 2, 1)
        plt.imshow(image2)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(image3)
        plt.title("Average Filter Image")
        plt.axis("off")
        plt.show()




```
<h2>OUTPUT</h2>

![image](https://github.com/user-attachments/assets/410fef39-ea5d-4560-ae92-462ae8569167)





ii) Using Weighted Averaging Filter
```Python
kernel1=np.array([[1,2,1],[2,4,2],[1,2,1]])/16
image3=cv2.filter2D(image2,-1,kernel1)
plt.imshow(image3)
plt.title("Weighted Average Filter Image")
plt.axis("off")
plt.show()





```
<h2>OUTPUT</h2>

![image](https://github.com/user-attachments/assets/bb2e0a2d-7990-40ad-a0ea-d6fc5dacde22)




iii) Using Gaussian Filter
```Python


gaussian_blur=cv2.GaussianBlur(image2,(33,33),0,0)
plt.imshow(gaussian_blur)
plt.title("Gaussian Blur")
plt.axis("off")
plt.show()







```
<h2>OUTPUT</h2>

![image](https://github.com/user-attachments/assets/b1c46039-85ba-4558-b9ed-371f712b565b)



iv)Using Median Filter
```Python


median = cv2.medianBlur(image2, 13)
plt.imshow(cv2.cvtColor(median, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display
plt.title("Median Blur")
plt.axis("off")
plt.show()



```
<h2>OUTPUT</h2>

![image](https://github.com/user-attachments/assets/52c62fad-c41b-480a-99ff-4f72f5e5a30a)



### 2. Sharpening Filters
i) Using Laplacian Linear Kernal
```Python

kernel2=np.array([[-1,-1,-1],[2,-2,1],[2,1,-1]])
image3=cv2.filter2D(image2,-1,kernel2)
plt.imshow(image3)
plt.title("Laplacian Kernel")
plt.axis("off")
plt.show()

```
<h2>OUTPUT</h2>


![image](https://github.com/user-attachments/assets/173eccdf-9ce2-4409-a9fc-bae48ad3b292)



ii) Using Laplacian Operator
```Python



laplacian=cv2.Laplacian(image2,cv2.CV_64F)
plt.imshow(laplacian)
plt.title("Laplacian Operator")
plt.axis("off")
plt.show()


```
<h2>OUTPUT</h2>


![image](https://github.com/user-attachments/assets/18332f09-4d82-465b-9baa-cc2495a72629)


</br>

## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
