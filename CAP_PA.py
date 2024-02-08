
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

#The standard deviation is used as the parameter in this function, which employs the one-dimensional Gaussian Equation.

# *****************************************************Function for One Dimensional Gaussian*******************************************************************
def SingleD_gau(k, stdn_dvn): #SingleD_gau --> One dimensional Gaussian 
    k = (np.sqrt(2 * np.pi) * stdn_dvn)
    v = np.e ** (-0.5 * np.power((k) / stdn_dvn, 2))
    result = 1 / k* v
    return result

#By using standard deviation and sze as parameters, this function computes the one-dimensional gaussian kenl.

def krnl_gaus(sze, sig):# sze-->SIZE sig---> SIGMA
    l = -(sze // 2)
    p = sze // 2
    u = sze
    f = (sze,1)
    kenl = np.linspace(l, p, u)
    for i in range(sze):
        kenl[i] = SingleD_gau(kenl[i], sig)
    # kernel_xy = np.outer(kenl.T, kenl.T)
    kenl = np.reshape(kenl,f)
    kenl *= 1.0 / np.sum(kenl)
    return kenl

#This method calls the kernel gaussian function to create a gaussian kernel, which is then convolved with the image in the x and y directions.
# *****************************************************Function for applying Blur *******************************************************************

def apply_gauss_blurr_x(imag, size_of_kernel):# applying gaussian blurr on x cordinates 
    kenl = krnl_gaus(size_of_kernel, 1)
    kenl = np.transpose(kenl)
    k4enl = np.transpose(kenl)
    return apply_Convolution(imag, kenl, average=True)

def apply_gauss_blurr_y(imag, size_of_kernel): # applying gaussian blurr on y cordinates 
    kenl = krnl_gaus(size_of_kernel, 1)
    kenl2 = krnl_gaus(size_of_kernel, 2)
    return apply_Convolution(imag, kenl, average=True)

def apply_Convolution(image, kenl, average): # Convolution function starts here
    if len(image.shape) == 3:
        print("Retrieved 3 : {}".format(image.shape))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # changing the color to gray
    else:
        print("The shape will be : {}".format(image.shape))

    print("Shape of the Kernel : {}".format(kenl.shape))
    
    Row_of_Img, Col_of_Img = image.shape # Combining the Row & Col with Image shape
    Row_of_Krnl, Col_of_Krnl = kenl.shape # Combining the Row & Col with Kernel shape
    print()

    a = np.zeros(image.shape)
    Result = a # storing the image in result

# ------------------- Function to do image padding ------------------- #

# Padding is done with the images stored
    a = Row_of_Krnl - 1
    b = Col_of_Krnl - 1
    Reqd_height = int(a / 2) # Height of the padding
    Reqd_width = int(b / 2) ## Width of the padding

    f = np.zeros((Row_of_Img + (2 * Reqd_height), Col_of_Img + (2 * Reqd_width)))
    changed_image = f
    changed_image[Reqd_height:changed_image.shape[0] - Reqd_height, Reqd_width:changed_image.shape[1] - Reqd_width] = image  # Final Image after applying Padding

# Now we need to combine the X & Y cordinates of the Image with the Gaussian Kernel 
    for row in range(Row_of_Img): # Traversing inside the matrix
        for col in range(Col_of_Img):
            Result[row, col] = np.sum(kenl * changed_image[row:row + Row_of_Krnl, col:col + Col_of_Krnl])
            if average:
                Result[row, col] /= kenl.shape[0] * kenl.shape[1]

    return Result

def GD_for_X(image, sig, sze): #Gaussian Derivative of X coordinate
    l = -(sze // 2)
    p = sze // 2
    u = sze
    f = (sze,1)
    gd1_x = np.linspace(l, p, u, f) # sze--> SIZE
    c = (np.sqrt(2 * np.pi) * (sig ** 3))
    v = np.e ** (-(gd1_x ** 2) / (2 * (sig ** 2)))
    for i in gd1_x:
        gd2_x = -gd1_x / (c * v)
    gd2_x = np.reshape(gd2_x,(sze,1))
    final = apply_Convolution(image, np.transpose(gd2_x), average = True)
    return final

def GD_for_Y(image, sig, sze):  #Gaussian Derivative of Y coordinate
    l = -(sze // 2)
    p = sze // 2
    u = sze
    f = (sze,1)
    gd1_y = np.flip(np.linspace(l, p, u, f))
    c = (np.sqrt(2 * np.pi) * (sig ** 3))
    v = np.e ** (-(gd1_y ** 2) / (2 * (sig ** 2)))
    for i in gd1_y:
        gd2_y = -gd1_y / (c * v) # sig--> Sigma
    gd2_y = np.reshape(gd2_y,(sze,1))
    final = apply_Convolution(image, gd2_y, average=True)
    print(np.shape(gd2_y))
    return final

def Compute_GM_GD(GaussD_X, GaussD_Y, convert_to_degree=True): #GM- Gradient Magnitute & GD- Gradient Direction

#Calculating the value of GM
    k = np.square(GaussD_X)
    c = np.square(GaussD_Y)
    G_Mag = np.sqrt(k + c)
    G_Mag *= 255.0 / np.max(G_Mag)

#Calculating the value of GD
    G_Dir = np.arctan2(GaussD_X, GaussD_Y)
    if convert_to_degree:
        G_Dir = np.rad2deg(G_Dir)
        G_Dir += 180

    return G_Mag, G_Dir

# *****************************************************Function for NON-MAX Supression *******************************************************************

def nonmax_suppression(G_Mag, G_Dir):   #GM- Gradient Magnitute & GD- Gradient Direction
    Row_of_Img, Col_of_Img = G_Mag.shape

    k = np.zeros(G_Mag.shape)
    Supressed = k

    PI = 180

    c1 =  Row_of_Img - 1
    c2 =  Col_of_Img - 1

    for r in range(1, c1):
        for c in range(1, c2):
            Dir = G_Dir[r, c] # r here is represented as ROWS & c as Columns

            if (0 <= Dir < PI / 8) or (15 * PI / 8 <= Dir <= 2 * PI):
                Earlier = G_Mag[r, c - 1]
                Later = G_Mag[r, c + 1]

            elif (PI / 8 <= Dir < 3 * PI / 8) or (9 * PI / 8 <= Dir < 11 * PI / 8):
                Earlier = G_Mag[r + 1, c - 1]
                Later = G_Mag[r - 1, c + 1]

            elif (3 * PI / 8 <= Dir < 5 * PI / 8) or (11 * PI / 8 <= Dir < 13 * PI / 8):
                Earlier = G_Mag[r - 1, c]
                Later = G_Mag[r + 1, c]

            else:
                Earlier = G_Mag[r - 1, c - 1]
                Later = G_Mag[r + 1, c + 1]

            if G_Mag[r, c] >= Earlier and G_Mag[r, c] >= Later:
                Supressed[r, c] = G_Mag[r, c]

    return Supressed

# *****************************************************Function for Applying Thresholding *******************************************************************


def threshold(image, l, h, low_th): #low_th--> Low_Threshold
    k = np.zeros(image.shape)
    Result = k

    high_th = 22 #High_Threshold

    high_th_row, high_th_col = np.where(image >= h)
    low_th_row, low_th_col = np.where((image < h) & (image >= l))

    Result[high_th_row, high_th_col] = high_th
    Result[low_th_row, low_th_col] = low_th

    return Result

# *****************************************************Function for applying Hysteresis *******************************************************************


def Hysteresis(img, wk, str=255):
    No_of_Rows = len(img)
    No_of_Cols = len(img[0]) 
    for i in range(1, No_of_Rows-1):
        for j in range(1, No_of_Cols-1):
            if (img[i,j] == wk):
                if ((img[i+1, j-1] == str) or (img[i+1, j] == str) or (img[i+1, j+1] == str)
                    or (img[i, j-1] == str) or (img[i, j+1] == str)
                    or (img[i-1, j-1] == str) or (img[i-1, j] == str) or (img[i-1, j+1] == str)):
                    img[i, j] = str
                else:
                    img[i, j] = 0
    return img


# ******************************************************************* Main Function *****************************************************************************


if __name__ == '__main__':

    List_of_Images = ['testimage1.jpg', 'testimage2.jpg', 'testimage3.jpg']
    for pic in List_of_Images:
        image = cv2.imread(pic)

        for i in [1,10,20]:
            sig = 1* i

            image_blurred_x = apply_gauss_blurr_x(image, size_of_kernel=5) #calling the function apply_gauss_blurr_x

            image_blurred_y = apply_gauss_blurr_y(image, size_of_kernel=5) #calling the function apply_gauss_blurr_y

            GaussD_X = GD_for_X(image_blurred_x, sig, sze = 5) #calling the function GD_for_X

            GaussD_Y = GD_for_Y(image_blurred_y, sig, sze=5)#calling the function GD_for_Y

            G_Mag, G_Dir = Compute_GM_GD(GaussD_X, GaussD_Y,  convert_to_degree=True) #calling the function Compute_GM_GD

            image_non_max = nonmax_suppression(G_Mag, G_Dir) #calling the function nonmax_suppression
            low_th = 11
            image_new = threshold(image_non_max, 12, 30, low_th=low_th) #calling the function threshold
            image_final = Hysteresis(image_new, low_th) #calling the function Hysteresis

            # #Raw Image
            # plt.imshow(image, cmap='gray')
            # plt.title("(Step-1)")
            # plt.show()

            #After applying the Gaussian Kernel in the x and y Directions
            plt.imshow(image_blurred_x, cmap='gray')
            plt.title("(1)")
            plt.show()

            plt.imshow(image_blurred_y, cmap='gray')
            plt.title("(2)")
            plt.show()

            #image's X component convolved with a Gaussian derivative
            plt.imshow(GaussD_X, cmap='gray')
            plt.title("(3)")
            plt.show()

            #image's Y component convolved with a Gaussian derivative
            plt.imshow(GaussD_Y, cmap='gray')
            plt.title("(4)")
            plt.show()

            #Gradient Magnitude
            plt.imshow(G_Mag, cmap='gray')
            plt.title("(5)")
            plt.show()

            
            #nny-edge image following non-maximum suppression and hysteresis thresholding
            plt.imshow(image_non_max, cmap='gray')
            plt.title("(6)")
            plt.show()

            #nny-edge image following non-maximum suppression and hysteresis thresholding
            plt.imshow(image_final, cmap='gray')
            plt.title("(7)")
            plt.show()


