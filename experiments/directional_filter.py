import numpy as np
import matplotlib.pyplot as plt
import cv2

'''
Micah Williamson
University of Michigan
ECE 556 
Directional Filter Implementation inspired by:
[1] Liu, C., Pang, Y., Wang, J., Yang, A., Pan, J. (2014).
Frequency Domain Directional Filtering Based Rain Streaks Removal from a Single 
Color Image. In: Huang, DS., Bevilacqua, V., Premaratne, P. (eds) Intelligent 
Computing Theory. ICIC 2014. Lecture Notes in Computer Science, vol 8588. 
Springer, Cham. https://doi.org/10.1007/978-3-319-09333-8_45
'''

class DirectionalFilter:
    '''Directional Filter For Deraining'''
    def __init__(self):
        print("Direc filter")
        return
    

    def rgb_to_ycbcr(self, image):
        '''
        Convert image to YCbCr space (https://en.wikipedia.org/wiki/YCbCr)
        You can see equations for this conversion in section 2.1 of [1]

        Parameters:
        image (BGR image): Image loaded using CV2 (NOTE: Expects BGR format)

        Returns:
        Y: luminance component
        Cb: Blue difference component
        Cr: Red difference component
        '''
        Y = 16 + 0.257*image[:, :, 2]+0.564*image[:, :, 1]+0.098*image[:, :, 0]
        Cb = 128-0.148*image[:, :, 2]-0.291*image[:, :, 1]+0.439*image[:, :, 0]
        Cr = 128+0.439*image[:, :, 2]-0.368*image[:, :, 1]-0.071*image[:, :, 0]
        # image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        # # For some reason, opencv uses the ordering YCrCb rather than the more
        # # traditional YCbCr.
        # Y, Cr, Cb = cv2.split(image_ycrcb)
        # # I return in the traditional ordering of YCbCr
        return Y, Cb, Cr
    
    
    def canny_edge_detection(self, Y, low_thresh=50, high_thresh=100):
        '''
        Find edges from Y component of an image using Classic Canny Operator

        Parameters:
        Y (float image): the luminance component of an image (from rgb_to_ycbcr)
        low_thresh (int, optional): lower thresh for Canny algorithm
        high_thresh (int, optional): higher thresh for Canny algoorithm

        Returns:
        edges (image): edges of Y image
        '''
        # I haven't really looked into how Canny works but here is a good 
        # reference doc for it:
        # https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
        edges = cv2.Canny(Y, threshold1=low_thresh, threshold2=high_thresh)
        return edges

    
    def HOE(self, Y, edges):
        '''
        Histogram of Oriented Edge. This is used to determine the direction
        of the rain streaks. Useful reference: 
        https://learnopencv.com/histogram-of-oriented-gradients/


        '''    
        
        gx = cv2.Sobel(Y, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(Y, cv2.CV_32F, 0, 1, ksize=3)
        angle = np.arctan2(gy, gx) * (180 / np.pi)

    
    def apply_filter(self, img):
        return
    


def main():
    df = DirectionalFilter()
    return


if __name__=="__main__":
    main()