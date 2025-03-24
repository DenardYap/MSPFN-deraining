import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import kaiserord, firwin
from numpy.polynomial.chebyshev import Chebyshev

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
    def __init__(self, minCanny=150, maxCanny=220):
        self.minCanny = minCanny
        self.maxCanny = maxCanny
        print("Direc filter")
        return
    

    def bgr_to_ycbcr(self, image):
        '''
        Convert BGR (Not RGB) image to YCbCr space 
        (https://en.wikipedia.org/wiki/YCbCr)
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
        return Y.astype(np.uint8), Cb.astype(np.uint8), Cr.astype(np.uint8)
    
    
    def canny_edge_detection(self, Y, low_thresh=150, high_thresh=220):
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

        # TODO: Investigate thresholds. No idea how to determine the optimal
        # min/max values. Could set up some optimization problem maybe
        edges = cv2.Canny(Y, threshold1=low_thresh, threshold2=high_thresh)
        return edges

    
    def HOE(self, edges):
        '''
        Histogram of Oriented Edge. This is used to determine the direction
        of the rain streaks. Useful reference: 
        https://learnopencv.com/histogram-of-oriented-gradients/


        '''    
        gx = cv2.Sobel(edges, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(edges, cv2.CV_32F, 0, 1, ksize=3)
        # mags, angles = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        # angles = (angles + 90) % 180
        mags = np.sqrt(gx ** 2 + gy ** 2)
        angles = ((np.arctan2(gy, gx)+90) * (180 / np.pi)) % 180
        # edge_orientations = angles[edges > 0]
        # histogram, bins = np.histogram(edge_orientations.flatten(), bins=8, range=(0, 180))

        num_bins = 8
        bin_width = 180 / num_bins
        histogram = np.zeros(num_bins, dtype=np.float64)
        angles_flat = angles.flatten()
        magnitudes_flat = mags.flatten()

        for angle, mag in zip(angles_flat, magnitudes_flat):
            # Algorithm per ChatGPT
            bin_idx = angle / bin_width
            lower_bin = int(np.floor(bin_idx)) % num_bins
            upper_bin = (lower_bin + 1) % num_bins
            fractional_part = bin_idx - lower_bin
            

            histogram[lower_bin] += mag * (1 - fractional_part)
            histogram[upper_bin] += mag * fractional_part 

        # # Uncomment to print histogram
        # for i, val in enumerate(histogram):
        #     print(f'Bin {i}: [{i*bin_width:.1f}-{(i+1)*bin_width:.1f}) degrees -> {val:.2f}')

        top_3_bins = np.argsort(histogram)[-3:]
        return top_3_bins
            


    def apply_filter(self, img):
        M, N = img.shape[0:2]
        # 1. Convert to YCbCr space
        Y, Cb, Cr = self.bgr_to_ycbcr(img)
        # 2. Use classic Canny edge detection on the Y (luma) component
        edges = self.canny_edge_detection(Y, self.minCanny, self.maxCanny)
        # 3. Break edges and Y into 16 blocks
        block_M = M // 4
        block_N = N // 4
        edge_blocks = [edges[x:x+block_M,y:y+block_N] for x in range(0,M,block_M) \
                       for y in range(0,N,block_N)]
        # #Look at a couple of blocks:
        # block_samples = np.hstack((edge_blocks[0],edge_blocks[5]))
        # cv2.imwrite('block_samples.png', block_samples)
        # 4. Process each block and compute HOE if it is not "clean"
        
        interval_nums = []
        for edge_blk in edge_blocks:
            # Don't process a block if it is "clean" (i.e. low proportion of
            # edges in the block)
            proportion_edges = np.count_nonzero(edge_blk) / (block_M*block_N)
            #TODO: Determine what the proportion bound is supposed to be
            if proportion_edges > 0.01:
                top_3_bins = self.HOE(edge_blk)
                interval_nums.append(top_3_bins)
        # 5. Compute global salient histogram of oriented edge, find largest
        # value to estimate rain direction
        GHIST = np.zeros(8)
        for interval in interval_nums:
            GHIST[interval] += 1
        top_interval = np.argsort(GHIST)[-1]
        
        # 6. Compute 2D FFT Of Y
        I = np.fft.fft2(Y)
        F = np.abs(I)
        P = np.angle(I)



def main():
    df = DirectionalFilter()
    # im = cv2.imread('/Users/micahwilliamson/code/ECE556/MSPFN-deraining/experiments/rain_images/heavy/2.jpg_rain.png')
    im = cv2.imread('/Users/micahwilliamson/code/ECE556/MSPFN-deraining/experiments/giraffe_rain.png')
    df.apply_filter(im)
    return


if __name__=="__main__":
    main()