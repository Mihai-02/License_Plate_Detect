import numpy as np
import cv2

def detect_edges(init_image):
    filter_size = (3,3)
    x_filter_size = filter_size[1]
    y_filter_size = filter_size[0]

    # Convolution
    y_img_s, x_img_s = init_image.shape

    h = y_filter_size//2
    w = x_filter_size//2   

    init_image = cv2.copyMakeBorder(init_image, h, h, w, w, borderType=cv2.BORDER_REPLICATE)

    for a in range(2):
        convoluted = np.zeros((y_img_s, x_img_s))

        #Applying X and Y Sobel Kernels
        if a == 0 :
            kernel = [[-1, 0, 1], [-1, 0, 1], [-1,0,1]]
        else:
            kernel = [[1, 1, 1], [0, 0, 0], [-1,-1,-1]]

        for i in range(h, y_img_s-h):
            for j in range(w, x_img_s-w):
            
                block = init_image[i-h:i-h+y_filter_size, j-w:j-w+x_filter_size]

                convoluted[i-h][j-w] = np.sum(kernel * block)
        
        if a == 0:
            grad_x = convoluted
        else:
            grad_y = convoluted

    #Calculating the edge strength
    filtered_image_step_1 = np.sqrt(grad_x**2 + grad_y**2)

    #Calculating the gradient direction
    angle = np.arctan2(grad_y, grad_x)*180/np.pi
    angle[angle < 0] += 180

    filtered_image_step_2 = np.zeros(filtered_image_step_1.shape)
    y_img_s, x_img_s = filtered_image_step_2.shape

    #Non-maximum suppression
    for i in range(1, y_img_s-1):
        for j in range(1, x_img_s-1):
            if (angle[i][j] < 22.5 and angle[i][j] >= 0) or (angle[i][j] <= 180 and angle[i][j] >= 157.5):
                l = filtered_image_step_1[i][j-1]
                r = filtered_image_step_1[i][j+1]

            elif angle[i][j] >= 22.5 and angle[i][j] < 67.5:
                l = filtered_image_step_1[i+1][j-1]
                r = filtered_image_step_1[i-1][j+1]
        
            elif angle[i][j] >= 67.5 and angle[i][j] < 112.5:
                l = filtered_image_step_1[i-1][j]
                r = filtered_image_step_1[i+1][j]
            
            elif angle[i][j] >= 112.5 and angle[i][j] < 157.5:
                l = filtered_image_step_1[i-1][j-1]
                r = filtered_image_step_1[i+1][j+1]

            if filtered_image_step_1[i][j] > l and filtered_image_step_1[i][j] > r:
                filtered_image_step_2[i][j] = filtered_image_step_1[i][j]
            else:
                filtered_image_step_2[i][j] = 0

    # Thresholding
    filtered_image_final = np.zeros(filtered_image_step_2.shape)

    high_threshold = 140
    low_threshold = 30

    for i in range(y_img_s):
        for j in range(x_img_s):
            if filtered_image_step_2[i][j] > high_threshold:
                filtered_image_final[i][j] = 255
            elif filtered_image_step_2[i][j] > low_threshold:
                filtered_image_final[i][j] = low_threshold

    for i in range(1, y_img_s-1):
        for j in range(1, x_img_s-1):
            if filtered_image_final[i][j] == low_threshold:
                if filtered_image_final[i][j-1] == 255 or filtered_image_final[i][j+1] == 255 or filtered_image_final[i+1][j-1] == 255 or filtered_image_final[i-1][j+1] == 255 or filtered_image_final[i-1][j] == 255 or filtered_image_final[i+1][j] == 255 or filtered_image_final[i-1][j-1] == 255 or filtered_image_final[i+1][j+1] == 255:
                
                    filtered_image_final[i][j] = 255
                else:
                    filtered_image_final[i][j] = 0
                
    return filtered_image_final.astype(np.uint8)