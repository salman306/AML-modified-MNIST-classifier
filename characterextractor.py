import pandas as pd
import numpy as np
import cv2
import heapq
import matplotlib.pyplot as plt

class characterExtractor:

    WHITE = 255

    SOFT_THRESH = 205
    HARD_THRESH = 254

    MAX_WHITE_PX_THRESH = 730

    MAX_CONTOURS_THRESH = 10

    CHAR_PERIMETER_THRESH = 15
    CHAR_AREA_THRESH = 19

    def __init__(self, data_csv=None, pd_data=None):
        """ Expecting train_x or test_x csv file """

        if len(pd_data.index):
            self.data = pd_data
        else:
            self.data = pd.read_csv(data_csv, header = None)

    def extractCharacters(self, x_serie, draw = None):
        """ Extract all characters from img """
        global img_og
        img_og = x_serie.values.reshape(64,64).astype(np.uint8)
        img = img_og.copy()

        #apply first soft threshold
        _ , img = cv2.threshold(img, self.SOFT_THRESH, self.WHITE, cv2.THRESH_BINARY)


        #apply second hard threshold for bright patterns 
        img_white_px = len(img[img > self.SOFT_THRESH])
        if img_white_px > self.MAX_WHITE_PX_THRESH:
            _ , img = cv2.threshold(img_og.copy(), self.HARD_THRESH, self.WHITE, cv2.THRESH_BINARY)

        #apply median blur for the rice patterns
        _ , contours , _ = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > self.MAX_CONTOURS_THRESH: 
            img = cv2.medianBlur(img, 3)
            _ , contours , _ = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        #extract potential character contours
        c = self.findPotentialContours(contours, img, draw)
        (img_c1, img_c2, img_c3) = self.processPotentialContours(c, img)
        
        return (img_c1, img_c2, img_c3)
        #should just return the images
        #instead of plotting
        
    def findPotentialContours(self, contours, img, draw=None):
        """ Takes all contours and filters based on area and perimeter to find best contours"""
        
        if draw:
            global img_contour
            img_contour = img.copy()

        c = []
        for i, cnt in enumerate(contours):
            perimeter = cv2.arcLength(cnt, True)
            area = cv2.contourArea(cnt)
            if perimeter > self.CHAR_PERIMETER_THRESH and area > self.CHAR_AREA_THRESH:
                heapq.heappush(c, (-perimeter, i, cnt))

                if draw:
                    x,y,w,h = cv2.boundingRect(cnt)
                    cv2.rectangle(img_contour, (x,y), (x+w,y+h), 120,1)
                
        return c

    def processPotentialContours(self, c, img, d=None):
        """ Dispatches the problem based of on the number of contours found
                resulting from different background patterns. """
        num_chars = len(c)
        
        if num_chars > 3:
            print(">3")
            #restart image processing by dilating the characters font size
            if not d:
                d = 2
            img = img_og.copy()
            img = cv2.dilate(img,np.ones((d,d),np.uint8),iterations=1)
            _, img = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)
            _ , contours , _ = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            
            new_c = self.findPotentialContours(contours, img)
            return self.processPotentialContours(new_c,img,d=d+1)
            
        elif num_chars == 2:
            print("CUT MIDDLE")
            img_c1,img_c2,img_c3 = self.getCharactersAfterMiddleCut(c, img)
            return (img_c1,img_c2,img_c3)
        elif num_chars < 2:
            print("<2")
            if not d: # Once: dilate the characters font size to filter false positives
                d = 3
                img = img_og.copy()
                img = cv2.dilate(img,np.ones((d,d),np.uint8),iterations=1)
                _, img = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)
                _ , contours , _ = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                new_c = self.findPotentialContours(contours, img)
                return self.processPotentialContours(new_c,img,d=d+1)
            else:
                #CUT IN MIDDLE
                #AND CUT IN MIDDLE AGAIN
                print("TRUE <2")
                the_only_contour = heapq.heappop(c)[2]
                cnt1, img_cnt1, cnt2, img_cnt2 = self.cutImgInMiddle(the_only_contour, img)
                
                area_cnt1 = cv2.contourArea(cnt1)
                area_cnt2 = cv2.contourArea(cnt2)
                
                if area_cnt1 < area_cnt2:
                    biggest_sliced_contour = cnt2
                    cnt2, img_cnt2, cnt3, img_cnt3 = self.cutImgInMiddle(biggest_sliced_contour, img_cnt2)
                else:
                    biggest_sliced_contour = cnt1
                    cnt1, img_cnt1, cnt3, img_cnt3 = self.cutImgInMiddle(biggest_sliced_contour, img_cnt1)
                    
                c1 = cv2.boundingRect( cnt1 )
                c2 = cv2.boundingRect( cnt2 )
                c3 = cv2.boundingRect( cnt3 )

                img_c1 = self.getImgFromContour(c1, img_cnt1)
                img_c2 = self.getImgFromContour(c2, img_cnt2)
                img_c3 = self.getImgFromContour(c3, img_cnt3)
                    
                return (img_c1, img_c2, img_c3)

        else:
            print("PERFECT")
        #             print(c)
            cnt1 = heapq.heappop(c)[2]
            cnt2 = heapq.heappop(c)[2]
            cnt3 = heapq.heappop(c)[2]

            c1 = cv2.boundingRect( cnt1 )
            c2 = cv2.boundingRect( cnt2 )
            c3 = cv2.boundingRect( cnt3 )

            img_c1 = self.getImgFromContour(c1, img)
            img_c2 = self.getImgFromContour(c2, img)
            img_c3 = self.getImgFromContour(c3, img)

            return (img_c1, img_c2, img_c3)
            
    def getCharactersAfterMiddleCut(self, c, img):
        """ Processes the case where we have one character but the two other are capture in a single contour
                by cutting in the middle of the highest width or height and returns the 3 characters.
        """
        biggest_contour = heapq.heappop(c)[2]
        
        cnt1, img_cnt1, cnt2, img_cnt2 = self.cutImgInMiddle(biggest_contour, img)
        
        c1 = cv2.boundingRect( cnt1 )
        c2 = cv2.boundingRect( cnt2 )
        c3 = cv2.boundingRect( heapq.heappop(c)[2] )
        
        img_c1 = self.getImgFromContour(c1,img_cnt1)
        img_c2 = self.getImgFromContour(c2,img_cnt2)
        img_c3 = self.getImgFromContour(c3,img)
               
        return (img_c1,img_c2,img_c3)
    
    def cutImgInMiddle(self, contour, img):
        """ Cuts the image in the middle of the highest width or height
                returns the separated characters 
        """
        x,y,w,h = cv2.boundingRect(contour)
        
        if w > h: # width bigger than height, cut in middle of width
            m = w//2
            c1 = (x  , y, m, h)
            c2 = (x+m, y, m, h)
        else:
            m = h//2
            c1 = (x, y  , w, m)
            c2 = (x, y+m, w, m)
        
        #in case characters are in diagonal of each other
        img_c1 = self.getImgFromContour(c1,img)
        img_c2 = self.getImgFromContour(c2,img)
        
        _ , contours_c1 , _ = cv2.findContours(img_c1, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        _ , contours_c2 , _ = cv2.findContours(img_c2, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        cnt1 = max(contours_c1, key = cv2.contourArea)
        cnt2 = max(contours_c2, key = cv2.contourArea)
        
        return (cnt1, img_c1, cnt2, img_c2)
        
    
    def getImgFromContour(self, cnt_bound, img_copy):
        """ returns an np array from a contour's bounds """
        x,y,w,h = cnt_bound
        #print(x,y,w,h)
        return img_copy[y:y+h, x:x+w]

    def extractCharactersFrom(self, choice):
        x_serie = self.data.iloc[choice]
        self.extractCharacters(x_serie)

    def extractCharactersFromAndPlot(self, choice):
        x_serie = self.data.iloc[choice]
        char_imgs = self.extractCharacters(x_serie, draw = True)
        char_imgs = (img_og,) + (img_contour,) + char_imgs

        plt.figure(figsize=(10,10))
        m_size = (1,len(char_imgs))
        for i,img in enumerate(char_imgs):
            h,w = img.shape
            self.plot_img(img, str(w) + "x" + str(h), m_size,(i+1))
        plt.show()

    def plot_img(self,img, title, size, pos):
        plt.subplot(size[0], size[1], pos), plt.imshow(img, cmap="gray")
        plt.title(title), plt.xticks([]), plt.yticks([])
