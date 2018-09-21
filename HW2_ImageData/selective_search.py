import cv2

#Read image and only need rgb channel
img = cv2.imread('./Images/2007_000464.jpg')
def selectiveSearch():
    #Create SS object 
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    #add the segmentation to genenrate the initail regions R, and set threshold to 100.
    basic_segmentation=cv2.ximgproc.segmentation.createGraphSegmentation(k=100)
    ss.addGraphSegmentation(basic_segmentation)
    #add the img need to be processed
    ss.addImage(img)
    #Add strategy(only color)
    addStrategy(True, ss)
    #Get the output bounding boxes of only color strategy
    colorResult = ss.process()
    #clear the strategy
    ss.clearStrategies()
    #add strategy(all similarities)
    addStrategy(False, ss)
    #Get the output bounding boxes of only all similarities strategy
    allResult = ss.process()
    proposals = 100
        # create a copy of original image
    imOut = img.copy()
 
        # itereate over all the region proposals
    for i, rect in enumerate(colorResult):
            # draw rectangle for region proposal till 100
        if (i < proposals):
            x, y, w, h = rect
            cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
        else:
            break
 
        # show output
    cv2.imshow("Output", imOut)
 
        # record key press
    k = cv2.waitKey(0) & 0xFF
 
        # m is pressed
  #      if k == 109:
            # increase total number of rectangles to show by increment
  #          numShowRects += increment
        # l is pressed
  #      elif k == 108 and numShowRects > increment:
            # decrease total number of rectangles to show by increment
  #          numShowRects -= increment
        # q is pressed
  #      elif k == 113:
  #          break
    print(colorResult)
    print(allResult)
def addStrategy(onlyColor, ss):
    strategy = None
    if(onlyColor):
        #only color
        strategy = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyColor()
    else:
        #all similarities
        color = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyColor()
        fill = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyFill()
        size = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategySize()
        texture	= cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyTexture()
        strategy = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyMultiple(color, fill, size, texture)
    ss.addStrategy(strategy)

selectiveSearch()