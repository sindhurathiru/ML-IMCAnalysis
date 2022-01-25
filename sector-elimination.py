import image_slicer
import nrrd
import SimpleITK as sitk
import os
import sys
module_path = os.path.abspath(os.path.join('../../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from PIL import Image
import numpy as np
import pandas as pd
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import PIL.ImageShow as ImageShow 
from random import randrange


rois = ["05", "06", "07", "08", "09", "10", "11", "13","14", "15", "16", "17", "18", "21", "22", "23", "24", "27", "28", "31"]
slicedData = pd.DataFrame()

for roi in rois:
    # Get cell mask
    maskPath = "C:\\Users\\sindhura\\OneDrive - Queen's University\\Queens\\Research MSc\\Bladder Data\\SLIDE_" + slide + "\\ROI0" + roi + "_ROI_0" + roi + "\\ROI0" + roi + "_ROI_0" + roi + " Cell Mask.nrrd"
    mask = sitk.ReadImage(maskPath)
    maskArray = sitk.GetArrayFromImage(mask)
    maskArray = maskArray[0,:, :, ]

    # Get center of image
    xCenter = maskArray.shape[1]//2
    yCenter = maskArray.shape[0]//2
    center = (xCenter, yCenter)
    
    sectors = 50
    angle = 30
    count = 1

    for i in range(sectors):
        startangle=randrange(360)
        im = Image.new("1", np.transpose(maskArray).shape)
        draw = ImageDraw.Draw(im)
        draw.pieslice((-xCenter,-yCenter,maskArray.shape[1]+xCenter,maskArray.shape[0]+yCenter),startangle,startangle+angle,fill=(1))
        sliceArr = np.array(im)
        
        cellList = []
        rows = sliceArr.shape[1]
        cols = sliceArr.shape[0]

        for x in range(0, rows):
            for y in range(0, cols):
                elem = sliceArr[y,x]
                if elem == True:
                    cellLabel = maskArray[y,x]
                    if cellLabel not in cellList:
                        cellList.append(cellLabel)
        maskedArray = np.copy(maskArray)
        for cell in cellList:
            maskedArray[maskedArray == cell] = 0
        np.save(os.path.join(path, "Mask Slices", "50slice30angle", "ROI" + roi + "slice" + str(count) + ".npy"), maskedArray)
        
        data = pd.read_csv(os.path.join(path, "Clustered Data", "compensated_clustered_rawData_ROI0" + roi + "_ROI_0" + roi + ".csv"))
        df = data[~data['Cell Label'].isin(cellList)]
        nCells = len(df)
        cellTypes = df["Cell Type"].unique()
        cellDistDict = {"Slide": slide, "ROI": roi, "Slice": str(count)}

        for i in cellTypes:
            tempdf = df[df['Cell Type'].astype(str).str.contains(i, regex=False)]
            nCellType = len(tempdf)
            dist = nCellType / nCells
            cellDistDict[i] = dist
        
        slicedData = slicedData.append(cellDistDict, ignore_index=True)
        count += 1
        

        
slicedData = slicedData.groupby(level=0, axis=1).sum()
slicedData.to_csv("C:\\Users\\sindhura\\OneDrive - Queen's University\\Queens\\Research MSc\\Bladder Data\\slicedTrainDataMerged.csv")
