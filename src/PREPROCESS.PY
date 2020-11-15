"""
    Initial step in the workflow, data cleaning and annotation
"""

from pathlib import Path
import os
import pandas as pd

def annotation(baseURL, output_path):

    """ Map the RGB image to its corresponding depth file, creates a mapping dataframe that can later be used for training
    """    

    subfolders = [ f.path for f in os.scandir(baseURL) if f.is_dir() ]

    nbRGB = 0
    imgID = 0
    imgTimestamp = 0
    RGBPath = ""
    nbDepth = 0
    depthPath = ""

    RGBDF = pd.DataFrame()
    DepthDF = pd.DataFrame()

    tmpRGBDict = {}
    tmpDepthDict = {}

    for idx, subfolder in enumerate(subfolders):
        print("Scanning "+subfolder+"...")
        for path in Path(subfolder).rglob('*.ppm'):
            tmp = path.name.split(".")
            if int(tmp[0].split("-")[1]) != imgID:
                # print(imgID, imgTimestamp)
                tmpRGBDict[nbRGB] = {"id": imgID, "RGBTimestamp": imgTimestamp, "RGBPath": RGBPath}
                nbRGB = nbRGB + 1
            imgID = int(tmp[0].split("-")[1])
            imgTimestamp = int(tmp[1].split("-")[0])
            RGBPath = path
        for path in Path(subfolder).rglob('*.pgm'):
            tmp = path.name.split(".")
            if int(tmp[0].split("-")[1]) != imgID:
                # print(imgID, imgTimestamp)
                tmpDepthDict[nbDepth] = {"id": imgID, "DepthTimestamp": imgTimestamp, "DepthPath": depthPath}
                nbDepth = nbDepth + 1
            imgID = int(tmp[0].split("-")[1])
            imgTimestamp = int(tmp[1].split("-")[0])
            depthPath = str(path).replace('\\','/')

    tmpDF = pd.DataFrame.from_dict(tmpRGBDict, "index")
    RGBDF = RGBDF.append(tmpDF)
    tmpDF = pd.DataFrame.from_dict(tmpDepthDict, "index")
    DepthDF = DepthDF.append(tmpDF)

    JoinDF = pd.merge(DepthDF, RGBDF, on='id')
    JoinDF = JoinDF.iloc[1:]

    JoinDF.to_csv(output_path)

def test():
    annotation("F:/NYU Data/nyu_depth_v2_raw", "JoinDF.csv")