import os
import sys
module_path = os.path.abspath(os.path.join('../../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import phate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%load_ext autoreload
%autoreload 2
%matplotlib inline
import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import phenograph
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import statistics
import seaborn as sns
import SimpleITK as sitk
import myshow
from sklearn.preprocessing import normalize
from scipy.stats.mstats import winsorize
from pylab import savefig

def readData(roi, roiname, compPath, slide):
  # Read in compensated data file
  filename = os.path.join(compPath, "SLIDE_" + slide + "_ROI_0" + roi + "_" + roi + "_compensated.txt")
  compData = pd.read_csv(filename, sep='\t', lineterminator='\r')
  # Drop last NaN row of df
  compData = compData[:-1]
  # Get cell array and attach to compData
  cellMask = sitk.ReadImage("C:\\Users\\sindhura\\OneDrive - Queen's University\\Queens\\Research MSc\\Bladder Data\\Slide_" + slide + "\\" + roiname +"\\" + roiname + " Cell Mask.nrrd")
  cellArray = sitk.GetArrayFromImage(cellMask)
  cellArray = cellArray.flatten()
  if cellArray.shape[0] != len(compData):
      return None
  compData["Cell Label"] = cellArray
  compData = compData.drop(["Start_push", "End_push", "Pushes_duration", "X", "Y", "Z"], 1)
  cellCompData = pd.DataFrame()
  cellLabels = np.unique(cellArray)
  cellLabels = np.delete(cellLabels, [0])
  for i in cellLabels:
      rows = compData.loc[compData['Cell Label'] == i]
      rowMeans = rows.mean(axis=0)
      cellCompData = cellCompData.append(rowMeans, ignore_index=True)
  cellCompData.to_csv(os.path.join(compPath, "compensated_slide" + slide + roiname + ".csv"))
  return cellCompData

def readDataNoncomp(roi, roiname, path, slide):
  # Read in non-compensated data file
  filename = os.path.join(path, "rawData_ROI0" + roi + "_ROI_0" + roi + ".csv")
  data = pd.read_csv(filename)
  data = data.drop(["ROI", "Cell Label", "190BCKG_190BCKG.ome.tiff", "191Ir_191Ir-DNA1.ome.tiff", "193Ir_193Ir-DNA2.ome.tiff"], 1)
  return data

def getPhenographData(cellCompData):
  dataArray = np.array(cellCompData)
  logData = cellCompData.transform(lambda x: np.log(x+1))
  # 99th percentile normalization
  normData = logData.copy()
  for col in logData:
      arr = np.array(logData[col])
      percentile = np.percentile(arr, 99)
      normArr = arr/percentile
      normData[col] = normArr
  normDataArray = np.array(normData)
  phenographData = normData[['141Pr_alpha-actin(Pr141Di)', '143Nd-Vimentin(Nd143Di)', "147Sm-CD163(Sm147Di)", "148Nd-PANCK(Nd148Di)",
              "150Nd-PDL1(Nd150Di)", "151Eu-GATA3(Eu151Di)", "152Sm-CD45(Sm152Di)", "155Gd-FOXP3(Gd155Di)",
              "156Gd-CD4(Gd156Di)", "158Gd-CD11c(Gd158Di)", "159Tb-CD68(Tb159Di)", "161Dy-CD20(Dy161Di)",
              "162Dy-CD8a(Dy162Di)", "165Ho-PD1(Ho165Di)", "167Er-GRANZB(Er167Di)", "168Er-KI67(Er168Di)",
              "169Tm-DCLAMP(Tm169Di)", "170Er-CD3(Er170Di)", "173Yb-CD45RO(Yb173Di)", "174Yb-HLA-DR(Yb174Di)"]]
  phenographDataArray = np.array(phenographData)
  return phenographData, phenographDataArray

def getPhenographDataNoncomp(data):
  dataArray = np.array(data)
  logData = data.transform(lambda x: np.log(x+1))
  # 99th percentile normalization
  normData = logData.copy()
  for col in logData:
      arr = np.array(logData[col])
      percentile = np.percentile(arr, 99)
      normArr = arr/percentile
      normData[col] = normArr
  normDataArray = np.array(normData)
  phenographData = normData[['141Pr_141Pr_alpha-actin.ome.tiff', '143Nd_143Nd-Vimentin.ome.tiff', "147Sm_147Sm-CD163.ome.tiff", "148Nd_148Nd-PANCK.ome.tiff",
              "150Nd_150Nd-PDL1.ome.tiff", "151Eu_151Eu-GATA3.ome.tiff", "152Sm_152Sm-CD45.ome.tiff", "155Gd_155Gd-FOXP3.ome.tiff",
              "156Gd_156Gd-CD4.ome.tiff", "158Gd_158Gd-CD11c.ome.tiff", "159Tb_159Tb-CD68.ome.tiff", "161Dy_161Dy-CD20.ome.tiff",
              "162Dy_162Dy-CD8a.ome.tiff", "165Ho_165Ho-PD1.ome.tiff", "167Er_167Er-GRANZB.ome.tiff", "168Er_168Er-KI67.ome.tiff",
              "169Tm_169Tm-DCLAMP.ome.tiff", "170Er_170Er-CD3.ome.tiff", "173Yb_173Yb-CD45RO.ome.tiff", "174Yb_174Yb-HLA-DR.ome.tiff"]]
  phenographDataArray = np.array(phenographData)
  return phenographData, phenographDataArray

def runPhenograph(phenographData, path):
  pca30 = PCA(n_components=20, copy=True).fit_transform(phenographData)
  communities, graph, Q = phenograph.cluster(phenographData, k=15)
  # Save communities array
  np.save(os.path.join(path, "Community arrays", "compensated_slide" + slide + "roi" + roi + "clusters.npy"), communities)
  return communities

def calcMeanIntensities(communities, phenographData, phenographDataArray):
  meanIntensities = np.full((len(np.unique(communities)), len(phenographData.columns)), 0.00)
  for clus in np.unique(communities):
      cells = np.where(communities==clus)
      for channel in range(len(phenographData.columns)):
          sumIntens = []
          for i in cells[0]:
              intensVal = phenographDataArray[i][channel]
              sumIntens.append(intensVal)
          medianIntens = statistics.median(sumIntens)
          meanIntensities[clus][channel] = medianIntens
  # Normalize by row
  count = 0
  for i in meanIntensities:
      norm = np.interp(i, (i.min(), i.max()), (0, 1))
      meanIntensities[count] = norm
      count += 1
  return meanIntensities

def calcMeanIntensitiesColumnNorm(communities, phenographData, phenographDataArray):
  meanIntensities = np.full((len(np.unique(communities)), len(phenographData.columns)), 0.00)
  for clus in np.unique(communities):
      cells = np.where(communities==clus)
      for channel in range(len(phenographData.columns)):
          sumIntens = []
          for i in cells[0]:
              intensVal = phenographDataArray[i][channel]
              sumIntens.append(intensVal)
          medianIntens = statistics.median(sumIntens)
          meanIntensities[clus][channel] = medianIntens
 # Normalize by column
  count = 0
  transArr = np.transpose(meanIntensities)
  for i in transArr:
      norm = np.interp(i, (i.min(), i.max()), (0, 1))
      transArr[count] = norm
      count += 1
  meanIntensities = np.transpose(transArr)
  return meanIntensities

def saveHeatmap(phenographData, meanIntensities, path, roi):
  positions = list(np.arange(0,20))
  xaxis = phenographData.columns[positions]
  hmapData = meanIntensities[:, positions]
  sns.set(rc={'figure.figsize':(20,15)})
  hmap = sns.heatmap(hmapData)
  hmap.set_xticks(np.arange(len(xaxis))+0.5)
  hmap.set_yticks(np.arange(len(clusters))+0.5)
  hmap.set_xticklabels(xaxis, rotation = 90, ha="center")
  hmap.set_yticklabels(clusters, rotation = 0)
  figure = hmap.get_figure()    
  figure.savefig(os.path.join(path, "Heatmaps v2", "ROI" + roi + "_" + "Normalized_heatmap.png"), dpi=400)
  plt.close()
  
def getCounts(communities):
  unique, counts = np.unique(communities, return_counts=True)
  cellsInClus = dict(zip(unique, counts))
  return cellsInClus

def cellsCount(cellArray):
  cellsCount = 0
  for i in cellArray:
      cellsCount += cellsInClus[i]
  cellDist = cellsCount/totCellCount
  return cellsCount, cellDist

def cellIdentifying(path, data, communities):
  data["Cluster Label"] = communities
  data.to_csv(os.path.join(path, "Clustered Data", "noncompensated_clustered_rawData_ROI0" + roi + "_ROI_0" + roi + ".csv"))
    

    
    
    
    
def main():
  trainData = pd.DataFrame()
  for roi in rois:
    roiname = "ROI0"+ roi +"_ROI_0" + roi
    cellCompData = readDataNoncomp(roi, roiname, path, slide)
    if cellCompData is None:
        continue
    phenographData, phenographDataArray = getPhenographDataNoncomp(cellCompData)
    communities = runPhenograph(phenographData, path)
    nLabels = len(np.unique(communities))
    clusters = np.unique(communities)
    meanIntensities = calcMeanIntensities(communities, phenographData, phenographDataArray)
    saveHeatmap(phenographData, meanIntensities, path, roi)
    cellsInClus = getCounts(communities)
    totCellCount = sum(cellsInClus.values())
    cellIdentifying(path, cellCompData, communities)
    
    trainData.to_csv("C:\\Users\\sindhura\\OneDrive - Queen's University\\Queens\\Research MSc\\Bladder Data\\trainData.csv")
    
 
