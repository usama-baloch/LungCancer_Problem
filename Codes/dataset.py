import torch
from collections import namedtuple
import glob
import functools
import os, csv
import SimpleITK as sitk
import numpy as np
import torch
import copy
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


CandidateInfoTuple = namedtuple('CandidateInfoTuple',
                                'isNoduleBool, diameter_mm, series_uid, center_xyz')
@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool = True):
  mhd_list = glob.glob('/content/drive/MyDrive/DataSets/Luna_dataset/subset0/*.mhd')
  presentOnDisk = {os.path.split(p)[-1][:-4] for p in mhd_list}

  # To add diameter info using annotations.csv

  diameter_dict = {}

  with open('/content/drive/MyDrive/DataSets/Luna_dataset/annotations.csv') as f:

    for row in list(csv.reader(f))[1:]:

      series_uid = row[0]
      annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
      annotationDiameter_mm = float(row[4])

      diameter_dict.setdefault(series_uid, []).append((annotationCenter_xyz, annotationDiameter_mm))

  # now we get candidate info:

  CandidateInfo_List = []
  with open('/content/drive/MyDrive/DataSets/Luna_dataset/candidates.csv') as f:

    for row in list(csv.reader(f))[1:]:

      series_uid = row[0]

      if series_uid not in presentOnDisk and requireOnDisk_bool:
        continue

      isNoduleBool = bool(int(row[4]))
      CandidateCenter_xyz = tuple([float(x) for x in row[1:4]])
      CandidateDiameter_mm = 0.0


      for annotation_tup in diameter_dict.get(series_uid, []):
        annotationCenter_xyz, annotationDiameter_mm = annotation_tup

        for i in range(3):
          diameter_ = abs(CandidateCenter_xyz[i] - annotationCenter_xyz[i])

          if diameter_ > annotationDiameter_mm / 4:
            break
          else:
            CandidateDiameter_mm = annotationDiameter_mm
            break

      CandidateInfo_List.append(CandidateInfoTuple(
          isNoduleBool,
          CandidateDiameter_mm,
          series_uid,
          CandidateCenter_xyz
      ))

  CandidateInfo_List.sort(reverse=True)
  return CandidateInfo_List


'''
Conver from IRC To XYZ Steps:

1) Convert the Coordinate IRC TO CRI to align with XYZ
2) Scale the indices with the voxel sizes
3) matrix-multiplication of scaled indices with direction matrix using @ python
4) add the offset origin

'''
Irc_tuple = namedtuple('Irc_tuple', ['index', 'row', 'column'])
Xyz_tuple = namedtuple('Xyz_typle', ['x', 'y', 'z'])

def Irc2xyz(center_irc, origin_xyz, vxSize_xyz, direction_a):
  coord_a = np.array(center_irc)[::-1]
  origin_a = np.array(origin_xyz)
  vxSize_a = np.array(vxSize_xyz)

  coord_xyz = ((coord_a * vxSize_a) @ direction_a) + origin_a
  return Xyz_tuple(*coord_xyz)


def Xyz2irc(center_xyz, origin_xyz, vxSize_xyz, direction_a):

  vxSize_a = np.array(vxSize_xyz)
  origin_a = np.array(origin_xyz)
  coord_a = np.array(center_xyz)

  coord_irc = ((coord_a - origin_a) @ np.linalg.inv(direction_a)) / vxSize_a
  coord_irc = np.round(coord_irc)

  return Irc_tuple(int(coord_irc[2]), int(coord_irc[1]), int(coord_irc[0]))