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
  mhd_list = glob.glob('Dataset/subset0/subset0/*.mhd')
  presentOnDisk = {os.path.split(p)[-1][:-4] for p in mhd_list}
 
  # To add diameter info using annotations.csv

  diameter_dict = {}

  with open('Dataset/annotations.csv') as f:

    for row in list(csv.reader(f))[1:]:

      series_uid = row[0]
      annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
      annotationDiameter_mm = float(row[4])

      diameter_dict.setdefault(series_uid, []).append((annotationCenter_xyz, annotationDiameter_mm))

  # now we get candidate info:

  CandidateInfo_List = []
  count = 0
  with open('Dataset/candidates.csv') as f:

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
      '''
      Now here we get the CandidateInfo_List which have the information of
      every candidate in a tuple form, the information is NoduleBool indicates
      whether this candidate have actual nodule or not, the nodule diameter in mm, 
      series uid of every candidate, (x, y, z) coordinates of candidate.
      '''

      CandidateInfo_List.append(CandidateInfoTuple(
          isNoduleBool,
          CandidateDiameter_mm,
          series_uid,
          CandidateCenter_xyz
      ))
  print(count)
  CandidateInfo_List.sort(reverse=True)
  return CandidateInfo_List


'''
Unfortunately, all the candidates from the CandidateInfo_List have center data in millimeters not voxels!,
we need to convert the (X, Y, Z) patient based coordinate system to the (I, R, C) voxel-based 
coordinate system.

The patient coordinate system is measured in millimeters and has an arbitrarily 
positioned origin that does not correspond to the origin of the CT voxel array

Here is a way:


Convert from IRC To XYZ Steps:

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

'''
Here we read the .mhd and .raw files from the disk and convert them into 3d array,
There is a concept of Housefield Units in the medical images, HU of -1000 means air, +1000 means
bones, metals and some others, 0 means water. most of the techniques are that remove -1000 HU values
from the CT Scan because we call those air and outliers and they would affect our model during training
like in batch normalization. we also remove +1000 hu values because they are also not relevent.
the hu value around 0 will be useful because they can be tissues of our concern.

we create GetRawCandidate function which will slice the candidate ct scan and only use
that chunked area for our training purpose and validation purpose.


'''
class CT:

  def __init__(self, series_uid):
    mhd_path = glob.glob('Dataset/subset0/subset0/{}.mhd'.format(series_uid))[0]
    ct_mhd = sitk.ReadImage(mhd_path)
    ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

    ct_a = ct_a.clip(-1000, 1000, ct_a)
    self.series_uid = series_uid
    self.hu_a = ct_a
    self.origin_xyz = Xyz_tuple(*ct_mhd.GetOrigin())
    self.vxSize_xyz = Xyz_tuple(*ct_mhd.GetSpacing())
    self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3,3)


  def getRawCandidate(self, center_xyz, width_irc):

    slice_list = []

    center_irc = Xyz2irc(center_xyz, 
            self.origin_xyz, 
            self.vxSize_xyz, 
            self.direction_a)

    for axis, center_val in enumerate(center_irc):
      start_ndx = int(round(center_val - width_irc[axis] / 2))
      end_ndx = int(start_ndx + width_irc[axis])

      assert center_val >=0 and center_val < self.hu_a.shape[axis], repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

      if start_ndx < 0 :
        start_ndx = 0

        end_ndx = width_irc[axis]

      if end_ndx > self.hu_a.shape[axis]:
        end_ndx = self.hu_a.shape[axis]

        start_ndx = int(round(self.hu_a.shape[axis] - width_irc[axis]))
      
      slice_list.append(slice(start_ndx, end_ndx))
    
    chunk_t = self.hu_a[tuple(slice_list)]

    return chunk_t, center_irc
  

@functools.lru_cache(1, typed=True)
def getCt(series_uid):
  return CT(series_uid)

def getCtRawCandidate(series_uid, center_xyz, width_irc):
  ct = getCt(series_uid)
  chunk_t, center_irc = ct.getRawCandidate(center_xyz, width_irc)

  return chunk_t, center_irc


class LunaDataset(Dataset):

  def __init__(self, val_stride = 0, isValSet_bool = None, series_uid = None):

    self.candidateInfo_list = copy.copy(getCandidateInfoList())
    if series_uid:
      self.candidateInfo_list = [x for x in self.candidateInfo_list if x.series_uid == series_uid]

    if isValSet_bool:
      assert val_stride > 0, val_stride
      self.candidateInfo_list = self.candidateInfo_list[::val_stride]
      assert self.candidateInfo_list
    elif val_stride > 0:
      del self.candidateInfo_list[::val_stride]
      assert self.candidateInfo_list


  def __len__(self):
    return len(self.candidateInfo_list)

  def __getitem__(self, ndx):

    candidateInfo_tup = self.candidateInfo_list[ndx]
    width_irc = (32, 48, 48)

    candidate_a, center_irc = getCtRawCandidate(candidateInfo_tup.series_uid,
                                                candidateInfo_tup.center_xyz,
                                                width_irc)

    candidate_t = torch.from_numpy(candidate_a)
    candidate_t = candidate_t.to(torch.float32)
    candidate_t = candidate_t.unsqueeze(0)

    pos_t = torch.tensor([
        not candidateInfo_tup.isNoduleBool,
        candidateInfo_tup.isNoduleBool
    ], dtype = torch.long)

    return (
        candidate_t,
        pos_t,
        candidateInfo_tup.series_uid,
        center_irc
    )

candidateInfo_List = getCandidateInfoList(requireOnDisk_bool = False)
positive_list = [x for x in candidateInfo_List if x[0]]
diameter_list = [x[1] for x in positive_list]

for i in range(0, len(diameter_list), 100):
  print(f'{i}   {diameter_list[i]}')