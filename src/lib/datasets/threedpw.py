import os
import torch.utils.data as data
import numpy as np
import torch
import cv2
import pickle
from utils.image import flip, shuffle_lr, get_affine_transform, affine_transform
from utils.image import draw_gaussian, adjust_aspect_ratio

class ThreeDPW(data.Dataset):
  def __init__(self, opt, split):
    print('==> initializing 3D {} data.'.format(split))
    self.data_path = os.path.join(opt.data_dir, '3dpw', 'sequenceFiles', split)
    self.files = os.listdir(self.data_path)
    self.split = split
    self.opt = opt

  def _load_image(self, index):
    with open(os.path.join(self.data_path, self.files[index]), 'rb') as f:
      data = pickle.load(f)
    image_path = data['images']
    img = cv2.imread(image_path)
    return img, data

  def _get_part_info(self, data):
    pts = data['joints2D'].copy().astype(np.float32)
    c = np.mean(pts, axis=0)
    s = max(pts[:, 0].max() - pts[:, 0].min(), pts[:, 1].max() - pts[:, 1].min())
    return pts, c, s
      
  def __getitem__(self, index):
    img, data = self._load_image(index)
    pts, c, s = self._get_part_info(data)
    r = 0

    # Transformations as in MPII
    if self.split == 'train':
      sf = self.opt.scale
      rf = self.opt.rotate
      s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
      r = np.clip(np.random.randn()*rf, -rf*2, rf*2) if np.random.random() <= 0.6 else 0

    trans_input = get_affine_transform(c, s, r, [self.opt.input_w, self.opt.input_h])
    inp = cv2.warpAffine(img, trans_input, (self.opt.input_w, self.opt.input_h),
                         flags=cv2.INTER_LINEAR)

    trans_output = get_affine_transform(c, s, r, [self.opt.output_w, self.opt.output_h])
    out = np.zeros((self.num_joints, self.opt.output_h, self.opt.output_w), dtype=np.float32)

    for i in range(self.num_joints):
      if pts[i, 0] > 0 or pts[i, 1] > 0:
        pt = affine_transform(pts[i], trans_output)
        out[i] = draw_gaussian(out[i], pt, self.opt.hm_gauss)

    inp = (inp.astype(np.float32) / 255.)
    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)
  
    return inp, out

  def __len__(self):
    return len(self.files)
