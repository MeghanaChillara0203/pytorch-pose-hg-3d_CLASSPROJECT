import os
import torch.utils.data as data
import numpy as np
import torch
import cv2
import pickle
from utils.image import flip, shuffle_lr, get_affine_transform, affine_transform
from utils.image import draw_gaussian, adjust_aspect_ratio
from utils.image import transform_preds

class ThreeDPW(data.Dataset):
  def __init__(self, opt, split):
    print('==> initializing 3D {} data.'.format(split))
    self.data_path = os.path.join(opt.data_dir, '3dpw', 'sequenceFiles', split)
    self.num_joints = 16
    self.acc_idxs = [0, 1, 2, 3, 4, 5, 10, 11, 14, 15]
    self.shuffle_ref = [[0, 5], [1, 4], [2, 3], 
                        [10, 15], [11, 14], [12, 13]]
    self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
    self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)
    self.edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], 
                  [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15], 
                  [6, 8], [8, 9]]
    self.edges_3d = [[3, 2], [2, 1], [1, 0], [0, 4], [4, 5], [5, 6], \
                     [0, 7], [7, 8], [8, 10],\
                     [16, 15], [15, 14], [14, 8], [8, 11], [11, 12], [12, 13]]
    annot = {}
    tags = ['image','joints','center','scale']
    self.files = os.listdir(self.data_path)
    self.split = split
    self.opt = opt

  def _load_image(self, index):
    with open(os.path.join(self.data_path, self.files[index]), 'rb') as f:
      data = pickle.load(f, encoding='latin1')
    sequence = data['sequence']
    img_id = data['img_frame_ids']
    # assuming image filename is in format sequence_imgid.jpg
    image_filename = "{}_{}.jpg".format(sequence, img_id)
    image_path = os.path.join(self.opt.data_dir, '3dpw', 'imageFiles', sequence, image_filename)
    img = cv2.imread(image_path)
    return img, data

  def _get_part_info(self, data):
    pts = np.array(data['poses2d']).copy().astype(np.float32)
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
  
  def convert_eval_format(self, pred, conf, meta):
    ret = np.zeros((pred.shape[0], pred.shape[1], 2))
    for i in range(pred.shape[0]):
      ret[i] = transform_preds(
        pred[i], meta['center'][i].numpy(), meta['scale'][i].numpy(), 
        [self.opt.output_h, self.opt.output_w])
    return ret
