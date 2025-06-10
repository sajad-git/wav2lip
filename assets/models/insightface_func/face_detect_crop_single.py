'''
Author: Naiyuan liu
Github: https://github.com/NNNNAI
Date: 2021-11-23 17:03:58
LastEditors: Naiyuan liu
LastEditTime: 2021-11-24 16:46:04
Description: 
'''
from __future__ import division
import collections
import numpy as np
import glob
import os
import os.path as osp
import cv2
from insightface.model_zoo import model_zoo

__all__ = ['Face_detect_crop', 'Face']

Face = collections.namedtuple('Face', [
    'bbox', 'kps', 'det_score', 'embedding', 'gender', 'age',
    'embedding_norm', 'normed_embedding',
    'landmark'
])

Face.__new__.__defaults__ = (None, ) * len(Face._fields)


def estimate_norm(lmk, image_size=96, mode='None'):
    """
    Simple face alignment function to replace the missing face_align module.
    
    Args:
        lmk: facial landmarks (5 points: left_eye, right_eye, nose, left_mouth, right_mouth)
        image_size: target image size
        mode: alignment mode (unused for compatibility)
    
    Returns:
        M: transformation matrix
        pose_index: pose index (always 0 for compatibility)
    """
    if lmk is None:
        # Return identity transform if no landmarks
        M = np.eye(2, 3, dtype=np.float32)
        return M, 0
    
    # Standard facial landmark positions for alignment
    src_pts = np.array([
        [30.2946, 51.6963],  # left eye
        [65.5318, 51.5014],  # right eye
        [48.0252, 71.7366],  # nose tip
        [33.5493, 92.3655],  # left mouth corner
        [62.7299, 92.2041]   # right mouth corner
    ], dtype=np.float32)
    
    # Scale to target image size
    src_pts *= image_size / 96.0
    
    # Extract landmarks
    if lmk.shape[0] >= 5:
        dst_pts = lmk[:5].astype(np.float32)
    else:
        # If we don't have 5 landmarks, create a dummy transformation
        M = np.eye(2, 3, dtype=np.float32)
        return M, 0
    
    # Compute similarity transformation
    M = cv2.estimateAffinePartial2D(dst_pts, src_pts)[0]
    
    if M is None:
        # Fallback to identity if transformation fails
        M = np.eye(2, 3, dtype=np.float32)
    
    return M, 0


class Face_detect_crop:
    def __init__(self, name, root='~/.insightface_func/models'):
        self.models = {}
        root = os.path.expanduser(root)
        onnx_files = glob.glob(osp.join(root, name, '*.onnx'))
        onnx_files = sorted(onnx_files)
        for onnx_file in onnx_files:
            if onnx_file.find('_selfgen_')>0:
                #print('ignore:', onnx_file)
                continue
            model = model_zoo.get_model(onnx_file)
            if model.taskname not in self.models:
                print('find model:', onnx_file, model.taskname)
                self.models[model.taskname] = model
            else:
                print('duplicated model task type, ignore:', onnx_file, model.taskname)
                del model
        assert 'detection' in self.models
        self.det_model = self.models['detection']

    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640), mode ='None'):
        self.det_thresh = det_thresh
        self.mode = mode
        assert det_size is not None
        print('set det-size:', det_size)
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname=='detection':
                model.prepare(ctx_id, input_size=det_size)
            else:
                model.prepare(ctx_id)

    def detect(self, img, max_num=0, metric='default'):
        """
        Detect faces in image.
        
        Args:
            img: input image
            max_num: maximum number of faces to detect
            metric: detection metric
            
        Returns:
            bboxes: detected face bounding boxes
            kpss: detected face landmarks
        """
        return self.det_model.detect(img, max_num=max_num, metric=metric)

    def get(self, img, crop_size, max_num=0):
        bboxes, kpss = self.det_model.detect(img,
                                             #threshold=self.det_thresh,
                                             max_num=max_num,
                                             metric='default')
        if bboxes.shape[0] == 0:
            return None

        det_score = bboxes[..., 4]
                
        # select the face with the hightest detection score
        best_index = np.argmax(det_score)

        kps = None
        if kpss is not None:
            kps = kpss[best_index]
        M, _ = estimate_norm(kps, crop_size, mode = self.mode) 
        align_img = cv2.warpAffine(img, M, (crop_size, crop_size), borderValue=0.0)

        return [align_img], [M]

    def getBox(self, img, max_num=0):
        bboxes, kpss = self.det_model.detect(img,
                                             #threshold=self.det_thresh,
                                             max_num=max_num,
                                             metric='default')
        if bboxes.shape[0] == 0:
            return None

        x1 = int(bboxes[0, 0:1])
        y1 = int(bboxes[0, 1:2])
        x2 = int(bboxes[0, 2:3])
        y2 = int(bboxes[0, 3:4])
        

        return (x1,y1,x2,y2)