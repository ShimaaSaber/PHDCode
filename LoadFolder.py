# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 23:46:31 2021

@author: Shimaa
"""

from torchvision import datasets
import os
import numpy as np

class LoadFolder(datasets.ImageFolder):

    def __init__(self, root, transform):
        super(LoadFolder, self).__init__(root, transform)
        targets = np.asarray([s[1] for s in self.samples])
        self.targets = targets
        cams = []
        for s in self.samples:
            cams.append(self._get_cam_id(s[0]))
        self.cams = np.asarray(cams)

    def _get_cam_id(self, path):
        camera_id = []
        filename = os.path.basename(path)
        camera_id = filename.split('c')[1][0]
        # camera_id = filename.split('_')[2][0]
        return int(camera_id)-1


    def __getitem__(self, index):
        path, target = self.samples[index]
        
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)
            
       

        return sample, target
