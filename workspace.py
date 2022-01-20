#!/usr/bin/python3
import argparse
import os
import shutil
import time

class Workspace:
    """
    Workspaces is a collection of Tensorflow tensorflow trainig scenarios. Primarily,
    this is for image object detection model development and training.
    """
    BASE_FOLDER = "workspaces"
    ANNOTATIONS_FOLDER = "annotations"
    IMAGES_FOLDER = "images"
    IMG_TRAIN = "train"
    IMG_TEST = "test"
    IMG_EVAL = "eval"
    TRAINING_FOLDER = "models"
    TRAINED_FOLDER = "pre-trained-models"
    CLASS_FILE = "classfile"
    
    def __init__(self, workspaces=None, workspace=None):
        if workspaces is None:
            workspaces = os.path.join(os.path.expanduser('~'), Workspace.BASE_FOLDER)
        self._workspaces = workspaces
        assert os.path.isdir(self._workspaces), "Workspace[s]/base(%s) must exist" % self._workspaces
        # Workspace
        assert workspace is not None, "Workspace cannot be none"
        self._workspace = self.__make_folder(os.path.join(self._workspaces, workspace))
        # Folders
        self._annotations = self.__make_folder(os.path.join(self._workspace, Workspace.ANNOTATIONS_FOLDER))
        self._images = self.__make_folder(os.path.join(self._workspace, Workspace.IMAGES_FOLDER))
        self._train = self.__make_folder(os.path.join(self._workspace, Workspace.IMAGES_FOLDER, Workspace.IMG_TRAIN))
        self._test = self.__make_folder(os.path.join(self._workspace, Workspace.IMAGES_FOLDER, Workspace.IMG_TEST))
        # self._eval = self.__make_folder(os.path.join(self._workspace, Workspace.IMAGES_FOLDER, Workspace.IMG_EVAL))
        self._training = self.__make_folder(os.path.join(self._workspace, Workspace.TRAINING_FOLDER))
        self._trained = self.__make_folder(os.path.join(self._workspace, Workspace.TRAINED_FOLDER))
        # Files
        self._classfile = self.__check_file(os.path.join(self._workspace, Workspace.ANNOTATIONS_FOLDER, Workspace.CLASS_FILE))
        self._labels = None
        
    def __make_folder(self, path):
        assert path is not None, "[MAKE FOLDER] path(%s) cannot be none" % path
        if os.path.isdir(path): path
        name = os.path.basename(path)
        assert os.path.isdir(path[:-1*len(name)-1]), "[MAKE FOLDER] parent folder(%s) must exist" % path
        os.makedirs(path, exist_ok=True)
        assert os.path.isdir(path), "[MAKE FOLDER] Folder(%s) must exist" % path
        return path
        
    def __check_file(self, path):
        assert path is not None, "[CHECK FILE] path(%s) cannot be none." % path
        assert os.path.isfile(path), "[CHECK FILE] file(%s) must exist" % path
        return path
        
    def _load_labels(self):
        self._labels = []
        with open(self._classfile, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                line = line.strip()
                assert line not in self._labels, "Cannot have duplicate(%s) labels" % line
                if 0 < len(line):
                    self._labels.append(line)
        assert 0 < len(self._labels), "There must be at least one label"
        
