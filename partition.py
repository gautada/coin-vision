#!/usr/bin/python3
import argparse
import csv
import io
import math
import os
import random
import shutil
import re

from collections import namedtuple
import xml.etree.ElementTree as et

import pandas as pd
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import tensorflow.compat.v1 as tf
from object_detection.utils import dataset_util


import workspace as ws



"""
This class should:
- read the classfile
- read the labels from the xml files
- write the map file
- append the classfile
- write the train/test csv files
- build the tf record files
"""

class Partitioner(ws.Workspace):
    LABELMAP_FILE = "label_map.pbtxt"
    LABELMAP_TEMPLATE = "item {\n\tid: %s\n\tname: '%s'\n}\n"
    MIN_ANNOTATIONS = 10
    PARTITION_TARGET = 10
    
    def __init__(self, workspaces=None, workspace=None):
        super().__init__(workspaces=workspaces, workspace=workspace)
        self._load_labels()
        
        self._check_labels(self._images)
        self._check_labels(self._train)
        self._check_labels(self._test)
        print("Loaded %s labels from Class File(%s)" % (len(self._labels), self._classfile))

    def _check_labels(self, folder):
        count = 0
        new_names = []
        name_lookup = {}
        files = os.listdir(folder)
        for file in files:
            filepath = os.path.join(folder, file)
            if os.path.isfile(filepath):
                if "xml" == filepath.split(".")[-1]:
                    names = self._read_label_names(filepath)
                    for name in names:
                        if name in self._labels:
                            count += 1
                        else:
                            new_names.append(name)
                            if name not in name_lookup:
                                name_lookup[name] = []
                            name_lookup[name].append(filepath)
        if 0 < len(new_names):
            for name in new_names:
                print("Unknown Label: %s" % name)
                for file in name_lookup[name]:
                    print(" - %s" % file)
        assert 0 == len(new_names), "Found %s new label names in annotation files" % len(new_names)
        return count

    def _collect_by_label(self, name, folders):
        collection = {}
        for folder in folders:
            collection[folder] = []
            files = os.listdir(folder)
            for file in files:
                path = os.path.join(folder, file)
                if os.path.isfile(path):
                    if "xml" == file.split(".")[1]:
                        names = self._read_label_names(path)
                        if name in names:
                            collection[folder].append(path)
        return collection


    def _collect_label_names(self, folder, label_names):
        files = os.listdir(folder)
        for file in files:
            filepath = os.path.join(folder, file)
            if os.path.isfile(filepath):
                if "xml" == filepath.split(".")[-1]:
                    names = self._read_label_names(filepath)
                    for name in names:
                        if name not in label_names:
                            label_names.append(name)
        return label_names

    def _convert_folder_to_csv(self, folder):
        columns = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
        xml_data = []
        files = os.listdir(folder)
        for file in files:
            path = os.path.join(folder, file)
            if os.path.isfile(path):
                if "xml" == file.split(".")[1]:
                    tree = et.parse(path)
                    root = tree.getroot()
                    filename = root.find('filename').text
                    width = int(root.find('size').find('width').text)
                    height = int(root.find('size').find('height').text)
                    for member in root.findall('object'):
                        bndbox = member.find('bndbox')
                        value = (filename,
                                 width,
                                 height,
                                 member.find('name').text,
                                 int(bndbox.find('xmin').text),
                                 int(bndbox.find('ymin').text),
                                 int(bndbox.find('xmax').text),
                                 int(bndbox.find('ymax').text),
                                )
                        xml_data.append(value)
        
        out_record = os.path.join(self._annotations, "%s.csv" % os.path.basename(folder))
        with open(out_record, 'w') as rf:
            writer = csv.writer(rf)
            writer.writerow(columns)
            for xml_datum in xml_data:
                writer.writerow(xml_datum)
        
        dataframe = pd.DataFrame(xml_data, columns=columns)
        return dataframe

    def _move_all_annotated_files(self, src, dst):
        print("Moving from %s to %s" % (src, dst))
        assert os.path.isdir(src), "Source folder must exist"
        assert os.path.isdir(dst), "Desitnation folder must exist"
        
        hashes = []
        files = os.listdir(src)
        for file in files:
            split = file.split(".")
            if os.path.isfile(os.path.join(src, file)) and "xml" == split[-1]:
                hashes.append(split[0])
        
        print("Moving %s annotated files" % len(hashes))
        for hash in hashes:
            xmlfile = "%s.xml" % hash
            imgfile = "%s.jpg" % hash
            xmlsrc = os.path.join(src, xmlfile)
            xmldst = os.path.join(dst, xmlfile)
            imgsrc = os.path.join(src, imgfile)
            imgdst = os.path.join(dst, imgfile)
            
            assert os.path.isfile(xmlsrc), "XML source(%s) must be a valid file" % xmlsrc
            assert os.path.isfile(imgsrc), "Image source(%s) must be a valid file" % imgsrc
            assert not os.path.isfile(xmldst), "XML destination(%s) must not be an existing file" % xmldst
            assert not os.path.isfile(imgdst), "Image destination(%s) must not be an existing file" % imgdst
                    
            print("Moving annotated image(%s)" % hash)
            shutil.move(xmlsrc, xmldst)
            shutil.move(imgsrc, imgdst)
        print("Moved %s annotated files" % len(hashes))

    def _read_label_names(self, file):
        names = []
        assert os.path.isfile(file), "File (%s) must exist" % file
        assert "xml" == file.split(".")[-1], "File should be an XML file"
        tree = et.parse(file)
        root = tree.getroot()
        for member in root.findall('object'):
            name = member.find('name').text
            if name not in names:
                names.append(name)
        return names
    
    def _reset_training_files(self):
        self._move_all_annotated_files(self._images, self._train)
        self._move_all_annotated_files(self._test, self._train)
    
    def _create_a_record(self, folder, group):
        with tf.gfile.GFile(os.path.join(folder, '{}'.format(group.filename)), 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        width, height = image.size
    
        filename = group.filename.encode('utf8')
        image_format = b'jpg'
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []

        for index, row in group.object.iterrows():
            xmins.append(row['xmin'] / width)
            xmaxs.append(row['xmax'] / width)
            ymins.append(row['ymin'] / height)
            ymaxs.append(row['ymax'] / height)
            classes_text.append(row['class'].encode('utf8'))
            classes.append(self._labels.index(row['class'])+1)

        return tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes)}))
    
    
    def _convert_to_record(self, folder):
        assert os.path.isdir(folder), "Folder to convert must be a directory"
        dataframe = self._convert_folder_to_csv(folder)
        # print(dataframe)
        # print
        # print("---------------------------------------------------------------------------")
        # Re-group dataframe for tf-record.
        data = namedtuple('data', ['filename', 'object'])
        gb = dataframe.groupby('filename')
        grouped = [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]
        # print(grouped)
        recordfile = os.path.join(self._annotations, "%s.record" % os.path.basename(folder))
        # print(recordfile)
        writer = tf.python_io.TFRecordWriter(recordfile)
        for group in grouped:
            record = self._create_a_record(folder, group)
            writer.write(record.SerializeToString())
        writer.close()
    
    def convert(self):
        self._convert_to_record(self._test)
        self._convert_to_record(self._train)
        # train_dataframe = self._convert_folder_to_csv(self._train)
        # writer = tf.python_io.TFRecordWriter(args.output_path)
        # path = os.path.join(args.image_dir)
        # examples = xml_to_csv(args.xml_dir)
        # grouped = split(examples, 'filename')
    
    
        # self._convert_folder_to_csv(self._test)

    def map(self):
        """
        Create the Label Map File
        """
        mapfile = os.path.join(self._annotations, Partitioner.LABELMAP_FILE)
        i = 1
        with open(mapfile, 'w') as mf:
            for label in self._labels:
                mf.write(Partitioner.LABELMAP_TEMPLATE % (i, label))
                i += 1
        print("Write %s label to Map File(%s)" % (i-1, mapfile))
                
    def partition(self, force=False):
        current = math.ceil(float(len(os.listdir(self._test)))/float(len(os.listdir(self._test))+len(os.listdir(self._train))) * 100.0 )
        if Partitioner.PARTITION_TARGET <= current and False == force:
            print("Skip re-partition, user force=True to re-partition")
            return
        self._reset_training_files()
        # For each name check that 10% are in test
        for label in self._labels:
            collection = self._collect_by_label(label, [self._test, self._train])
            train = len(collection[self._train])
            test = len(collection[self._test])
            assert 0 == test, "All annotated images should be reset to train folder"
            total = train + test
            assert Partitioner.MIN_ANNOTATIONS < total, "Minimum annotations for label(%s) not found: [%s, %s]" % (label, train, test)
            target = max(math.trunc(total * float(Partitioner.PARTITION_TARGET)/100.0), 1)
            assert target < train, "Target number for testing(%s) should be less than total training files(%s)" % (target, train)
            while test < target:
                r = random.randint(0, len(collection[self._train])-1)
                xml_src = collection[self._train][r]
                    
                hash = os.path.basename(xml_src).split(".")[0]
                    
                xml_dst = os.path.join(self._test, "%s.xml" % hash)
                img_src = os.path.join(self._train, "%s.jpg" % hash)
                img_dst = os.path.join(self._test, "%s.jpg" % hash)
                    
                assert os.path.isfile(xml_src), "XML source(%s) must be a valid file" % xml_src
                assert os.path.isfile(img_src), "Image source(%s) must be a valid file" % img_src
                assert not os.path.isfile(xml_dst), "XML destination must not be an existing file"
                assert not os.path.isfile(img_dst), "Image source must be an existing file"
                    
                shutil.move(xml_src, xml_dst)
                shutil.move(img_src, img_dst)
                        
                collection = self._collect_by_label(label, [self._test, self._train])
                train = len(collection[self._train])
                test = len(collection[self._test])
                total = train + test
                print("%s: %s[%s, %s]" % (label, total, train, test))
        
if "__main__" == __name__:
    parser = argparse.ArgumentParser(description='Partition the image files between train and test and create records for each')
    parser.add_argument('--workspace', help='Path to workspace project')
    args = parser.parse_args()
    partitioner = Partitioner(workspace=args.workspace)
    partitioner.map()
    partitioner.partition(force=True)
    partitioner.convert()
    
    
