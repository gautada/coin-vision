#!/usr/bin/python3
import os
import shutil
from tensorflow.python.keras.utils.data_utils import get_file

if "__main__" == __name__:
    cache = "/home/tensorflow/workspaces/coinvision"
    model_url = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz"
    model_file = os.path.basename(model_url)
    model_name = model_file[:model_file.index(".")]
    new_model_name = "ssd_resnet50_v1_fpn_coinvision"
    get_file(fname=model_file,
                 origin=model_url,
                 cache_dir=cache,
                 cache_subdir="pre-trained-models",
                 extract=True)
    new_model_path = os.path.join(cache, "models", new_model_name)
    os.makedirs(new_model_path, exist_ok=True)
    new_model_pipeline = os.path.join(new_model_path, "pipeline.conigf")
    if not os.path.isfile(new_model_pipeline):
        source = os.path.join(cache, "pre-trained-models", model_name, "pipeline.config")
        dest = os.path.join(cache, "models", new_model_name, "pipeline.config")
        shutil.copy(source, dest)
        
    # tf.keras.backend.clear_session()
    # self._model = tf.saved_model.load(os.path.join(self.cache_models, self.model_name, "saved_model"))

