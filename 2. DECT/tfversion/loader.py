import os
import random
import numpy as np
import math
import tensorflow as tf
import tensorflow_datasets as tfds

class DectLoader(tfds.core.GeneratorBasedBuilder):
    
    def _info(self):
        return tfds.core.DatasetInfo(
            description=("This is the dataset for Zeng's DECT tasks"), 
            features=tfds.features.FeaturesDict({
                "image_description": tfds.features.Text(), 
                "image": tfds.features.Image(), 
                "label": tfds.features.ClassLabel(num_classes=2), 
            }), 
            supervised_keys("image", "label"), 
            homepage="https://www.sysu.edu.cn", 
            citation=r"""@Gary_Pan{Foshan-dataset-2023, 
                                    author={Hai, Jiang}, "}""", 
        )

    def _split_generator(self, dl_manager):
        pass

    def _generate_examples(self):
        yield 'key', {}