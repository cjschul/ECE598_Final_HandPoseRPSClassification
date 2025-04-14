import tensorflow as tf
import tensorflow_datasets as tfds
import os
import numpy as np

class MyHandPoseRpsDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')
    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description="Custom Rock Paper Scissors dataset using hand joint positions",
            features=tfds.features.FeaturesDict({
                'pose': tfds.features.Tensor(shape=(21, 3), dtype=np.float32),
                'label': tfds.features.ClassLabel(names=['rock','paper','scissors'])
            }),
            supervised_keys=('pose', 'label'),
        )

    def _split_generators(self, dl_manager):
        data_path = os.path.dirname(os.path.abspath(__file__))

        return {
            'train': self._generate_examples(os.path.join(data_path, 'train')),
            'test': self._generate_examples(os.path.join(data_path, 'test')),
        }

    def _generate_examples(self, data_path):
        label_names = ['rock', 'paper', 'scissors']
        for fname in os.listdir(data_path):
            if fname.endswith('.npy'):
                fpath = os.path.join(data_path, fname)
                data = np.load(fpath, allow_pickle=True).item()
                yield fname, {
                    'pose' : np.array(data['pose'], dtype=np.float32),
                    'label' : label_names[int(data['label'])]
                }