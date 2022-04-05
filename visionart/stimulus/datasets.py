import os
import re
import cv2
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow_datasets as tfds


def preprocess_image(img: np.ndarray, dataset_type: str) -> np.ndarray:
    if dataset_type == 'mnist':
        img = (255 - img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.uint8(img)


def build_dataset(dataset_type: str, dataset_dir: str or None = None):
    tfds.enable_progress_bar()

    builder_kwargs = dict()
    builder = tfds.builder(dataset_type, try_gcs=False, data_dir=dataset_dir, **builder_kwargs)

    download_and_prepare_kwargs = dict()
    builder.download_and_prepare(**download_and_prepare_kwargs)

    return builder


def info_dataset(builder,
                 verbose: bool = False):
    dataset_info = dict(
        general={"name": builder.info.name.upper(), "full_name": builder.info.full_name, "path": builder.info.data_dir},
        features={"image": {"shape": builder.info.features["image"].shape,
                            "dtype": builder.info.features["image"].dtype,
                            "samples": builder.info.splits["train"].num_examples +
                                       builder.info.splits["test"].num_examples},
                  "label": {"shape": builder.info.features["label"].shape,
                            "dtype": builder.info.features["label"].dtype,
                            "classes": builder.info.features["label"].num_classes,
                            "names": builder.info.features["label"].names}},
        splits={"train": {"samples": builder.info.splits["train"].num_examples,
                          "shards": builder.info.splits["train"].num_shards,
                          "len_shards": builder.info.splits["train"].shard_lengths,
                          "filenames": builder.info.splits["train"].filenames},
                "test": {"samples": builder.info.splits["test"].num_examples,
                         "shards": builder.info.splits["test"].num_shards,
                         "len_shards": builder.info.splits["test"].shard_lengths,
                         "filenames": builder.info.splits["test"].filenames}}
    )
    if verbose:
        print(f"\nThe {dataset_info['general']['name']} dataset has {dataset_info['features']['image']['samples']} "
              f"images (with <shape: '{dataset_info['features']['image']['shape']}'> and "
              f"{dataset_info['features']['image']['dtype']}) divided in "
              f"{dataset_info['features']['label']['classes']} classes {dataset_info['features']['label']['names']}:\n"
              f"   - train: {dataset_info['splits']['train']['samples']} samples split in "
              f"{dataset_info['splits']['train']['shards']} shards having "
              f"{dataset_info['splits']['train']['len_shards']} samples respectively\n"
              f"   - test: {dataset_info['splits']['test']['samples']} samples split in "
              f"{dataset_info['splits']['test']['shards']} shards having "
              f"{dataset_info['splits']['train']['len_shards']} samples respectively\n")
    return dataset_info


def load_dataset(builder,
                 split: str or (str, str) = 'train'):
    as_dataset_kwargs = dict(
        split=split,
        as_supervised=False,
        shuffle_files=False,
        batch_size=None,
        read_config=tfds.ReadConfig(add_tfds_id=True),
        decoders=None
    )

    ds = builder.as_dataset(**as_dataset_kwargs)
    return ds


def take_samples(ds,
                 take: int, skip: int = 0):
    ds2take = ds.skip(skip).take(take).cache()
    if take < 2:
        ds2take, = ds2take
    return ds2take


def take_batch(ds,
               batch_size: int, batches2skip: int = 0):
    batch, = batch_dataset(ds, batch_size).skip(batches2skip).take(1).cache()
    return batch


def retrieve_image(sample):
    return sample['image'].numpy()


def retrieve_image_from_batch(ds_batched,
                              sample_num: int):
    return ds_batched['image'][sample_num].numpy()


def retrieve_label(sample) -> int:
    return int(sample['label'])


def retrieve_label_from_batch(ds_batched,
                              sample_num: int):
    return int(ds_batched['label'][sample_num])


def retrieve_id(sample) -> str:
    return str(sample['tfds_id'].numpy().decode('utf-8'))


def retrieve_id_from_batch(ds_batched,
                           sample_num: int):
    return str(ds_batched['tfds_id'][sample_num].numpy().decode('utf-8'))


def id2int(tfds_id: str,
           data_info: dict) -> int:
    """Format the tfds_id in a more human-readable way."""
    match = re.match(r'\w+-(\w+).\w+-(\d+)-of-\d+__(\d+)', tfds_id)
    split_name, shard_id, ex_id = match.groups()
    return sum(data_info['splits'][split_name]['len_shards'][:int(shard_id)]) + int(ex_id)


def retrieve_sample_from_id(ds,
                            tfds_id: int):
    return take_samples(ds, take=1, skip=tfds_id)


def portion2batch(num_samples: int, portion: float):
    batch_size = int(portion * num_samples)
    if num_samples % batch_size == 0:
        batch_num = int(num_samples / batch_size)
    else:
        batch_num = int(num_samples / batch_size) + 1
    return batch_size, batch_num


def batch_dataset(ds, batch_size: int):
    return ds.batch(batch_size)


def evaluate_recording_duration(duration4rec: float, num_records: int, verbose: bool = False):
    """Input duration of a single recording must be in seconds.
    The output duration of the whole session of recordings will be in hours."""
    session_duration = num_records * duration4rec / 3600
    if verbose:
        print(f'\nThe whole session of recordings will last approximately {round(session_duration, 1)} hours: '
              f'i.e. ~{int(duration4rec)} seconds for each one of the {num_records} images to record.\n')
    return session_duration
