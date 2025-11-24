from typing import List, Dict, Iterable, Optional

import random
import os
import glob
import torch
import webdataset as wds
import numpy
import io
from PIL import Image
import pyarrow.parquet as pq
import pyarrow.compute as pc
from torch.utils.data import IterableDataset
from torchvision.transforms import Normalize
from torchvision.transforms.functional import to_tensor
import copy
import functools

def resize(pil_image, image_size=256):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    return pil_image

def center_crop_fn(image, height, width):
    crop_x = (image.width - width) // 2
    crop_y = (image.height - height) // 2
    return image.crop((crop_x, crop_y, crop_x + width, crop_y + height))

def random_crop_fn(image, height, width):
    crop_x = random.randint(0, image.width - width)
    crop_y = random.randint(0, image.height - height)
    return image.crop((crop_x, crop_y, crop_x + width, crop_y + height))

def find_nearest_aspect_ratio_bins(aspect_ratio, aspect_ratio_bins):
    min_distance = 1000000
    min_index = 0
    for i in range(len(aspect_ratio_bins)):
        dis = abs(aspect_ratio - aspect_ratio_bins[i])
        if dis < min_distance:
            min_distance = dis
            min_index = i
    return min_index

class PackedParquetDataset(IterableDataset):
    def __init__(self,
                 data_sources: dict,
                 caption_weight: dict,
                 resolution=256,
                 random_crop=False,
                 ):
        self.data_sources = data_sources
        self.resolution = resolution
        self.normalize = Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
        if random_crop:
            self.crop_fn = random_crop_fn
        else:
            self.crop_fn = center_crop_fn

        self.parquet_files = []

        for root, repeat in self.data_sources.items():
            parquet_files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.parquet')]
            parquet_files = parquet_files * repeat
            self.parquet_files.extend(parquet_files)
        print("parquet files: ", len(self.parquet_files))
        self.caption_weight = caption_weight
        self.prefix_template = [
            "A photo of ",
            "A picture of ",
            "A visual representation of ",
            "A image of ",
            "A scene of ",
            "A view of ",
            "A depiction of ",
        ]


    def __iter__(self):
        # when seed everything. each work, no matter local or global will have distinct seed!
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = len(self.parquet_files)
        else:
            per_worker = int(len(self.parquet_files) / worker_info.num_workers)
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = iter_start + per_worker if worker_id < worker_info.num_workers - 1 else len(self.parquet_files)
        print("iter_start: ", iter_start, "iter_end: ", iter_end, "worker_id: ", worker_info.id, "num_workers: ", worker_info.num_workers, "len: ", len(self.parquet_files))


        while True:
            index  = random.choice(range(iter_start, iter_end))
            file = self.parquet_files[index]
            table = pq.read_table(file)

            # random order
            sampled_indices = numpy.random.choice(table.num_rows, size=table.num_rows, replace=False)
            sampled_indices = sampled_indices.tolist()

            for i in sampled_indices:
                metadata = dict()
                for cname in table.column_names:
                    metadata[cname] = table[cname][i].as_py()
                # select a caption
                caption_key = numpy.random.choice(list(self.caption_weight.keys()), p=list(self.caption_weight.values()))
                if caption_key not in metadata:
                    continue
                caption = metadata[caption_key]

                # prefix template for short caption
                if random.random() < 0.5 and 'long' not in caption_key:
                    caption = random.choice(self.prefix_template) + caption
                image_bytes = metadata.pop('image')

                try:
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    pil_image = pil_image.convert('RGB')
                    height, width = pil_image.size
                    if min(height, width) < self.resolution:
                        continue
                    pil_image = resize(pil_image, self.resolution)
                    pil_image = self.crop_fn(pil_image, self.resolution, self.resolution)
                    raw_image = to_tensor(pil_image)
                    normalized_image = self.normalize(raw_image)
                    metadata = {
                        "raw_image": raw_image,
                        "prompt": caption,
                    }
                    data = (normalized_image, caption, metadata)
                    yield copy.deepcopy(data)
                except:
                    print("not ok")


class WebDatasetPackedDataset(IterableDataset):
    """
    A highly efficient WebDataset loader for large-scale pre-training.
    It streams data from .tar files and is optimized for multi-worker data loading.

    This dataset yields tuples of: (normalized_image_tensor, caption_str, metadata_dict)

    Args:
        urls (Iterable[str]): A list of URLs or glob patterns pointing to .tar files.
        resolution (int): The target short-side resolution for image resizing.
        random_crop (bool): If True, applies random cropping; otherwise, center cropping.
        shuffle_buffer (int): The size of the buffer for shuffling samples. A larger buffer
                              provides better randomness but uses more memory. 0 to disable.
        sample_shuffle (bool): If True, enables sample shuffling within the buffer.
        repeat (bool): If True, the dataset will loop indefinitely. Set to True for training.
    """
    def __init__(
        self,
        urls: Iterable[str],
        resolution: int = 256,
        random_crop: bool = False,
        shuffle_buffer: int = 1000,
        sample_shuffle: bool = True,
        repeat: bool = True,
    ):
        super().__init__()
        # Ensure urls is a list, even if a single string is passed
        if isinstance(urls, str):
            urls = [urls]
        print("INFO: Resolving dataset URLs and glob patterns...")
        
        tar_files = []
        for url in urls:
            tar_files.extend(glob.glob(os.path.join(url, "**/*.tar"), recursive=True))
            tar_files.extend(glob.glob(os.path.join(url, "**/*.tar.gz"), recursive=True))
        num_tars = len(tar_files)   
        if num_tars == 0:
            raise ValueError(f"No tar files found. Please check your URLs/patterns: {urls}")
            
        print(f"INFO: Found {num_tars} tar files to stream from.")
        
        self.urls = tar_files

        self.resolution = resolution
        self.normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.crop_fn = random_crop_fn if random_crop else center_crop_fn
        self.shuffle_buffer = shuffle_buffer
        self.sample_shuffle = sample_shuffle
        self.repeat = repeat

        # A simple text augmentation to add variety to short captions
        self.prefix_template = [
            "A photo of ", "A picture of ", "A visual representation of ",
            "A image of ", "A scene of ", "A view of ", "A depiction of ",
        ]

    def _extract_image_from_sample(self, sample: Dict) -> Optional[Image.Image]:
        """Extracts the PIL image from a sample dict using the 'jpg' key."""
        if 'jpg' in sample:
            img_key = 'jpg'
        elif 'output_image' in sample:
            img_key = 'output_image'
        else:
            return None
        
        image_data = sample[img_key]
        if isinstance(image_data, Image.Image):
            return image_data
        if isinstance(image_data, (bytes, bytearray)):
            try:
                return Image.open(io.BytesIO(image_data))
            except Exception:
                return None # Return None if bytes are not a valid image
        return None

    def _extract_caption_from_sample(self, sample: Dict) -> str:
        """Extracts the caption from a sample dict using the 'txt' key."""
        if 'txt' in sample:
            text_key = 'txt'
        elif "input_prompt" in sample:
            text_key = 'input_prompt'
        else:
            return "" # Return an empty string if caption is missing
        
        caption_data = sample[text_key]
        if isinstance(caption_data, (bytes, bytearray)):
            return caption_data.decode("utf-8", errors="ignore")
        if isinstance(caption_data, str):
            return caption_data
        return str(caption_data)

    def _process_pil(self, pil_image: Image.Image):
        """
        Processes a PIL image: converts to RGB, resizes, crops, and normalizes.
        Returns both the normalized tensor and the pre-normalization tensor.
        """
        pil_image = pil_image.convert('RGB')
        
        # Skip images that are smaller than the target resolution
        if min(*pil_image.size) < self.resolution:
            return None
            
        # Resize, crop, and convert to tensor
        pil_image = resize(pil_image, self.resolution)
        pil_image = self.crop_fn(pil_image, self.resolution, self.resolution)
        raw_image_tensor = to_tensor(pil_image)
        normalized_image_tensor = self.normalize(raw_image_tensor)
        
        return normalized_image_tensor, raw_image_tensor

    def _make_pipeline(self, worker_id: int, num_workers: int):
        """
        Creates the data loading pipeline using webdataset.
        This pipeline is optimized for performance in a multi-worker setup.
        """
        # `resampled=self.repeat` is a robust way to handle repeating datasets
        # and shuffling the order of shards for each epoch.
        handler = wds.warn_and_continue
        dataset = wds.DataPipeline(
            wds.SimpleShardList(self.urls),
            # at this point we have an iterator over all the shards
            # this shuffles the shards
            wds.shuffle(100),
            # add wds.split_by_node here if you are using multiple nodes
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(),
            # this shuffles the samples in memory
            wds.shuffle(self.shuffle_buffer),
            # this decodes the images and json
            wds.decode("pil", handler=handler),
        )
                    
        return dataset

    def __iter__(self):
        """The iterator method that yields data samples."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process data loading
            worker_id, num_workers = 0, 1
        else:
            # Multi-process data loading
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # Create a unique pipeline for each worker
        pipeline = self._make_pipeline(worker_id, num_workers)

        for sample in pipeline:
            try:
                pil_image = self._extract_image_from_sample(sample)
                if pil_image is None:
                    print("skip image")
                    continue # Skip if sample has no valid image

                processed_data = self._process_pil(pil_image)
                if processed_data is None:
                    continue # Skip if image processing fails (e.g., too small)
                
                img_tensor, raw_image = processed_data
                caption = self._extract_caption_from_sample(sample)
                
                # Optional: Add a random prefix to short captions for augmentation
                if random.random() < 0.5 and len(caption.split()) < 30:
                    caption = random.choice(self.prefix_template) + caption

                metadata = {
                    "raw_image": raw_image, # Tensor before normalization
                    "prompt": caption,
                }
                yield (img_tensor, caption, metadata)
                
            except GeneratorExit:
                # This exception is raised when the consumer of the iterator stops.
                raise
            except Exception:
                # Catch-all for any other errors in a sample to make the stream robust.
                # In a real-world scenario, you might want to log this error.
                # e.g., print(f"Warning: Skipping a bad sample due to {e}")
                print("fail")
                continue
    
class WebDatasetPackedDataset_gpt(IterableDataset):
    """
    Stream webdataset (.tar/.gz/...) files using webdataset library.
    Yields tuples: (normalized_image_tensor, caption_str, metadata_dict)
    Arguments:
      - urls: list[str] or str (glob, tar path, or list)
      - caption_weight: dict mapping caption-field-name -> probability. If provided, tries to select field accordingly.
      - resolution: int target short-side before crop (same semantics as original)
      - random_crop: bool
      - shuffle_buffer: int for wds.shuffle (0 means no shuffle)
      - sample_shuffle: bool when True will call .shuffle() in pipeline
    """
    def __init__(
        self,
        urls: Iterable[str],
        caption_weight: Optional[Dict[str, float]] = None,
        resolution: int = 256,
        random_crop: bool = False,
        shuffle_buffer: int = 1000,
        sample_shuffle: bool = True,
        repeat: bool = True,
    ):
        super().__init__()
        if isinstance(urls, str):
            urls = [urls]
        self.urls = list(urls)
        if len(self.urls) == 0:
            raise ValueError("No webdataset urls provided.")
        self.resolution = resolution
        self.normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.crop_fn = random_crop_fn if random_crop else center_crop_fn
        self.caption_weight = caption_weight or {}
        self.prefix_template = [
            "A photo of ",
            "A picture of ",
            "A visual representation of ",
            "A image of ",
            "A scene of ",
            "A view of ",
            "A depiction of ",
        ]
        self.shuffle_buffer = shuffle_buffer
        self.sample_shuffle = sample_shuffle
        self.repeat = repeat

    def _extract_image_from_sample(self, sample: Dict):
        # webdataset with .decode("pil") will decode images into PIL objects for image keys.
        # We'll prefer already-decoded images; otherwise try bytes->PIL.
        # Look for common keys:
        possible_image_keys = ['jpg', 'png', 'jpeg', 'image', 'img', 'data']
        for k in possible_image_keys:
            if k in sample:
                v = sample[k]
                if isinstance(v, Image.Image):
                    return v
                if isinstance(v, (bytes, bytearray)):
                    try:
                        return Image.open(io.BytesIO(v))
                    except Exception:
                        pass
        # fallback: find first PIL or bytes among values
        for v in sample.values():
            if isinstance(v, Image.Image):
                return v
            if isinstance(v, (bytes, bytearray)):
                try:
                    return Image.open(io.BytesIO(v))
                except Exception:
                    continue
        # nothing found
        return None

    def _extract_caption_from_sample(self, sample: Dict):
        # If caption_weight provided, try to choose a key according to probabilities.
        if self.caption_weight:
            keys = list(self.caption_weight.keys())
            probs = list(self.caption_weight.values())
            # choose key (might not exist in sample)
            chosen_key = random.choices(keys, weights=probs, k=1)[0]
            if chosen_key in sample:
                val = sample[chosen_key]
                if isinstance(val, (bytes, bytearray)):
                    return val.decode("utf-8", errors="ignore")
                return str(val)
            # if chosen not present, fallthrough to generic search

        # generic search for text-like fields
        text_keys = ['txt', 'caption', 'text', 'json', 'meta', 'label']
        for k in text_keys:
            if k in sample:
                v = sample[k]
                if isinstance(v, (bytes, bytearray)):
                    return v.decode("utf-8", errors="ignore")
                return str(v)
        # fallback: any string/bytes value
        for v in sample.values():
            if isinstance(v, (bytes, bytearray)):
                return v.decode("utf-8", errors="ignore")
            if isinstance(v, str):
                return v
        return ""  # empty caption if none found

    def _process_pil(self, pil_image: Image.Image):
        pil_image = pil_image.convert('RGB')
        if min(*pil_image.size) < self.resolution:
            return None  # skip
        pil_image = resize(pil_image, self.resolution)
        pil_image = self.crop_fn(pil_image, self.resolution, self.resolution)
        raw_image = to_tensor(pil_image)
        normalized_image = self.normalize(raw_image)
        return normalized_image, raw_image

    def _make_pipeline(self, worker_id: int, num_workers: int):
        # Build webdataset pipeline; shard by worker_id/num_workers; optional shuffle; decode images to PIL
        urls = self.urls
        ds = wds.WebDataset(urls).decode("pil")
        # shard for multi-worker
        if num_workers > 1:
            ds = ds.shard(worker_id, num_workers)
        if self.sample_shuffle and self.shuffle_buffer > 0:
            ds = ds.shuffle(self.shuffle_buffer)
        if self.repeat:
            ds = ds.repeat()  # infinite stream
        return ds

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        ds = self._make_pipeline(worker_id, num_workers)

        # iterate forever (or until process killed)
        for sample in ds:
            # sample is a dict-like mapping keys->values
            try:
                pil_image = self._extract_image_from_sample(sample)
                if pil_image is None:
                    continue
                caption = self._extract_caption_from_sample(sample)
                # optionally add prefix for short captions (replicates original behavior)
                if random.random() < 0.5 and len(caption.split()) < 30:
                    caption = random.choice(self.prefix_template) + caption

                img_tensor, raw_image = self._process_pil(pil_image)
                if img_tensor is None:
                    continue

                metadata = {"raw_image": raw_image, "prompt": caption,}
                yield (img_tensor, caption, metadata)
            except GeneratorExit:
                raise
            except Exception:
                # ignore single-bad-sample errors to keep stream robust
                continue
