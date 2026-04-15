# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Protocol."""

import copy
import io
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import numpy as np
import ray
import torch
from numpy.typing import NDArray
from tensordict import TensorDict
from torch.distributed import ProcessGroup
from torch.utils.data import DataLoader

from .utils.py_functional import union_two_dict

try:
    import tensordict

    tensordict.set_lazy_legacy(False).set()
except Exception:
    pass

__all__ = ["DataProto", "union_tensor_dict"]

def pad_dataproto_to_divisor(data: "DataProto", size_divisor: int) -> tuple["DataProto", int]:
    """Pad Dataproto To Divisor."""
    assert isinstance(data, DataProto), "data must be a DataProto"
    if len(data) % size_divisor != 0:
        pad_size = size_divisor - len(data) % size_divisor
        padding_protos = []
        remaining_pad = pad_size
        while remaining_pad > 0:
            take_size = min(remaining_pad, len(data))
            padding_protos.append(data[:take_size])
            remaining_pad -= take_size

        data_padded = DataProto.concat([data] + padding_protos)
    else:
        pad_size = 0
        data_padded = data

    return data_padded, pad_size

def unpad_dataproto(data: "DataProto", pad_size: int) -> "DataProto":
    """Unpad Dataproto."""
    if pad_size != 0:
        data = data[:-pad_size]

    return data

def union_tensor_dict(tensor_dict1: TensorDict, tensor_dict2: TensorDict) -> TensorDict:
    """Union Tensor Dict."""
    if tensor_dict1.batch_size != tensor_dict2.batch_size:
        raise ValueError(
            f"Two tensor dict must have identical batch size. Got {tensor_dict1.batch_size} and {tensor_dict2.batch_size}"
        )

    for key in tensor_dict2.keys():
        if key in tensor_dict1 and not torch.equal(tensor_dict1[key], tensor_dict2[key]):
            raise ValueError(f"Key already exists: {key}.")

        tensor_dict1[key] = tensor_dict2[key]

    return tensor_dict1

def union_numpy_dict(tensor_dict1: dict[str, NDArray], tensor_dict2: dict[str, NDArray]) -> dict[str, NDArray]:
    """Union Numpy Dict."""
    for key in tensor_dict2.keys():
        if key in tensor_dict1:
            assert isinstance(tensor_dict2[key], np.ndarray)
            assert isinstance(tensor_dict1[key], np.ndarray)
            if not np.all(tensor_dict1[key] == tensor_dict2[key]):
                raise ValueError(f"Key already exists: {key}.")

        tensor_dict1[key] = tensor_dict2[key]

    return tensor_dict1

def batch_collate(features: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Batch Collate."""
    if len(features) == 0:
        return {}

    batch_features = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            batch_features[key].append(value)

    return batch_features

def fold_batch_dim(data: "DataProto", new_batch_size: int):
    """Fold Batch Dim."""
    batch_size = data.batch.batch_size[0]

    assert batch_size % new_batch_size == 0

    tensor: TensorDict = data.batch
    non_tensor = data.non_tensor_batch

    tensor = tensor.view(new_batch_size, -1)
    tensor.auto_batch_size_(batch_dims=1)

    for key, value in non_tensor.items():
        non_tensor[key] = np.reshape(value, newshape=(new_batch_size, -1, *value.shape[1:]))

    return DataProto(batch=tensor, non_tensor_batch=non_tensor, meta_info=data.meta_info)

def collate_fn(data_items: list["DataProtoItem"]):
    """Collate Fn."""
    batch = []
    non_tensor_batch = []
    for data in data_items:
        batch.append(data.batch)
        non_tensor_batch.append(data.non_tensor_batch)

    batch = torch.stack(batch).contiguous()
    non_tensor_batch = batch_collate(non_tensor_batch)
    non_tensor_batch = {key: np.array(value, dtype=object) for key, value in non_tensor_batch.items()}
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

@dataclass
class DataProtoItem:
    """Data Proto Item."""
    batch: Optional[TensorDict] = None
    non_tensor_batch: dict[str, NDArray] = field(default_factory=dict)
    meta_info: dict[str, Any] = field(default_factory=dict)

@dataclass
class DataProto:
    """Data Proto."""

    batch: Optional[TensorDict] = None
    non_tensor_batch: dict[str, NDArray] = field(default_factory=dict)
    meta_info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.check_consistency()  # perform necessary checking

    def __len__(self) -> int:
        """Len."""
        if self.batch is not None:
            return self.batch.batch_size[0]
        elif self.non_tensor_batch is not None and len(self.non_tensor_batch) > 0:
            pivot_key = list(self.non_tensor_batch.keys())[0]
            return self.non_tensor_batch[pivot_key].shape[0]
        else:
            return 0

    def __getitem__(
        self, item: Union[int, slice, list[int], np.ndarray, torch.Tensor]
    ) -> Union["DataProto", "DataProtoItem"]:
        """Getitem."""
        if isinstance(item, slice):
            return self.slice_select(item.start, item.stop, item.step)

        if isinstance(item, (list, np.ndarray, torch.Tensor)):
            return self.index_select(item)

        if isinstance(item, (int, np.integer)):
            tensor_data = self.batch[item] if self.batch is not None else None
            non_tensor_data = {key: value[item] for key, value in self.non_tensor_batch.items()}
            return DataProtoItem(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=self.meta_info)

        raise TypeError(f"Indexing with {type(item)} is not supported.")

    def __getstate__(self) -> tuple[bytes, dict[str, NDArray], dict[str, Any]]:
        if self.batch is not None:
            batch_to_save: TensorDict = self.batch.contiguous()
            batch_to_save: TensorDict = batch_to_save.consolidate()
        else:
            batch_to_save = None

        buffer = io.BytesIO()
        torch.save(batch_to_save, buffer)
        buffer_bytes = buffer.getvalue()
        return buffer_bytes, self.non_tensor_batch, self.meta_info

    def __setstate__(self, data: tuple[bytes, dict[str, NDArray], dict[str, Any]]) -> None:
        batch_deserialized_bytes, non_tensor_batch, meta_info = data
        batch_deserialized = io.BytesIO(batch_deserialized_bytes)
        batch = torch.load(batch_deserialized, weights_only=False, map_location="cpu")
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch
        self.meta_info = meta_info

    def save_to_disk(self, filepath: str) -> None:
        """Save To Disk."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_disk(filepath: str) -> "DataProto":
        """Load From Disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            return data

    def print_size(self, prefix: str = "") -> None:
        """Print Size."""
        size_of_tensordict = 0
        if self.batch is not None:
            for tensor in self.batch.values():
                if isinstance(tensor, torch.Tensor):
                    size_of_tensordict += tensor.element_size() * tensor.numel()

        size_of_numpy_array = 0
        for value in self.non_tensor_batch.values():
            size_of_numpy_array += value.nbytes

        size_of_numpy_array /= 1024**3
        size_of_tensordict /= 1024**3

        message = f"Size of tensordict: {size_of_tensordict} GB, size of non_tensor_batch: {size_of_numpy_array} GB."
        print({prefix}, {message})

    def check_consistency(self):
        """Check Consistency."""
        if self.batch is not None:
            assert len(self.batch.batch_size) == 1, "only support num_batch_dims=1"

        if self.batch is not None and len(self.non_tensor_batch) != 0:
            # TODO: we can actually lift this restriction if needed
            assert len(self.batch.batch_size) == 1, "only support num_batch_dims=1 when non_tensor_batch is not empty."

            batch_size = self.batch.batch_size[0]
            for key, value in self.non_tensor_batch.items():
                assert len(value) == batch_size, f"key {key} length {len(value)} is not equal to bsz {batch_size}."

    @classmethod
    def from_single_dict(
        cls,
        data: dict[str, Union[torch.Tensor, NDArray]],
        meta_info: Optional[dict[str, Any]] = None,
    ) -> "DataProto":
        """From Single Dict."""
        tensors, non_tensors = {}, {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                tensors[key] = value
            elif isinstance(value, np.ndarray):
                non_tensors[key] = value
            else:
                raise ValueError(f"Unsupported type in data {type(value)}")

        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)

    @classmethod
    def from_dict(
        cls,
        tensors: Optional[dict[str, torch.Tensor]] = None,
        non_tensors: Optional[dict[str, NDArray]] = None,
        meta_info: Optional[dict[str, Any]] = None,
        num_batch_dims: int = 1,
    ) -> "DataProto":
        """From Dict."""
        assert num_batch_dims > 0, "num_batch_dims must be greater than zero"
        if non_tensors is not None:
            assert num_batch_dims == 1, "only support num_batch_dims=1 when non_tensors is not None."

        tensors = tensors or {}
        non_tensors = non_tensors or {}
        meta_info = meta_info or {}
        assert isinstance(tensors, dict) and isinstance(non_tensors, dict) and isinstance(meta_info, dict)

        # get and check batch size
        batch_size = None
        pivot_key = None
        for key, tensor in tensors.items():
            if batch_size is None:
                batch_size = tensor.shape[:num_batch_dims]
                pivot_key = key
            else:
                current_batch = tensor.shape[:num_batch_dims]
                assert batch_size == current_batch, (
                    f"Not all the tensor in tensors have the same batch size with batch_dims={num_batch_dims}. "
                    f"Got {pivot_key} has {batch_size}, {key} has {current_batch}."
                )

        for key, value in non_tensors.items():
            if not isinstance(value, np.ndarray) or value.dtype != np.dtype(object):
                non_tensors[key] = np.array(value, dtype=object)

        tensor_dict = TensorDict(source=tensors, batch_size=batch_size) if tensors else None
        return cls(batch=tensor_dict, non_tensor_batch=non_tensors, meta_info=meta_info)

    def to(self, device: torch.device, non_blocking: bool = False) -> "DataProto":
        """To."""
        if self.batch is not None:
            self.batch = self.batch.to(device, non_blocking=non_blocking)

        return self

    def select(
        self,
        batch_keys: Optional[list[str]] = None,
        non_tensor_batch_keys: Optional[list[str]] = None,
        meta_info_keys: Optional[list[str]] = None,
        deepcopy: bool = False,
    ) -> "DataProto":
        """Select."""
        # TODO (zhangchi.usc1992) whether to copy
        if batch_keys is not None:
            batch_keys = tuple(filter(lambda k: k in self.batch, batch_keys))
            sub_batch = self.batch.select(*batch_keys)
        else:
            sub_batch = self.batch

        if non_tensor_batch_keys is not None:
            # we must convert it to tuple to avoid the missing elements
            non_tensor_batch_keys = tuple(filter(lambda k: k in self.non_tensor_batch, non_tensor_batch_keys))
            non_tensor_batch = {k: v for k, v in self.non_tensor_batch.items() if k in non_tensor_batch_keys}
        else:
            non_tensor_batch = self.non_tensor_batch

        if deepcopy:
            non_tensor_batch = copy.deepcopy(non_tensor_batch)

        if meta_info_keys is not None:
            meta_info_keys = tuple(filter(lambda k: k in self.meta_info, meta_info_keys))
            sub_meta_info = {k: v for k, v in self.meta_info.items() if k in meta_info_keys}
        else:
            sub_meta_info = self.meta_info

        if deepcopy:
            sub_meta_info = copy.deepcopy(sub_meta_info)

        return DataProto(batch=sub_batch, non_tensor_batch=non_tensor_batch, meta_info=sub_meta_info)

    def index_select(self, index: Union[list[int], NDArray, torch.Tensor]) -> "DataProto":
        """Index Select."""
        if isinstance(index, list):
            index = np.array(index, dtype=bool if isinstance(index[0], bool) else np.int32)
        elif isinstance(index, torch.Tensor):
            index = index.detach().cpu().numpy()

        tensor_data = self.batch[index] if self.batch is not None else None
        non_tensor_data = {key: value[index] for key, value in self.non_tensor_batch.items()}
        return DataProto(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=self.meta_info)

    def slice_select(
        self, start: Optional[int] = None, end: Optional[int] = None, step: Optional[int] = None
    ) -> "DataProto":
        """Slice Select."""
        index = slice(start, end, step)
        tensor_data = self.batch[index] if self.batch is not None else None
        non_tensor_data = {key: value[index] for key, value in self.non_tensor_batch.items()}
        return DataProto(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=self.meta_info)

    def pop(
        self,
        batch_keys: Optional[list[str]] = None,
        non_tensor_batch_keys: Optional[list[str]] = None,
        meta_info_keys: Optional[list[str]] = None,
    ) -> "DataProto":
        """Pop."""
        assert batch_keys is not None
        non_tensor_batch_keys = non_tensor_batch_keys or []
        meta_info_keys = meta_info_keys or []

        tensors = {}
        for key in filter(lambda k: k in self.batch, batch_keys):
            tensors[key] = self.batch.pop(key)

        non_tensors = {}
        for key in filter(lambda k: k in self.non_tensor_batch, non_tensor_batch_keys):
            non_tensors[key] = self.non_tensor_batch.pop(key)

        meta_info = {}
        for key in filter(lambda k: k in self.meta_info, meta_info_keys):
            meta_info[key] = self.meta_info.pop(key)

        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)

    def rename(
        self, old_keys: Optional[Union[str, list[str]]] = None, new_keys: Optional[Union[str, list[str]]] = None
    ) -> "DataProto":
        """Rename."""

        def validate_input(keys):
            if keys is not None:
                if isinstance(keys, str):
                    keys = [keys]
                elif isinstance(keys, list):
                    pass
                else:
                    raise TypeError(f"keys must be a list or a string, but got {type(keys)}")
            return keys

        old_keys = validate_input(old_keys)
        new_keys = validate_input(new_keys)

        if len(new_keys) != len(old_keys):
            raise ValueError(
                f"new_keys and old_keys must have the same length, but got {len(new_keys)} and {len(old_keys)}"
            )

        self.batch.rename_key_(tuple(old_keys), tuple(new_keys))

        return self

    def union(self, other: "DataProto") -> "DataProto":
        """Union."""
        self.batch = union_tensor_dict(self.batch, other.batch)
        self.non_tensor_batch = union_numpy_dict(self.non_tensor_batch, other.non_tensor_batch)
        self.meta_info = union_two_dict(self.meta_info, other.meta_info)
        return self

    def make_iterator(
        self, mini_batch_size: int, epochs: int, seed: int = None, dataloader_kwargs: dict[str, Any] = None
    ):
        """Make Iterator."""
        assert self.batch.batch_size[0] % mini_batch_size == 0, f"{self.batch.batch_size[0]} % {mini_batch_size} != 0"
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = None

        dataloader_kwargs = dataloader_kwargs or {}
        assert isinstance(dataloader_kwargs, dict)
        train_dataloader = DataLoader(
            dataset=self,
            batch_size=mini_batch_size,
            collate_fn=collate_fn,
            generator=generator,
            **dataloader_kwargs,
        )

        def get_data():
            for _ in range(epochs):
                for data in train_dataloader:
                    setattr(data, "meta_info", self.meta_info)
                    yield data

        return iter(get_data())

    def chunk(self, chunks: int) -> list["DataProto"]:
        """Chunk."""
        assert len(self) % chunks == 0, (
            f"only support equal chunk. Got size of DataProto {len(self)} and chunk {chunks}."
        )
        if self.batch is not None:
            batch_lst = self.batch.chunk(chunks=chunks, dim=0)
        else:
            batch_lst = [None for _ in range(chunks)]

        non_tensor_batch_lst = [{} for _ in range(chunks)]
        for key, value in self.non_tensor_batch.items():
            non_tensor_lst = np.array_split(value, chunks)
            for i in range(chunks):
                non_tensor_batch_lst[i][key] = non_tensor_lst[i]

        return [
            DataProto(batch=batch_lst[i], non_tensor_batch=non_tensor_batch_lst[i], meta_info=self.meta_info)
            for i in range(chunks)
        ]

    def split(self, split_size: int) -> list["DataProto"]:
        """Split."""
        assert len(self) % split_size == 0, (
            f"only support equal split. Got size of DataProto {len(self)} and split {split_size}."
        )
        chunks = len(self) // split_size
        return self.chunk(chunks)

    @staticmethod
    def concat(data: list["DataProto"]) -> "DataProto":
        """Concat."""
        batch_lst = [batch.batch for batch in data]
        new_batch = torch.cat(batch_lst, dim=0) if batch_lst[0] is not None else None
        non_tensor_batch = batch_collate([d.non_tensor_batch for d in data])
        for key, value in non_tensor_batch.items():
            non_tensor_batch[key] = np.concatenate(value, axis=0)

        return DataProto(batch=new_batch, non_tensor_batch=non_tensor_batch, meta_info=data[0].meta_info)

    def reorder(self, indices: torch.Tensor) -> None:
        """Reorder."""
        indices_np = indices.detach().numpy()
        self.batch = self.batch[indices]
        self.non_tensor_batch = {key: value[indices_np] for key, value in self.non_tensor_batch.items()}

    def repeat(self, repeat_times: int, interleave: bool = True) -> "DataProto":
        """Repeat."""
        if self.batch is not None:
            if interleave:  # interleave the data
                repeated_tensors = {
                    key: tensor.repeat_interleave(repeat_times, dim=0) for key, tensor in self.batch.items()
                }
            else:  # stack the data
                repeated_tensors = {
                    key: tensor.unsqueeze(0).expand(repeat_times, *tensor.shape).reshape(-1, *tensor.shape[1:])
                    for key, tensor in self.batch.items()
                }

            repeated_batch = TensorDict(
                source=repeated_tensors,
                batch_size=(self.batch.batch_size[0] * repeat_times,),
            )
        else:
            repeated_batch = None

        repeated_non_tensor_batch = {}
        for key, value in self.non_tensor_batch.items():
            if interleave:
                repeated_non_tensor_batch[key] = np.repeat(value, repeat_times, axis=0)
            else:
                repeated_non_tensor_batch[key] = np.tile(value, (repeat_times,) + (1,) * (value.ndim - 1))

        return DataProto(
            batch=repeated_batch,
            non_tensor_batch=repeated_non_tensor_batch,
            meta_info=self.meta_info,
        )

@dataclass
class DataProtoFuture:
    """Data Proto Future."""

    collect_fn: Callable
    futures: list[ray.ObjectRef]
    dispatch_fn: Callable = None

    @staticmethod
    def concat(data: list[ray.ObjectRef]) -> "DataProtoFuture":
        output = DataProtoFuture(collect_fn=DataProto.concat, futures=data)
        return output

    def chunk(self, chunks: int) -> list["DataProtoFuture"]:
        from functools import partial

        arg_future_lst = []
        for i in range(chunks):
            # note that we can't directly pass i and chunks
            def dispatch_fn(x, i, chunks):
                return x.chunk(chunks=chunks)[i]

            arg_future = DataProtoFuture(
                collect_fn=self.collect_fn, dispatch_fn=partial(dispatch_fn, i=i, chunks=chunks), futures=self.futures
            )
            arg_future_lst.append(arg_future)
        return arg_future_lst

    def get(self):
        """Get."""
        outputs = ray.get(self.futures)  # dp_size
        for output in outputs:
            assert isinstance(output, DataProto)

        outputs = self.collect_fn(outputs)  # select dp, concat
        if self.dispatch_fn is not None:
            outputs = self.dispatch_fn(outputs)  # split in batch dim, select using dp

        return outputs

def allgather_dict_tensors(
    tensors: Union[dict[str, torch.Tensor], TensorDict], size: int, group: ProcessGroup, dim: int = 0
) -> Union[dict[str, torch.Tensor], TensorDict]:
    """Allgather Dict Tensors."""
    if isinstance(tensors, TensorDict):
        is_tensor_dict = True
        tensors_as_dict = tensors.to_dict()
    else:
        tensors_as_dict = tensors
        is_tensor_dict = False

    output = {}
    sorted_keys = sorted(tensors_as_dict.keys())
    for key in sorted_keys:
        value = tensors_as_dict[key]
        output[key] = [torch.empty_like(value) for _ in range(size)]
        torch.distributed.all_gather(output[key], value, group=group, async_op=False)
        output[key] = torch.cat(output[key], dim=dim)

    if is_tensor_dict:
        output = TensorDict(source=output, batch_size=tensors.batch_size[0] * size)

    return output

def all_gather_data_proto(data: DataProto, size: int, group: ProcessGroup) -> None:
    """All Gather Data Proto."""
    # Note that this is an inplace operator just like torch.distributed.all_gather
    prev_device = data.batch.device
    data.batch = data.batch.cuda(device=torch.cuda.current_device())
    data.batch = allgather_dict_tensors(data.batch.contiguous(), size=size, group=group, dim=0)
    data.batch = data.batch.to(prev_device)
    # all gather non_tensor_batch
    all_non_tensor_batch = [None for _ in range(size)]
    torch.distributed.all_gather_object(all_non_tensor_batch, data.non_tensor_batch, group=group)
    data.non_tensor_batch = {k: np.concatenate([d[k] for d in all_non_tensor_batch]) for k in data.non_tensor_batch}
