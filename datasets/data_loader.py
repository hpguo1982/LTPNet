from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.dataloader import _collate_fn_t, _worker_init_fn_t
from registers import AIITModel
from typing import Iterable, Sequence, Optional, Union

class AIITDataLoader(DataLoader, AIITModel):
    r"""
        Data loader. Combines a dataset and a sampler, and provides an iterable over
        the given dataset.

        The :class:`~torch.utils.data.DataLoader` supports both map-style and
        iterable-style datasets with single- or multi-process loading, customizing
        loading order and optional automatic batching (collation) and memory pinning.

        See :py:mod:`torch.utils.data` documentation page for more details.

        Args:
            dataset (Dataset): dataset from which to load the data.
            batch_size (int, optional): how many samples per batch to load
                (default: ``1``).
            shuffle (bool, optional): set to ``True`` to have the data reshuffled
                at every epoch (default: ``False``).
            sampler (Sampler or Iterable, optional): defines the strategy to draw
                samples from the dataset. Can be any ``Iterable`` with ``__len__``
                implemented. If specified, :attr:`shuffle` must not be specified.
            batch_sampler (Sampler or Iterable, optional): like :attr:`sampler`, but
                returns a batch of indices at a time. Mutually exclusive with
                :attr:`batch_size`, :attr:`shuffle`, :attr:`sampler`,
                and :attr:`drop_last`.
            num_workers (int, optional): how many subprocesses to use for data
                loading. ``0`` means that the data will be loaded in the main process.
                (default: ``0``)
            collate_fn (Callable, optional): merges a list of samples to form a
                mini-batch of Tensor(s).  Used when using batched loading from a
                map-style dataset.
            pin_memory (bool, optional): If ``True``, the data loader will copy Tensors
                into device/CUDA pinned memory before returning them.  If your data elements
                are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type,
                see the example below.
            drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
                if the dataset size is not divisible by the batch size. If ``False`` and
                the size of dataset is not divisible by the batch size, then the last batch
                will be smaller. (default: ``False``)
            timeout (numeric, optional): if positive, the timeout value for collecting a batch
                from workers. Should always be non-negative. (default: ``0``)
            worker_init_fn (Callable, optional): If not ``None``, this will be called on each
                worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
                input, after seeding and before data loading. (default: ``None``)
            generator (torch.Generator, optional): If not ``None``, this RNG will be used
                by RandomSampler to generate random indexes and multiprocessing to generate
                `base_seed` for workers. (default: ``None``)
            prefetch_factor (int, optional, keyword-only arg): Number of batches loaded
                in advance by each worker. ``2`` means there will be a total of
                2 * num_workers batches prefetched across all workers. (default value depends
                on the set value for num_workers. If value of num_workers=0 default is ``None``.
                Otherwise if value of num_workers>0 default is ``2``).
            persistent_workers (bool, optional): If ``True``, the data loader will not shutdown
                the worker processes after a dataset has been consumed once. This allows to
                maintain the workers `Dataset` instances alive. (default: ``False``)
            pin_memory_device (str, optional): the data loader will copy Tensors
                into device pinned memory before returning them if pin_memory is set to true.


        .. warning:: If the ``spawn`` start method is used, :attr:`worker_init_fn`
                     cannot be an unpicklable object, e.g., a lambda function. See
                     :ref:`multiprocessing-best-practices` on more details related
                     to multiprocessing in PyTorch.

        .. warning:: ``len(dataloader)`` heuristic is based on the length of the sampler used.
                     When :attr:`dataset` is an :class:`~torch.utils.data.IterableDataset`,
                     it instead returns an estimate based on ``len(dataset) / batch_size``, with proper
                     rounding depending on :attr:`drop_last`, regardless of multi-process loading
                     configurations. This represents the best guess PyTorch can make because PyTorch
                     trusts user :attr:`dataset` code in correctly handling multi-process
                     loading to avoid duplicate data.

                     However, if sharding results in multiple workers having incomplete last batches,
                     this estimate can still be inaccurate, because (1) an otherwise complete batch can
                     be broken into multiple ones and (2) more than one batch worth of samples can be
                     dropped when :attr:`drop_last` is set. Unfortunately, PyTorch can not detect such
                     cases in general.

                     See `Dataset Types`_ for more details on these two types of datasets and how
                     :class:`~torch.utils.data.IterableDataset` interacts with
                     `Multi-process data loading`_.

        .. warning:: See :ref:`reproducibility`, and :ref:`dataloader-workers-random-seed`, and
                     :ref:`data-loading-randomness` notes for random seed related questions.
        """
    def __init__(self, dataset,
                 batch_size: Optional[int] = 1,
                 shuffle: Optional[bool] = None,
                 sampler: Union[Sampler, Iterable, None] = None,
                 batch_sampler: Union[Sampler[Sequence], Iterable[Sequence], None] = None,
                 num_workers: int = 0, collate_fn: Optional[_collate_fn_t] = None,
                 pin_memory: bool = False, drop_last: bool = False,
                 timeout: float = 0, worker_init_fn: Optional[_worker_init_fn_t] = None,
                 multiprocessing_context=None, generator=None,
                 *, prefetch_factor: Optional[int] = None,
                 persistent_workers: bool = False,
                 pin_memory_device: str = ""):
        super().__init__(dataset,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         sampler=sampler,
                         batch_sampler=batch_sampler,
                         num_workers=num_workers,
                         collate_fn=collate_fn,
                         pin_memory=pin_memory,drop_last=drop_last,
                         timeout=timeout, worker_init_fn=worker_init_fn,
                         multiprocessing_context=multiprocessing_context,
                         generator=generator,
                         prefetch_factor=prefetch_factor,
                         persistent_workers=persistent_workers,
                         pin_memory_device=pin_memory_device)
