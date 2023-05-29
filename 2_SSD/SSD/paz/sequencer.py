from keras.utils import Sequence
import numpy as np

class SequentialProcessor(object):
    """Abstract class for creating a sequential pipeline of processors.

    # Arguments
        processors: List of instantiated child classes of ``Processor``
            classes.
        name: String indicating name of the processing unit.

    # Methods
        add()
        remove()
        pop()
        insert()
        get_processor()

    # Example
    ```python
    AugmentImage = SequentialProcessor()
    AugmentImage.add(pr.RandomContrast())
    AugmentImage.add(pr.RandomBrightness())
    augment_image = AugmentImage()

    transformed_image = augment_image(image)
    ```
    """
    def __init__(self, processors=None, name=None):
        self.processors = []
        if processors is not None:
            [self.add(processor) for processor in processors]
        self.name = name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        if name is None:
            name = self.__class__.__name__
        self._name = name

    def add(self, processor):
        """Adds a process to the sequence of processes to be applied to input.

        # Arguments
            processor: An instantiated child class of of ``Processor``.
        """
        self.processors.append(processor)

    def __call__(self, *args, **kwargs):
        # Manual change: updated the function to ignore any processing if there's no processors
        if len(self.processors) > 0:
            # first call can take list or dictionary values.
            args = self.processors[0](*args, **kwargs)
            # further calls can be a tuple or single values.
            for processor in self.processors[1:]:
                if isinstance(args, tuple):
                    args = processor(*args)
                else:
                    args = processor(args)
        return args

    def remove(self, name):
        """Removes processor from sequence

        # Arguments
            name: String indicating the process name
        """
        for processor in self.processors:
            if processor.name == name:
                self.processors.remove(processor)

    def pop(self, index=-1):
        """Pops processor in given index from sequence

        # Arguments
            index: Int.
        """
        return self.processors.pop(index)

    def insert(self, index, processor):
        """Inserts ``processor`` to self.processors queue at ``index``

        # Argument
            index: Int.
            processor: An instantiated child class of of ``Processor``.
        """
        return self.processors.insert(index, processor)

    def get_processor(self, name):
        """Gets processor from sequencer

        # Arguments
            name: String indicating the process name
        """
        for processor in self.processors:
            if processor.name == name:
                return processor

class SequenceExtra(Sequence):
    def __init__(self, pipeline, batch_size, as_list=False):
        if not isinstance(pipeline, SequentialProcessor):
            raise ValueError('``processor`` must be a ``SequentialProcessor``')
        self.output_wrapper = pipeline.processors[-1]
        self.pipeline = pipeline
        self.inputs_name_to_shape = self.output_wrapper.inputs_name_to_shape
        self.labels_name_to_shape = self.output_wrapper.labels_name_to_shape
        self.ordered_input_names = self.output_wrapper.ordered_input_names
        self.ordered_label_names = self.output_wrapper.ordered_label_names
        self.batch_size = batch_size
        self.as_list = as_list

    def make_empty_batches(self, name_to_shape):
        batch = {}
        for name, shape in name_to_shape.items():
            batch[name] = np.zeros((self.batch_size, *shape))
        return batch

    def _to_list(self, batch, names):
        return [batch[name] for name in names]

    def _place_sample(self, sample, sample_arg, batch):
        for name, data in sample.items():
            batch[name][sample_arg] = data

    def _get_unprocessed_batch(self, data, batch_index):
        batch_arg_A = self.batch_size * (batch_index)
        batch_arg_B = self.batch_size * (batch_index + 1)
        unprocessed_batch = data[batch_arg_A:batch_arg_B]
        return unprocessed_batch

    def __getitem__(self, batch_index):
        inputs = self.make_empty_batches(self.inputs_name_to_shape)
        labels = self.make_empty_batches(self.labels_name_to_shape)
        inputs, labels = self.process_batch(inputs, labels, batch_index)
        if self.as_list:
            inputs = self._to_list(inputs, self.ordered_input_names)
            labels = self._to_list(labels, self.ordered_label_names)
        return inputs, labels

    def process_batch(self, inputs, labels, batch_index=None):
        raise NotImplementedError

class ProcessingSequence(SequenceExtra):
    """Sequence generator used for processing samples given in ``data``.

    # Arguments
        processor: Function, used for processing elements of ``data``.
        batch_size: Int.
        data: List. Each element of the list is processed by ``processor``.
        as_list: Bool, if True ``inputs`` and ``labels`` are dispatched as
            lists. If false ``inputs`` and ``labels`` are dispatched as
            dictionaries.
    """
    def __init__(self, processor, batch_size, data, as_list=False):
        self.data = data
        super(ProcessingSequence, self).__init__(
            processor, batch_size, as_list)

    def __len__(self):
        return int(np.ceil(len(self.data) / float(self.batch_size)))

    def process_batch(self, inputs, labels, batch_index):
        unprocessed_batch = self._get_unprocessed_batch(self.data, batch_index)

        for sample_arg, unprocessed_sample in enumerate(unprocessed_batch):
            sample = self.pipeline(unprocessed_sample.copy())
            self._place_sample(sample['inputs'], sample_arg, inputs)
            self._place_sample(sample['labels'], sample_arg, labels)

        return inputs, labels
