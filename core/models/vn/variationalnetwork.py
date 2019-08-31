import tensorflow as tf

class VariationalNetwork(object):
    """Variational Network class. Defines variational network for a given cell,
     defining a single stage, and a given number of stages.
    Args:
    cell: single stage, defined by the given application
    labels: number of stages
    num_cycles: number of cycles (optional). For all standard variational applications its default value 1 is used.
    parallel_iterations: number of parallel iterations used in while loop (optional) default=1
    swap_memory: Allow swapping of memory in while loop (optional). default=false
    """
    def __init__(self, cell, num_stages, num_cycles=1, parallel_iterations=1, swap_memory=False):
        # Basic computational graph of a vn cell
        self.cell = cell
        # Define the number of repetitions
        self._num_cycles = num_cycles
        self._num_stages = num_stages
        # Tensorflow specific details
        self._parallel_iterations = parallel_iterations
        self._swap_memory = swap_memory

    def get_outputs(self, x, stage_outputs=False):
        """ Get the outputs of the variational network.
        Args:
            stage_outputs: get all stage outputs (optional) default=False
        """
        outputs = [[] for self._num_stages]
        tmp = x
        for i in range(self._num_stages):
            for j in range(self.num_cycles):
                tmp = self.cell(tmp)
                outputs[i, j].append(tmp)
        
        if stage_outputs:
            return outputs
        else:
            return [out[-1] for out in self._outputs]
