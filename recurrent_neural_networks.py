class sequence:
    """
    A utility class for creating sequences from input data.

    Sequences are created by sliding a window of specified length over the input data.
    Each sequence consists of multiple consecutive elements from the input data.

    The input data and corresponding target values are returned as 3D tensors, where
    the first dimension represents the number of sequences, the second dimension represents
    the length of each sequence, and the third dimension represents the number of features.

    Attributes:
        None

    Methods:
        create(data, length):
            Create sequences from input data.

    Usage:
        Call the create() method to create sequences from the input data.

    Example:
        >>> sequences.create([1, 2, 3, 4, 5], 3)
        (array([[[1],
                 [2],
                 [3]],
                [[2],
                 [3],
                 [4]]]),
         array([[4],
                [5]]))
    """

    @staticmethod
    def create(data, length):
        """
        Create sequences from input data.

        Args:
            data (list or np.ndarray): The input data as a sequence.
            length (int): The length of each sequence to create.

        Returns:
            np.ndarray: The input data divided into sequences of the specified length.
            np.ndarray: The corresponding target values for each sequence.
        """
        import numpy as np

        X = []
        y = []
        for i in range(len(data) - length):
            X.append(data[i:i+length])
            y.append(data[i+length])
        return np.array(X), np.array(y)
