import torch
import speechbrain as sb

class CNN(torch.nn.Module):
    """CNN.

    Arguments
    ---------
    input_shape : tuple
        The shape of the input.
    cnn_temporal_kernels : int
        Number of kernels in the 2d temporal convolution.
    cnn_temporal_kernelsize : tuple
        Kernel size of the 2d temporal convolution.
    cnn_spatial_kernels : int
        Number of kernels in the 2d spatial depthwise convolution.
    cnn_poolsize: tuple
        Pool size.
    cnn_poolstride: tuple
        Pool stride.
    cnn_pool_type: string
        Pooling type.
    dropout: float
        Dropout probability.
    dense_max_norm: float
        Weight max norm of the fully-connected layer.
    dense_layer_neuron: int
        Number of neurons in hidden layer 1.
    dense_layer_2_neuron: int
        Number of neurons in hidden layer 2.
    dense_layer_3_neuron: int
        Number of neurons in hidden layer 2.
    dense_out_neuron: int
        Number of output neurons.

    Example
    -------
    >>> inp_tensor = torch.rand([1, 200, 32, 1])
    >>> model = CDNNet(input_shape=inp_tensor.shape)
    >>> output = model(inp_tensor)
    >>> output.shape
    torch.Size([1,4])
    """

    def __init__(
        self,
        input_shape=None,
        cnn_temporal_kernels=40,
        cnn_temporal_kernelsize=(13, 1),
        cnn_spatial_kernels=40,
        cnn_poolsize=(38, 1), # determines the spatial extent over which pooling is applied, affect downsampling factor
        cnn_poolstride=(8, 1), # determines the step size of the pooling window, affect the overlap and output size
        cnn_pool_type="max",
        dropout=0.5,
        dense_max_norm=0.25,
        dense_layer_1_neuron=12,
        dense_layer_2_neuron=8,
        dense_out_neuron=4,
    ):
        super().__init__()
        if input_shape is None:
            raise ValueError("Must specify input_shape")
        self.default_sf = 250  # sampling rate of the original publication (Hz)

        # T = input_shape[1]
        C = input_shape[2]
        # CONVOLUTIONAL MODULE
        self.conv_module = torch.nn.Sequential()
        # Temporal convolution
        self.conv_module.add_module(
            "conv_0",
            sb.nnet.CNN.Conv2d(
                in_channels=1,
                out_channels=cnn_temporal_kernels,
                kernel_size=cnn_temporal_kernelsize,
                padding="valid",
                bias=True,
                swap=True,
            ),
        )
        self.conv_module.add_module(
            "bnorm_0",
            sb.nnet.normalization.BatchNorm2d(
                input_size=cnn_temporal_kernels, momentum=0.01, affine=True,
            ),
        )
        # Spatial convolution
        self.conv_module.add_module(
            "conv_1",
            sb.nnet.CNN.Conv2d(
                in_channels=cnn_temporal_kernels,
                out_channels=cnn_spatial_kernels,
                kernel_size=(1, C),
                padding="valid",
                bias=False,
                swap=True,
            ),
        )
        self.conv_module.add_module(
            "bnorm_1",
            sb.nnet.normalization.BatchNorm2d(
                input_size=cnn_spatial_kernels, momentum=0.1, affine=True,
            ),
        )
        self.conv_module.add_module(
            "pool_1",
            sb.nnet.pooling.Pooling2d(
                pool_type=cnn_pool_type,
                kernel_size=cnn_poolsize,
                stride=cnn_poolstride,
                pool_axis=[1, 2],
            ),
        )
        self.conv_module.add_module(
            "dropout_1", torch.nn.Dropout(p=dropout),
        )
        # Shape of intermediate feature maps
        out = self.conv_module(
            torch.ones((1,) + tuple(input_shape[1:-1]) + (1,))
        )
        dense_input_size = self._num_flat_features(out)
        # DENSE MODULE
        self.dense_module = torch.nn.Sequential()
        self.dense_module.add_module(
            "flatten", torch.nn.Flatten(),
        )
        # FC - hidden layer 1
        self.dense_module.add_module(
            "fc_1",
            sb.nnet.linear.Linear(
                input_size=dense_input_size, n_neurons=dense_layer_1_neuron,
            ),
        )
        self.dense_module.add_module(
            "act_1", torch.nn.ELU(),
        )
        self.dense_module.add_module(
            "dropout_1", torch.nn.Dropout(p=dropout),
        )
        # FC- hidden layer 2        
        self.dense_module.add_module(
            "fc_2",
            sb.nnet.linear.Linear(
                input_size=dense_layer_1_neuron, n_neurons=dense_layer_2_neuron,
            ),
        )
        self.dense_module.add_module(
            "act_2", torch.nn.ELU(),
        )
        self.dense_module.add_module(
            "dropout_1", torch.nn.Dropout(p=dropout),
        )

        # FC - Output layer
        self.dense_module.add_module(
            "fc_out",
            sb.nnet.linear.Linear(
                input_size=dense_layer_2_neuron,
                n_neurons=dense_out_neuron,
                max_norm=dense_max_norm,
            ),
        )
        self.dense_module.add_module("act_out", torch.nn.LogSoftmax(dim=1))

    def _num_flat_features(self, x):
        """Returns the number of flattened features from a tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input feature map.
        """

        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        """Returns the output of the model.

        Arguments
        ---------
        x : torch.Tensor (batch, time, EEG channel, channel)
            Input to convolve. 4d tensors are expected.
        """
        x = self.conv_module(x)
        x = self.dense_module(x)
        return x