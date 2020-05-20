import tvm
from tvm.relay import testing

def get_network(name, batch_size, dtype='float32'):
        """Get the symbol definition and random weight of a network

        Parameters
        ----------
        name: str
            The name of the network, can be 'resnet-18', 'resnet-50', 'vgg-16', 'inception_v3', 'mobilenet', ...
        batch_size: int
            batch size
        dtype: str
            Data type

        Returns
        -------
        net: relay.Module
            The relay function of network definition
        params: dict
            The random parameters for benchmark
        input_shape: tuple
            The shape of input tensor
        output_shape: tuple
            The shape of output tensor
        """
        input_shape = (batch_size, 3, 224, 224)
        output_shape = (batch_size, 1000)

        if name == 'mobilenet':
            net, params = testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype)
        elif name == 'inception_v3':
            input_shape = (batch_size, 3, 299, 299)
            net, params = testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
        elif "resnet" in name:
            n_layer = int(name.split('-')[1])
            net, params = testing.resnet.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
        elif "vgg" in name:
            n_layer = int(name.split('-')[1])
            net, params = testing.vgg.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
        elif "densenet" in name:
            n_layer = int(name.split('-')[1])
            net, params = testing.densenet.get_workload(densenet_size=n_layer, batch_size=batch_size, dtype=dtype)
        elif "squeezenet" in name:
            version = name.split("_v")[1]
            net, params = testing.squeezenet.get_workload(batch_size=batch_size, version=version, dtype=dtype)
        else:
            raise ValueError("Unsupported network: " + name)

        return net, params, input_shape, output_shape
