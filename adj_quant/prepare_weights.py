import os

from scripts.effnetb0.effnetb0_data_weights import get_weights_for_effnetb0
import scripts.helpers as hp


def extract_weights_from_pb(filename, return_dict=False):
    """Extracts weights dictionary from any MNasNet model hosted at
    **www.tensorflow.org/lite/models**

    Parameters
    ----------
    filename: str
        Path to the TensorFlow model saved as a *.pb file.

    return_dict: bool
        Whether return the weights dictionary or not

    Returns
    -------
    dict: optional
        A dictionary containing the processed weights of the provided model
    """

    filename = os.path.realpath(filename)

    if os.path.isdir(filename):
        raise IsADirectoryError("Provided file name must refer to a file, not a folder")

    base_name = os.path.splitext(os.path.basename(filename))[0]
    dir_name = os.path.dirname(filename)

    model = hp.graph_from_graph_def(hp.load_pb(filename))

    model_weights = get_weights_for_mnasnet(model)

    hp.save_pickle(model_weights, os.path.join(dir_name, base_name + "_weights.pickle"))

    if return_dict:
        return model_weights


if __name__ == "__main__":
    import sys

    default_path = '/home/yf22/pretrained/efficient_b0/frozen_efficientnet_b0.pb'
    if len(sys.argv) < 2:
        extract_weights_from_pb(default_path)
    else:
    	extract_weights_from_pb(sys.argv[1])
