from ..tensor_dataset import load_indian_pines, load_kinetic


def test_indian_pines():
    """ Test that data import dimensions match. """
    data = load_indian_pines()

    tensor = data["tensor"]
    assert tensor.shape[0] == 145


def test_kinetic():
    """ Test that data import dimensions match. """
    data = load_kinetic()

    tensor = data["tensor"]
    assert tensor.shape[0] == 64
