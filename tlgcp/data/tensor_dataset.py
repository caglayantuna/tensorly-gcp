import requests
import scipy.io
from io import BytesIO
import imageio
import numpy as np
from zipfile import ZipFile
import gzip
import tensorly as tl
import numpy as np
import pandas
from os.path import dirname


class Bunch(dict):
    """ A Bunch, exposing dict keys as a keys() method.
    Definition from scikit-learn. """

    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setstate__(self, state):
        pass


def load_indian_pines():
    """
    Loads indian pines hyperspectral data from th website and returns it as a tensorly tensor without storing the data
    in the hard drive. This dataset could be useful for non-negative constrained decomposition methods and
    classification/segmentation applications with tha available ground truth in
    http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat.
    """

    url = 'http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat'
    r = requests.get(url, allow_redirects=True)
    image = scipy.io.loadmat(BytesIO(r.content))['indian_pines_corrected']
    reference = "Baumgardner, M. F., Biehl, L. L., Landgrebe, D. A. (2015). 220 Band AVIRIS Hyperspectral " \
                "Image Data Set: June 12, 1992 Indian Pine Test Site 3. Purdue University Research Repository. " \
                "doi:10.4231/R7RX991C"
    licence = "Licensed under Attribution 3.0 Unported (https://creativecommons.org/licenses/by/3.0/legalcode)"
    desc =  "Airborne Visible / Infrared Imaging Spectrometer (AVIRIS)  hyperspectral sensor data (aviris.jpl.nasa.gov/) " \
            "were acquired on June 12, 1992 over the Purdue University Agronomy farm northwest " \
            "of West Lafayette and the surrounding area. This scene consists of 145 times 145 pixels and 220 spectral " \
            "reflectance bands in the wavelength range 0.4–2.5 10^(-6) meters."
    return Bunch(
        tensor=np.array(image, "float"),
        dims=["Spatial dimension", "Spatial dimension", "Hyperspectral bands"],
        reference=reference,
        DESC=desc,
        LICENCE=licence)


def load_kinetic():
    """
    Loads kinetic fluorescence dataset from website and returns it as tensorly tensor without storing the data
    in the hard drive.The data is well suited for Parafac and multi-way partial least squares regression (N-PLS).

    """
    url = 'http://models.life.ku.dk/sites/default/files/Kinetic_Fluor.zip'
    r = requests.get(url, allow_redirects=True)
    zip = ZipFile(BytesIO(r.content))
    tensor = scipy.io.loadmat(zip.open('Xlarge.mat'))['Xlarge']
    tensor[np.isnan(tensor)] = 0
    reference = "Nikolajsen, R. P., Booksh, K. S., Hansen, Å. M., & Bro, R. (2003). \
                Quantifying catecholamines using multi-way kinetic modelling. \
                Analytica Chimica Acta, 475(1-2), 137-150."
    licence = "http://www.models.life.ku.dk/datasets. All downloadable material listed on these pages - " \
              "appended by specifics mentioned under " \
              "the individual headers/chapters - is available for public use. " \
              "Please note that while great care has been taken, the software, code and data are provided" \
              "as is and that Q&T, LIFE, KU does not accept any responsibility or liability."
    desc = "A four-way data set with the modes: Concentration, excitation wavelength, emission wavelength and time"

    return Bunch(
        tensor=tensor,
        dims=["Measurements", "Emissions", "Excitations", "Time points"],
        reference=reference,
        DESC=desc,
        LICENCE=licence)


def load_rainfall():
    """
    (TODO)
    """
    path = dirname(__file__)
    df = pandas.read_csv(path + "/rainfall_india.csv")
    years = df.YEAR.unique()
    states = df.SUBDIVISION.unique()
    tensor = tl.zeros([36, 115, 12])
    for i in range(115):
        for j in range(36):
            c = df[df.SUBDIVISION == states[j]].to_numpy()
            if years[i] in c[:, 1]:
                row = np.where(c[:, 1] == years[i])
                tensor[j, i, :] = c[row, 2:14]
    return Bunch(
        tensor=tensor,
        dims=["division", "todo", "month"],
        reference="https://www.kaggle.com/datasets/rajanand/rainfall-in-india",
        DESC="This data set contains monthly rainfall detail of 36 meteorological sub-divisions of India",
        LICENCE="CC BY-SA 4.0, https://data.gov.in/government-open-data-license-india")


def load_chicago_crime():
    """
    (TODO)
    """
    url = 'https://s3.us-east-2.amazonaws.com/frostt/frostt_data/chicago-crime/comm/chicago-crime-comm.tns.gz'
    r = requests.get(url, allow_redirects=True)
    zip = gzip.open(BytesIO(r.content))
    array = [[int(x) for x in line.split()] for line in zip]
    tensor = np.zeros([6186, 24, 77, 32])
    for i in range(len(array)):
        tensor[array[i][0]-1, array[i][1]-1, array[i][2]-1, array[i][3]-1] = array[i][4]
    reference = "Smith, S., Huang, K., Sidiropoulos, N. D., & Karypis, G. (2018, May). "\
                "Streaming tensor factorization for infinite data sources. In "\
                "Proceedings of the 2018 SIAM International Conference on Data Mining"\
                "(pp. 81-89). Society for Industrial and Applied Mathematics"
    desc = "Streaming Constraint Sparse"
    licence="Licence is needed"
    return Bunch(
        tensor=tensor,
        dims=["Day", "Hour", "Community", "Crime Type"],
        reference=reference,
        DESC=desc,
        LICENCE=licence)


def get_tensor(name):
    """
    Returns selected tensor among the possible options:
    {"indian_pines", "rainfall", "chicago_crime", "kinetic"}

    Parameters
    ----------
    name : string

    Returns
    -------
    tensor

    """
    if name == "indian_pines":
        return load_indian_pines()['tensor']
    elif name == "kinetic":
        return load_kinetic()['tensor']
    elif name == "chicago_crime":
        return load_chicago_crime()['tensor']
    elif name == "rainfall":
        return load_rainfall()['tensor']
