#permet a Python d'importer les fichiers de src comme un module
from .data_preprocessing import load_and_preprocess_data, discretize_mpg
from .svm_classifier import SVMFromScratch, OneVsRestSVM
from .svm_regressor import LinearSVRFromScratch