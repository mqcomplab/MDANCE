from importlib.metadata import version, PackageNotFoundError


try:
    __version__ = version("mdance")
except PackageNotFoundError:
    pass

__author__ = 'Lexin Chen'
__author_email__ = 'le.chen@ufl.edu'
__description__ = 'a flexible n-ary clustering package for molecular dynamics trajectories.'
__url__ = 'https://github.com/mqcomplab/MDANCE'
__license__ = 'MIT'
__doc__ = 'https://mdance.readthedocs.io/en/latest/'
__package__ = 'MDANCE'