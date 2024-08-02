from ._version import get_versions  # noqa: ABS101

versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions

__author__ = 'Lexin Chen'
__author_email__ = 'le.chen@ufl.edu'
__description__ = 'a flexible n-ary clustering package for molecular dynamics trajectories.'
__url__ = 'https://github.com/mqcomplab/MDANCE'
__license__ = 'MIT'
__doc__ = 'https://mdance.readthedocs.io/en/latest/'
__package__ = 'MDANCE'