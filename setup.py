from setuptools import setup
import versioneer

setup(
    name='mdance',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=['mdance'],  # Adjust this to match your package structure
    package_dir={'': 'src'},  # Adjust this if your source code is in a different directory
)