from setuptools import setup, find_packages
import versioneer


setup(
    name='mdance',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    package_dir={'': 'src'},
    packages=find_packages(where='src')
    
)