from setuptools import setup
import versioneer

setup(
    # Self-descriptive entries which should always be present
    name='mdance',    
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)