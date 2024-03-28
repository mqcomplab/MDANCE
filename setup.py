from setuptools import setup,find_packages

setup(name='mdance',
    version='0.2.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'}, 
    install_requires=[
        'numpy',
        'MDAnalysis',
        'shapeGMMTorch',
        'torch',
    ],
    entry_points={
        'console_scripts': [
            'MDANCE = mdance.main:entry',
        ]
    },
    author='',
    author_email='',
    description=""
)