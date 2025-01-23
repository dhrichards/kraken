from setuptools import setup, find_packages

setup(
    name="kraken3",
    version="0.1.0",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'mpi4py',
    ],

    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GPL-3.0 License',  
        'Operating System :: OS Independent',        
        'Programming Language :: Python :: 3',
    ],
)