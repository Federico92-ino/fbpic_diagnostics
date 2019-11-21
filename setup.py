from setuptools import setup, find_packages
import numpy

__version__ = "0.0.1"

# with open('README.md') as f:
    # long_description = f.read()

requirements = ['numpy', 'matplotlib', 'openPMD-viewer', 'scipy']
setup(
    name='fbpic_diag',
    version=__version__,
    # description="",
    # long_description=long_description,
    # long_description_content_type='text/markdown',
    packages=find_packages('.'),
    author='Federico',
    # author_email='davide.terzani@ino.it',
    # maintainer='Davide Terzani',
    # maintainer_email='davide.terzani@ino.it',
    license='GNU GPLv3',
    # url='https://github.com/ALaDyn/ALaDyn',
    # classifiers=[
    #     'Programming Language :: Python',
    #     'Natural Language :: English',
    #     'Environment :: Console',
    #     'Intended Audience :: Science/Research',
    #     'Operating System :: OS Independent',
    #     'Topic :: Scientific/Engineering :: Physics',
    #     'Programming Language :: Python :: 3.7'],
    # ext_modules=cythonize(extensions),
    install_requires=[requirements],
    # ext_package='lampy/compiled_cython',
    # libraries=[lib_read_binary, lib_read_phase_space],
    include_dirs=[numpy.get_include()]
)
