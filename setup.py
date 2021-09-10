import os
import setuptools
import versioneer


HERE = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(HERE, 'README.md'), 'r') as fid:
	LONG_DESCRIPTION = fid.read()

with open("requirements.txt", "r") as fh:
    requirements = [line.strip() for line in fh]

setuptools.setup(
    name="abpimputation",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Brian Hill",
    author_email="brian.l.hill11@gmail.com",
    description="abpimputation: Imputation of the continuous arterial line \
        blood pressure waveform from non-invasive measurements \
        using deep learning",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/brianhill11/ABPImputation",
    packages=setuptools.find_packages(),
    package_data={'abpimputation': []},
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.8",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires='>=3.8',
    install_requires=requirements,
)