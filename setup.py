import setuptools
# import runpy
# import os

# root = os.path.dirname(os.path.realpath(__file__))
# version = runpy.run_path(os.path.join(root, "visionart", "version.py"))["version"]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="VisionArt",
    version="0.1",
    author="Simone Testa",
    author_email="simonetesta994@gmail.com",
    description="Converting static images to neuromorphic data by using bio-inspired Fixational Eye Movements and a setup comprising a Pan-Tilt unit and an event-based vision camera.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/simo-net/MeyeView",
    include_package_data=False,
    packages=setuptools.find_packages(),
    install_requires=["numpy", "pyserial", "tqdm", "opencv-python", "matplotlib", "pyaer", "pandas", "scipy", "tensorflow-datasets", "pyyaml"],
    classifiers=[
        "Programming Language :: Python :: 3",
#         "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: Linux",
        "Intended Audience :: Science/Research",
    ],
)
