import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    required = f.read().splitlines()

setuptools.setup(
    name="MovNEye",
    version="0.1",
    author="Simone Testa",
    author_email="simonetesta994@gmail.com",
    description="Converting static images to event-based data using an active neuromorphic vision setup guided by bio-inspired Fixational Eye Movements.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/simo-net/MovNEye",
    include_package_data=False,
    packages=setuptools.find_packages(),
    setup_requires=["pyyaml", "scikit-build", "tensorflow"],
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Linux",
        "Intended Audience :: Science/Research",
    ],
)
