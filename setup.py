import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="citlalicue",
    version="0.3.3",
    author="Oscar Barragán",
    author_email="oscaribv@gmail.com",
    description="Create stellar light curves!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/oscaribv/citlalicue",
    packages=setuptools.find_packages(),
     install_requires = ["numpy",
    "matplotlib",
    "scipy",
    "pytransit",
    "george",
    "emcee"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3',
)


