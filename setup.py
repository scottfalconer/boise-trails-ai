import pathlib
from setuptools import setup, find_packages

here = pathlib.Path(__file__).parent

# Read requirements from requirements.txt
req_file = here / "requirements.txt"
with req_file.open() as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith('#')
    ]

setup(
    name="boise-trails-ai",
    version="0.1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=requirements,
)
