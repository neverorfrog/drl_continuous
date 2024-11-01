from setuptools import find_packages, setup

# Read the requirements from the requirements.txt file
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="drl_continuous",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
    description="A package for deep reinforcement learning in continuous action spaces",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/drl-continuous",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
