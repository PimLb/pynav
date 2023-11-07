from setuptools import setup, find_packages


setup(
    name="pynav",
    version="1.1.0",
    description="pynav: Navigation Library",
    python_requires='>=3.9',
    setup_requires=[
        'setuptools>=18.0'
    ],
    packages=find_packages(),
    install_requires=['numpy'],
    include_package_data=True,
)