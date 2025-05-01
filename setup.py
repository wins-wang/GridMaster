from setuptools import setup, find_packages

# Read README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
setup(
    name='gridmaster',
    version='0.2.0',
    description='A lightweight multi-stage grid search AutoML framework for classifiers. GridMaster v0.2.0: Log-scale search + plot customization features',
    author='Winston Wang',
    author_email='74311922+wins-wang@users.noreply.github.com',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/wins-wang/GridMaster.git',
    license='MIT',
    packages=find_packages(exclude=['examples*', '*.examples', '*.examples.*']),
    include_package_data=True,
    install_requires=[
        'scikit-learn>=1.0',
        'xgboost>=1.5',
        'lightgbm>=3.3',
        'catboost>=1.0',
        'pandas>=1.1',
        'numpy>=1.19',
        'matplotlib>=3.3',
        'joblib>=1.0'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    python_requires='>=3.7',
)
