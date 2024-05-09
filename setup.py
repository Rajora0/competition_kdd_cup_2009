from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='This project aims to develop a machine learning model to predict three consumer behaviors - account churn, propensity to adopt new products, and likelihood to buy upgrades - based on data provided by the French telecommunications company Orange. The data consists of 50,000 instances and 230 variables, and the goal is to create a model that maximizes the AUC metric, representing the trade-off between false positive and true positive rates. The project aims to enhance Orange's customer relationship strategies, enabling more informed decisions and service personalization.',
    author='Rafael Rangel',
    license='MIT',
)
