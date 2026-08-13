from setuptools import setup, find_packages

setup(
    name='multiagent-predictor',
    version='0.1.0',
    packages=find_packages(include=['agents', 'models', 'utils', 'tests', '*']),
    install_requires=[
        'yfinance>=0.2.31',
        'pandas>=2.0.0',
        'numpy>=1.22',
        'streamlit>=1.30.0',
        'scikit-learn>=1.2.0',
        'xgboost>=1.7.0',
        'shap>=0.42',
        'tensorflow>=2.9.0',
        'joblib>=1.3.0'
    ],
    entry_points={
        'console_scripts': [
            'multiagent-cli = cli:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
    author='Your Name',
    author_email='your.email@example.com',
    description='A multi-agent system for financial prediction using BiLSTM, Transformer, and Symbolic AI',
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.8',
)
