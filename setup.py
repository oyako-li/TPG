from setuptools import setup

setup(
    name="TPG",
    version="0.1.0",
    license='MIT',
    author="oyako-li",
    author_email="oyakoli3@gmail.com",
    description="TPG develop fork from https://github.com/Ryan-Amaral/PyTPG",
    url="https://github.com/oyako-li/TPG",
    packages=['_tpg', '_tpg.configuration'],
    install_requires=['numpy', 'unittest-xml-reporting', 'scipy']
    
)