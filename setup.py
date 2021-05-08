""" Setup file """

from setuptools import setup, find_packages

setup(name='mtcnn',
      version='0.1.0',
      description='Pytorch implementation of MTCNN',
      url='https://github.com/iorivn/mtcnn',
      author='Anh Dang',
      author_email='me@anhdang.info',
      license='MIT License',
      packages=find_packages(),
      zip_safe=False,
      classifiers=[
          "License :: OSI Approved :: MIT License",
          "Programming Language :: Python :: 3.8"
      ],
      python_requires=">=3.8",
      install_requires=['numpy', 'torch', 'torchvision']
      )
