from setuptools import setup, find_packages

setup(name="sedmodel",
      version="0.1.1",
      python_requires='>=3.8',
      description="Slip Effective Diffusion model",
      license="MIT",
      packages=find_packages(),
      install_requires=[
          "numpy", "scipy",
          "eznukutils @ git+https://github.com/eznuk/eznukutils.git",
      ],
)