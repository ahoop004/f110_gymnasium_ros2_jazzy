from setuptools import setup, find_packages

setup(
      name='f110_gym',
      version='0.2.1',
      author='Hongrui Zheng',
      author_email='billyzheng.bz@gmail.com',
      url='https://f1tenth.org',

      package_dir={'': 'gym'},
      install_requires=['gymnasium',
		        'numpy',
                        'Pillow',
                        'scipy',
                        'numba',
                        'pyyaml',
                        'pyglet',
                        # 'pyopengl',
                        ]
      )
