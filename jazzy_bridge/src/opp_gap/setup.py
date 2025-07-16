from setuptools import setup

package_name = 'opp_gap'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Steven Gong',
    maintainer_email='s36gong@uwaterloo.ca',
    description='f1tenth opp_gap lab',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'reactive_node = opp_gap.reactive_node:main',
        ],
    },
)
