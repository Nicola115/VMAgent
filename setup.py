from setuptools import setup, find_packages

setup(name='schedgym',
      version='0.0.1',
      description='Virtual Machine (Scheduling) GYM',
      url='https://github.com/mail-ecnu/VMAgent',
      author='Jarvis',
      author_email='2667356534@qq.com',
      license='MIT',
      packages=find_packages(include=['schedgym', 'schedgym*']),
      include_package_data=True,
      zip_safe=False,
      python_requires='>=3.6',
      install_requires=[
            'gym',
            'numpy',
            'stable-baselines',
            'pandas'
      ]
)
