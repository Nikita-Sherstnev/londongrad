import codecs

from setuptools import setup

name = 'londongrad'

with codecs.open('README.md', 'r', 'utf-8') as f:
    readme = f.read()

packages = [
    'londongrad'
]

install_requires = [
    'numpy>=1.23.2'
]

tests_require = [
    'coverage>=6.4.3',
    'flake8>=5.0.4',
    'pytest>=7.1.2'
]

setup(
    name=name,
    description='Deep Learning Framework',
    author='Sherstnyov',
    version='0.0.1',
    long_description_content_type='text/markdown',
    long_description=readme,
    package_dir={'londongrad': 'londongrad'},
    packages=packages,
    install_requires=install_requires,
    tests_require=tests_require,
    zip_safe=True,
    package_data={'': ['README.md']},
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.10',
        'Topic :: Internet',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    include_package_data=True
)