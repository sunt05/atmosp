from setuptools import setup
try:  # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:  # for pip <= 9.0.3
    from pip.req import parse_requirements

# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements('requirements.txt', session=False)

# reqs is a list of requirement
# e.g. ['django==1.5.1', 'mezzanine==1.4.6']
reqs = [str(ir.req) for ir in install_reqs]

setup(
    name='atmosp',
    packages=['atmosp'],
    version='0.2.10',
    description='Atmospheric sciences utility library',
    author='Ting Sun',
    author_email='sunting.05@gmail.com',
    install_requires=reqs,
    url='https://github.com/sunt05/atmosp',
    python_requires='~=3.6',
    keywords=['atmosp', 'atmospheric', 'equations', 'geoscience', 'science'],
    classifiers=[
        # 'Programming Language:: Python:: 3:: Only',
        # 'Programming Language :: Python :: 3.6',
        # 'Programming Language :: Python :: 3.7',
        # 'Intended Audience :: Education',
        # 'Intended Audience :: Science/Research',
        # 'Operating System :: MacOS :: MacOS X',
        # 'Operating System :: Microsoft :: Windows',
        # 'Operating System :: POSIX :: Linux',
    ],
    license='MIT',
)
