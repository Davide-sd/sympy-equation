import setuptools
import os


here = os.path.dirname(os.path.abspath(__file__))
version_ns = {}
with open(os.path.join(here, 'algebra_with_sympy', '_version.py')) as f:
    exec (f.read(), {}, version_ns)

with open("ReadMe.md", "r") as f:
    long_description = f.read()


setuptools.setup(
    name="Algebra_with_SymPy",
    url = "https://gutow.github.io/Algebra_with_Sympy/",
    version=version_ns["__version__"],
    description="Equations that can be algebraicly manipulated.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jonathan Gutow",
    author_email="gutow@uwosh.edu",
    keywords="symbolic algebra, computer algebra, CAS, calculations with "
             "units, sympy",
    license="GPL-3.0+",
    packages=setuptools.find_packages(),
    install_requires=[
        # 'jupyter>=1.0.0',
        # 'jupyterlab>=3.6',
        # 'sympy-for-algebra>=1.12'
        "param",
        "sympy",
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: End Users/Desktop',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent'
    ]
)
