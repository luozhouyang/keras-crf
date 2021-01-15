import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="keras_crf",
    version="0.1.0",
    description="A more elegant and convenient CRF built on tensorflow-addons.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/luozhouyang/keras-crf",
    author="ZhouYang Luo",
    author_email="zhouyang.luo@gmail.com",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        'tensorflow',
        'tensorflow-addons',
        'seqeval'
    ],
    extras_require={
        
    },
    license="Apache Software License",
    classifiers=(
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    )
)