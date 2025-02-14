#!/usr/bin/env python

from setuptools import find_packages, setup

if __name__ == "__main__":
    setup(
        name="jscv",
        version='1.0',
        description='en: jscv;\n zh_CN:jscv',  # 包的概括描述
        author='Jenson Chou',
        author_email='jensonchou@163.com',
        keywords="deep learning",
        packages=find_packages(exclude=("data", "libc", "work_dir", "old")),
        classifiers=[
            "Development Status :: Beta",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3.5",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
        ],
        license="Apache License 2.0",
        zip_safe=False,
    )
