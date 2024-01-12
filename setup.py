import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
	long_description = fh.read()

setuptools.setup(
	name = 'molecularsolidbuilder',
	version = '0.0.1',
	author = 'Pilsun Yoo',
	author_email = 'yoop@ornl.gov',
	description = 'MolecularSolidBuilder',
	long_description = long_description,
	long_description_content_type = 'text/markdown',
    url='https://github.com/MoleCraftHUB/MolecularSolidBuilder/',
	install_requires=['rdkit',
	                  'numpy',
					  'ase',
	                 ],
	project_urls = {
		"Bug Tracker": "package issues URL",
	},
	classifiers = [
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
	],
	package_dir = {"": "molecularsolidbuilder"},
	packages = setuptools.find_packages(where="molecularsolidbuilder"),
	python_requires = ">=3.6"
)
