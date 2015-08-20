from distutils.core import setup


setup(name='cspov',
      version='0.1',
      description="Fluid high resolution satellite and meteorological imagery viewer",
      author='Ray Garcia, SSEC',
      author_email='ray.garcia@ssec.wisc.edu',
      url='https://www.ssec.wisc.edu/',
      zip_safe=False,
      include_package_data=True,
      install_requires=['scipy','vispy','numpy', 'OpenGL', 'PyQt4', 'netCDF4', 'h5py'],
      packages=['distutils', 'distutils.command'],
      entry_points = {'console_scripts' : ['cspov = cspov.main:main']}
     )
