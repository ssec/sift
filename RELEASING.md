# Releasing SIFT

1. Checkout master (`git checkout master`)
2. Pull from repository (`git pull`)
3. Run any necessary tests. For basic dependency checks `python sift -h`
   should suffice.
4. Run `loghub` and update the `CHANGELOG.md` file. If `loghub` is not
   installed, do so by running `pip install loghub`.

```bash
loghub ssec/sift -u <username> -st v1.0.5 -plg bug "Bugs fixed" -plg enhancement "Features added" -plg documentation "Documentation changes" -plg backwards-incompatibility "Backwards incompatible changes"
```

5. Commit the change log changes.

6. Create a tag with the new version number, starting with a 'v', eg:

```bash
git tag -a 1.0.5 -m "Version 1.0.5"
```

See [semver.org](http://semver.org/) on how to write a version number.

6. Push changes to github `git push --follow-tags`
7. Create conda package and installers for all supported platforms. See
   [the wiki page](https://github.com/ssec/sift/wiki/conda-package-building#create-a-conda-package)
   for detailed instructions on generating the conda package and other
   special instructions. Typically this can be done with one command:
   
```bash
python build_release.py
```

Note that by default this will try to upload the installers and conda packages
to the appropriate servers to be hosted or uploaded to FTP. If you do not have
an account on these servers or do not which to upload/host the files then use
the `--no-conda-upload` and/or `--no-installer-upload` flags.

8. Create a release of the package on
   [github](https://github.com/ssec/sift/releases) by drafting a new release
   and copying the release notes from the changelog (see above).