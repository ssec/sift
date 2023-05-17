# Releasing SIFT

The following instructions will walk you through making a release of the
SIFT application and python library. These instructions assume that you
already have the SIFT git repository cloned from GitHub and that the
`origin` git remote is pointing to the `ssec/sift` repository. Instructions
must be adjusted if `origin` points to your fork of the `sift` repository.

1. Make sure you are on the master branch (`git checkout master`)
2. Pull the most recent changes (`git pull`)
3. Run any necessary tests. For basic dependency checks `python sift -h`
   should suffice.
4. Run `loghub` and update the `CHANGELOG.md` file. If `loghub` is not
   installed, do so by running `pip install loghub`.

   ```bash
   loghub ssec/sift --token $LOGHUB_GITHUB_TOKEN -st <previous_version_tag> -plg bug "Bugs fixed" -plg enhancement "Features added" -plg documentation "Documentation changes" -plg backwards-incompatibility "Backwards incompatible changes"
   ```

5. Commit the changelog changes.

6. Bump the version of the package:

   ```bash
   python setup.py bump --new-version 1.0.5 -c -t
   ```

   See [semver.org](http://semver.org/) on how to write a version number.

7. Push changes to github `git push --follow-tags`

8. Create a release of the package on
   [github](https://github.com/ssec/sift/releases) by drafting a new release
   and copying the release notes from the changelog (see above).
