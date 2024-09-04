Contributing
============

SIFT is an open source software project that welcomes contributions of any
kind. Contributions can range from letting the SIFT team know about bugs, to
submitting updates to documentation, to contributing new features to the
codebase. The documentation below is meant to point you in the right direction
on how you can contribute and what the SIFT team might expect for contributions
we receive.

If you'd like to chat with the developers about SIFT programming and design
questions you can do so on our `Gitter Chat <https://gitter.im/ssec/sift>`_ or
through GitHub issues (see below).

Development Team
----------------

Development on SIFT was started by a team at the Space Science and Engineering
Center (SSEC) at the University of Wisconsin - Madison, but has since had
contributions from other organizations. EUMETSAT has made major revisions to
SIFT including those from their contractor `ask – Innovative Visualisierungslösungen GmbH <https://askvisual.de/>`_, leading up to SIFT 2.0.
Detailed information about these teams can be found below. In general,
although these two teams may have differing goals, SIFT is kept in a flexible
state that allows it to be used for forecast training, Cal/Val operations for
instruments, or many other use cases.


SSEC
^^^^

The original Program Manager for SIFT and one of the forecast trainers was
Jordan Gerth. Since his transition to working for the National Weather
Service (NWS) these responsibilities have been taken over by Scott Lindstrom.
Additionally, we've had various testers of SIFT and trainers who used it
including Kathy Strabala, Tim Schmit, William Straka III, Jessica Braun,
and Graeme Martin.
You may find comments from Jordan (@jgerth), Scott (ScottLindstrom), and the
others throughout GitHub issues.

The SSEC development team is lead by Dave Hoese (@djhoese) and Ray Garcia
(@rayg-ssec). Additional
development can be seen from Eva Schiffer, Coda Phillips, and various UW-Madison
undergraduate student programmers.

EUMETSAT
^^^^^^^^

EUMETSAT's efforts were managed by Sauli Joro (@sjoro). Development is
primarily done by Andrea Meraner (@ameraner) and Johan Strandgren
(@strandgren). Contributions from ask came from many developers including
Alexander Rettig (@arcanerr). More information on those that have contributed
to SIFT can be found in the
`AUTHORS.md <https://github.com/ssec/sift/blob/master/AUTHORS.md>`_ file in
the root of the repository.

Bug Reporting
-------------

If you'd like to file a bug report please click the "Issues" button at the top
of the GitHub page and then click the button "New issue" button. It is a good
idea to search through the existing issues to make sure the bug you've found
or the feature you are requesting hasn't already been filed.

Adding new features
-------------------

If you have an idea for a feature you'd like to implement in SIFT please create
an issue on GitHub to discuss the feature with the core developers. You can
also chat with SIFT developers on the
`Gitter chat <https://gitter.im/ssec/sift>`_. They may be able to help guide
you on how to add the feature.

.. _dev_install:

Developer Installation
----------------------

The following instructions are for software developers or users who want to
install the **unstable** version of SIFT directly from GitHub. The easiest way
to do this is to first follow the
:doc:`conda installation instructions </installation>`
by creating a new environment with the stable version of SIFT installed.

Next we will need to uninstall the conda package of SIFT:

.. code-block:: bash

   conda uninstall --force uwsift

This will force `uwsift` to be uninstalled without uninstalling its
dependencies. We can then install the unstable version of SIFT from its source
code.

From GitHub
^^^^^^^^^^^

If you don't plan on making modifications to the SIFT source code then
we can run the following command and run the `uwsift` package as usual (see
above):

.. code-block:: bash

   pip install git+https://github.com/ssec/sift.git

From Source
^^^^^^^^^^^

If you do plan on making changes to the source code and running SIFT to see
the changes, you first need to clone the git repository:

.. code-block:: bash

   git clone git@github.com:ssec/sift.git

Note the above command uses "SSH" access to GitHub which requires
`setting up SSH keys <https://help.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh>`_
on your GitHub account. You can alternatively use the
`https://github.com/ssec/sift.git` URL.

Now we can install SIFT from the source code:

.. code-block:: bash

   cd sift
   pip install -e . --no-deps

Any changes made to the source code from here on out will take effect
immediately.

Building the Documentation
--------------------------

SIFT uses the sphinx documentation tool to generate its
`documentation website <https://sift.readthedocs.io/en/latest/>`_.
The website is automatically generated from the contents of the primary
branch on GitHub. If you'd like to make changes to the documentation you can
build the website locally to test your changes.
In addition to the above Developer Installation process, you'll need to run
the following commands to install sphinx-specific dependencies:

.. code-block:: bash

    conda install -c conda-forge sphinx sphinx_rtd_theme
    pip install blockdiag sphinxcontrib-seqdiag sphinxcontrib-blockdiag

You can then generate the documentation by running:

.. code-block:: bash

   cd doc
   make html

You can then open the `build/html/index.html` file in your preferred browser
to preview the website.

Additional Satpy Readers
------------------------

Starting with SIFT 1.1 Satpy is used for reading all input data files.
Check :doc:`the reader configuration docs </configuration/readers>` to configure which Satpy
readers are made available in SIFT. If you have a data source that is not yet supported in Satpy
(available readers are listed `here <https://satpy.readthedocs.io/en/stable/index.html#id1>`__ ), you
have to write a Satpy reader in order to be able to visualize it in SIFT. Head to the Satpy
documentation for guidelines on writing a new reader `here <https://satpy.readthedocs.io/en/stable/dev_guide/custom_reader.html>`__, and consider
contributing your reader to Satpy via a GitHub Pull Request. If you don't want to publish your reader, or as an
intermediate solution, you can plugin "local" readers in SIFT, see the :doc:`external satpy documentation </configuration/external_satpy>`

Writing Tests
-------------

All bug fixes and features contributed to SIFT should have an associated test.
Writing tests for an application as complex as SIFT (multithreaded data
loading, PyQt GUI framework, OpenGL visualization, etc) can be difficult. We've
gathered some of our lessons learned in writing tests for SIFT in the
:doc:`writing_tests` documentation.

Developer Workflow
------------------

For any contributions involving changes to the git repository on GitHub a
pull request should be submitted. A pull request is an official request
by you, the contributor, to the maintainers of SIFT to merge code from
your copy of SIFT to the primary upstream version of SIFT.

1. Follow the instructions in this GitHub documentation on creating
   a "fork": https://docs.github.com/en/get-started/quickstart/fork-a-repo
2. Create a new git branch specifically for your changes. See
   `this GitHub documentation <https://docs.github.com/en/get-started/quickstart/github-flow>`_
   for more information on how to do this. Please avoid modifying code in your
   `main` (or `master`) branch as this will make syncing upstream changes more
   difficult in the future.
3. If you haven't already, clone your fork locally and switch to the branch
   for the changes you are going to make (``git checkout <branch-name>``).
4. Make commits to this branch and push them to your fork. Your fork is likely
   referred to as the ``origin`` remote so ``git push -u origin <branch-name>``
   should work. Please include unit tests for any non-documentation changes.
5. Create a pull request by following
   `these instructions <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request>`_.
6. Wait for review from SIFT maintainers. Address any requested changes by
   making more commits on your existing local branch and pushing them to your
   fork on GitHub with ``git push``.
7. Avoid making large (especially backwards incompatible) changes without first
   discussing it with the SIFT maintainers in a GitHub issue. This avoids
   duplicate or unnecessary work.
