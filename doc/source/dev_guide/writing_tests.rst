Writing Tests
=============

SIFT uses `pytest <http://pytest.org/en/latest/>`_ as its testing framework and
tests are automatically run in GitHub Actions. All tests should go under
``uwsift/tests`` and should be named ``test_<name_of_file_under_test>.py``.
Any multi-file pytest fixtures should be placed in ``conftest.py``.

The below sections provide guidelines and gotchas for writing tests for SIFT.
If you have any remaining questions you can create a GitHub issue or talk to
the developers on `Gitter <https://gitter.im/ssec/sift>`_.

Testing the GUI
---------------

To test interactions with the SIFT GUI, you can use
`pytest-qt <https://pytest-qt.readthedocs.io/en/latest/intro.html>`_, which is
a pytest plugin that provides a fixture that makes it easy to interact with a
dialog window. Examples of using this fixture can be found in the pytest-qt
documentation or in
`SIFT <https://github.com/ssec/sift/blob/master/uwsift/tests/view/test_export_image.py#L223>`_.

A couple of things to note:

* The window that you're testing must be in focus for qtbot to act on it. You
  can ensure the window is in focus by calling ``window.raise_()`` and
  ``window.activateWindow()`` (an example of this found
  `here <https://github.com/ssec/sift/blob/master/uwsift/tests/conftest.py#L14>`_).
* Do not create an instance of the main SIFT window. There is a fixture of the
  main window provided in ``conftest.py`` that you should use instead.

Things to note
--------------

* Mocking in the pytest framework can be done with
  `monkeypatching <https://docs.pytest.org/en/latest/monkeypatch.html>`_,
  but `pytest-mock <https://github.com/pytest-dev/pytest-mock>`_
  (pytest plugin that wraps mock) can be used if more robust mock
  functionality is needed.
* Coverage for functions decorated with ``@jit`` does not work, so in order
  to get an accurate coverage value, you should run the test once with
  ``@jit`` and then once without ``@jit`` by calling ``my_function.pyfunc``
  instead.

  * An example of this is found in
    `test_tile_calculator.py <https://github.com/ssec/sift/blob/master/uwsift/tests/view/test_tile_calculator.py#L21>`_.
    There is a fixture that runs tests once with functions with ``@jit``
    enabled and once again with the original python functions

* If you're running the tests on your local machine, do not click anywhere
  else while the tests are running. Some tests require that the SIFT GUI is
  in focus, otherwise they'll fail.

Running the tests
-----------------

All tests can be run using the command ``pytest path/to/uwsift``. A specific
test directory or file can be run using ``pytest path/to/test``. The packages
required to run the tests can be installed from conda with:

.. code-block:: bash

   conda install pytest pytest-qt pytest-mock
