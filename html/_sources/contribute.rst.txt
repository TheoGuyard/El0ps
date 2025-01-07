.. _contribute:

==========
Contribute
==========

``el0ps`` is still in its early stages of development.
Any feedback or contribution is welcome.

**Bug report**
    Altough unit tests are in place, there might be some bugs.
    Please report any issues you encounter in the `issue page <https://github.com/TheoGuyard/El0ps/issues>`_. 
    You can also suggest new features or improvements in this page.

**Pull request**
    New contribution are made through the `pull requests page <https://github.com/TheoGuyard/El0ps/pulls>`_.
    Please make sure that your code is `PEP8 compliant <https://peps.python.org/pep-0008/>`_ and that you have added unit tests for your new features.
    You can proceed as explained below.

1. `Fork <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo>`_ the repository on your GitHub account.

2. Clone ``el0ps`` on your local machine using the command

.. prompt:: shell $

    git clone https://github.com/{YOUR_GITHUB_USERNAME}/El0ps

3. Install the package locally and the development dependencies using the command

.. prompt:: shell $

    cd El0ps
    pip install -e .[dev]

4. Make a new branch for your feature using the command

.. prompt:: shell $

    git branch my_new_feature
    git switch my_new_feature

5. Make your changes and add unit tests for them.

6. Open a `pull request <https://github.com/TheoGuyard/El0ps/pulls>`_ on ``el0ps`` repository.

You can build the documentation locally using the command

.. prompt:: shell $

    sphinx-build -M html doc/source/ doc/build/

from the root of the repository. The generated documentation will be in the ``doc/build/`` directory.