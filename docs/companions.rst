Companion packages (ffpopt / scission / ligandparam)
====================================================

ALPS is an orchestrator. It does not vendor the three science packages.
Sibling checkouts (or ``pip install -e`` of each) sit beside ``alps-main``::

    alps-main/            import alps
    ligandparam-main/     import ligandparam
    scission-main/        import scission
    ffpopt-main/          import ffpopt  (York: src/python/lib/ffpopt)

How ALPS chooses a tree
-----------------------

Set these before starting the process. They are read once on ``import alps``.

=============================== ===========================================
Variable                        Meaning
=============================== ===========================================
``ALPS_LIGANDPARAM_PATH``       Independent ligandparam directory
``ALPS_FFPOPT_PATH``            Independent ffpopt directory
``ALPS_SCISSION_PATH``          Independent scission directory
=============================== ===========================================

A PATH value is either the parent of the package (``.../lib`` with
``lib/ffpopt/``) or the package directory itself (``scission-main/`` with
``__init__.py``). Unset PATH falls back to sibling ``*-main`` folders, then
to whatever is already installed.

Call graph
----------

- ``lig-getparam`` -> ligandparam recipes
- ``lig-dihed-correct`` -> ``alps.workflows`` -> scission fragment/merge +
  ``ffpopt.Workflows.run_dihed_twist_workflow``
- ``lig-scission`` -> ``scission.Cli.main``

Hard rules: scission must not import ffpopt/ligandparam/alps. ffpopt must
not import ligandparam/alps. ALPS is the only package that imports both
scission and ffpopt.
