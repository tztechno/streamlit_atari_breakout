# streamlit_atari_breakout

deploy failure 2025-03-10 and give up, bye




Error installing requirements.
Click "Manage App" and consult the terminal for more details.

If you still have questions, leave a message in our forums and we will get back to you ASAP.

[     UTC     ] Logs for breakout-9v2wzpq3.streamlit.app/

────────────────────────────────────────────────────────────────────────────────────────

[03:37:57] 🚀 Starting up repository: 'streamlit_atari_breakout', branch: 'main', main module: 'app.py'

[03:37:57] 🐙 Cloning repository...

[03:37:58] 🐙 Cloning into '/mount/src/streamlit_atari_breakout'...
Warning: Permanently added the ED25519 host key for IP address '140.82.116.4' to the list of known hosts.

[03:37:59] 🐙 Cloned repository!

[03:37:59] 🐙 Pulling code changes from Github...

[03:38:00] 📦 Processing dependencies...


──────────────────────────────────────── uv ───────────────────────────────────────────


Using uv pip install.

Using Python 3.12.9 environment at /home/adminuser/venv

Resolved 53 packages in 705ms

  × Failed to download and build `numpy==1.22.4`

  ├─▶ Build backend failed to determine requirements with `build_wheel()`

  │   (exit status: 1)


  │   [stderr]

  │   Traceback (most recent call last):

  │     File "<string>", line 8, in <module>

  │     File

  │   "/home/adminuser/.cache/uv/builds-v0/.tmpAOFkvD/lib/python3.12/site-packages/setuptools/__init__.py",

  │   line 10, in <module>

  │       import distutils.core

  │   ModuleNotFoundError: No module named 'distutils'


  ╰─▶ distutils was removed from the standard library in Python 3.12. Consider

      adding a constraint (like `numpy >1.22.4`) to avoid building a version

      of numpy that depends on distutils.

Checking if Streamlit is installed

Installing rich for an improved exception logging

Using uv pip install.

Using Python 3.12.9 environment at /home/adminuser/venv

Resolved 4 packages in 3ms

Installed 4 packages in 18ms

 + markdown-it-py==3.0.0

 + mdurl==[2025-03-09 03:38:03.905731] 0.1.2

 + pygments==2.19.1

 + rich==13.9.4


────────────────────────────────────────────────────────────────────────────────────────



──────────────────────────────────────── pip ───────────────────────────────────────────


Using standard pip install.

Collecting streamlit==1.24.0 (from -r /mount/src/streamlit_atari_breakout/requirements.txt (line 1))

  Downloading streamlit-1.24.0-py2.py3-none-any.whl.metadata (8.0 kB)

Collecting numpy==1.22.4 (from -r /mount/src/streamlit_atari_breakout/requirements.txt (line 2))

  Downloading numpy-1.22.4.zip (11.5 MB)

     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11.5/11.5 MB 171.6 MB/s eta 0:00:00[2025-03-09 03:38:06.340337] 

  Installing build dependencies: started

  Installing build dependencies: finished with status 'done'

  Getting requirements to build wheel: started

  Getting requirements to build wheel: finished with status 'done'

ERROR: Exception:

Traceback (most recent call last):

  File "/home/adminuser/venv/lib/python3.12/site-packages/pip/_internal/cli/base_command.py", line 180, in exc_logging_wrapper

    status = run_func(*args)

             ^^^^^^^^^^^^^^^

  File "/home/adminuser/venv/lib/python3.12/site-packages/pip/_internal/cli/req_command.py", line 245, in wrapper

    return func(self, options, args)

           ^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/home/adminuser/venv/lib/python3.12/site-packages/pip/_internal/commands/install.py", line 377, in run

    requirement_set = resolver.resolve(

                      ^^^^^^^^^^^^^^^^^

  File "/home/adminuser/venv/lib/python3.12/site-packages/pip/_internal/resolution/resolvelib/resolver.py", line 95, in resolve

    result = self._result = resolver.resolve(

                            ^^^^^^^^^^^^^^^^^

  File "/home/adminuser/venv/lib/python3.12/site-packages/pip/_vendor/resolvelib/resolvers.py", line 546, in resolve

    state = resolution.resolve(requirements, max_rounds=max_rounds)

            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/home/adminuser/venv/lib/python3.12/site-packages/pip/_vendor/resolvelib/resolvers.py", line 397, in resolve

    self._add_to_criteria(self.state.criteria, r, parent=None)

  File "/home/adminuser/venv/lib/python3.12/site-packages/pip/_vendor/resolvelib/resolvers.py", line 173, in _add_to_criteria

    if not criterion.candidates:

           ^^^^^^^^^^^^^^^^^^^^

  File "/home/adminuser/venv/lib/python3.12/site-packages/pip/_vendor/resolvelib/structs.py", line 156, in __bool__

    return bool(self._sequence)

           ^^^^^^^^^^^^^^^^^^^^

  File "/home/adminuser/venv/lib/python3.12/site-packages/pip/_internal/resolution/resolvelib/found_candidates.py", line 155, in __bool__

    return any(self)

           ^^^^^^^^^

  File "/home/adminuser/venv/lib/python3.12/site-packages/pip/_internal/resolution/resolvelib/found_candidates.py", line 143, in <genexpr>

    return (c for c in iterator if id(c) not in self._incompatible_ids)

                       ^^^^^^^^

  File "/home/adminuser/venv/lib/python3.12/site-packages/pip/_internal/resolution/resolvelib/found_candidates.py", line 47, in _iter_built

    candidate = func()

                ^^^^^^

  File "/home/adminuser/venv/lib/python3.12/site-packages/pip/_internal/resolution/resolvelib/factory.py", line 182, in _make_candidate_from_link

    base: Optional[BaseCandidate] = self._make_base_candidate_from_link(

                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/home/adminuser/venv/lib/python3.12/site-packages/pip/_internal/resolution/resolvelib/factory.py", line 228, in _make_base_candidate_from_link

    self._link_candidate_cache[link] = LinkCandidate(

                                       ^^^^^^^^^^^^^^

  File "/home/adminuser/venv/lib/python3.12/site-packages/pip/_internal/resolution/resolvelib/candidates.py", line 290, in __init__

    super().__init__(

  File "/home/adminuser/venv/lib/python3.12/site-packages/pip/_internal/resolution/resolvelib/candidates.py", line 156, in __init__

    self.dist = self._prepare()

                ^^^^^^^^^^^^^^^

  File "/home/adminuser/venv/lib/python3.12/site-packages/pip/_internal/resolution/resolvelib/candidates.py", line 222, in _prepare

    dist = self._prepare_distribution()

           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/home/adminuser/venv/lib/python3.12/site-packages/pip/_internal/resolution/resolvelib/candidates.py", line 301, in _prepare_distribution

    return preparer.prepare_linked_requirement(self._ireq, parallel_builds=True)

           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/home/adminuser/venv/lib/python3.12/site-packages/pip/_internal/operations/prepare.py", line 525, in prepare_linked_requirement

    return self._prepare_linked_requirement(req, parallel_builds)

           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/home/adminuser/venv/lib/python3.12/site-packages/pip/_internal/operations/prepare.py", line 640, in _prepare_linked_requirement

    dist = _get_prepared_distribution(

           ^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/home/adminuser/venv/lib/python3.12/site-packages/pip/_internal/operations/prepare.py", line 71, in _get_prepared_distribution

    abstract_dist.prepare_distribution_metadata(

  File "/home/adminuser/venv/lib/python3.12/site-packages/pip/_internal/distributions/sdist.py", line 54, in prepare_distribution_metadata

    self._install_build_reqs(finder)

  File "/home/adminuser/venv/lib/python3.12/site-packages/pip/_internal/distributions/sdist.py", line 124, in _install_build_reqs

    build_reqs = self._get_build_requires_wheel()

                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/home/adminuser/venv/lib/python3.12/site-packages/pip/_internal/distributions/sdist.py", line 101, in _get_build_requires_wheel

    return backend.get_requires_for_build_wheel()

           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/home/adminuser/venv/lib/python3.12/site-packages/pip/_internal/utils/misc.py", line 745, in get_requires_for_build_wheel

    return super().get_requires_for_build_wheel(config_settings=cs)

           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/home/adminuser/venv/lib/python3.12/site-packages/pip/_vendor/pyproject_hooks/_impl.py", line 166, in get_requires_for_build_wheel

    return self._call_hook('get_requires_for_build_wheel', {

           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/home/adminuser/venv/lib/python3.12/site-packages/pip/_vendor/pyproject_hooks/_impl.py", line 321, in _call_hook

    raise BackendUnavailable(data.get('traceback', ''))

pip._vendor.pyproject_hooks._impl.BackendUnavailable: Traceback (most recent call last):

  File "/home/adminuser/venv/lib/python3.12/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 77, in _build_backend

    obj = import_module(mod_path)

          ^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.12/importlib/__init__.py", line 90, in import_module

    return _bootstrap._gcd_import(name[level:], package, level)

           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "<frozen importlib._bootstrap>", line 1387, in _gcd_import

  File "<frozen importlib._bootstrap>", line 1360, in _find_and_load

  File "<frozen importlib._bootstrap>", line 1310, in _find_and_load_unlocked

  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed

  File "<frozen importlib._bootstrap>", line 1387, in _gcd_import

  File "<frozen importlib._bootstrap>", line 1360, in _find_and_load

  File "<frozen importlib._bootstrap>", line 1331, in _find_and_load_unlocked

  File "<frozen importlib._bootstrap>", line 935, in _load_unlocked

  File "<frozen importlib._bootstrap_external>", line 999, in exec_module

  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed

  File "/tmp/pip-build-env-yqqsfe8_/overlay/lib/python3.12/site-packages/setuptools/__init__.py", line 10, in <module>

    import distutils.core

ModuleNotFoundError: No module named 'distutils'



[notice] A new release of pip is available: 24.0 -> 25.0.1

[notice] To update, run: pip install --upgrade pip

Checking if Streamlit is installed

Installing rich for an improved exception logging

Using standard pip install.

Collecting rich>=10.14.0

  Downloading rich-13.9.4-py3-none-any.whl.metadata (18 kB)

Collecting markdown-it-py>=2.2.0 (from rich>=10.14.0)

  Downloading markdown_it_py-3.0.0-py3-none-any.whl.metadata (6.9 kB)

Collecting pygments<3.0.0,>=2.13.0 (from rich>=10.14.0)

  Downloading pygments-2.19.1-py3-none-any.whl.metadata (2.5 kB)

Collecting mdurl~=0.1 (from markdown-it-py>=2.2.0->rich>=10.14.0)

  Downloading mdurl-0.1.2-py3-none-any.whl.metadata (1.6 kB)

Downloading rich-13.9.4-py3-none-any.whl (242 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 242.4/242.4 kB 11.9 MB/s eta 0:00:00[2025-03-09 03:38:12.464743] 

Downloading markdown_it_py-3.0.0-py3-none-any.whl (87 kB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 87.5/87.5 kB 110.6 MB/s eta 0:00:00[2025-03-09 03:38:12.480966] 

Downloading pygments-2.19.1-py3-none-any.whl (1.2 MB)

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 66.8 MB/s eta 0:00:00[2025-03-09 03:38:12.514469] 

Downloading mdurl-0.1.2-py3-none-any.whl (10.0 kB)

Installing collected packages: pygments, mdurl, markdown-it-py, rich

  Attempting uninstall: pygments

    Found existing installation: Pygments 2.19.1

    Uninstalling Pygments-2.19.1:

      Successfully uninstalled Pygments-2.19.1

  Attempting uninstall: mdurl

    Found existing installation: mdurl 0.1.2

    Uninstalling mdurl-0.1.2:

      Successfully uninstalled mdurl-0.1.2

  Attempting uninstall: markdown-it-py

    Found existing installation: markdown-it-py 3.0.0

    Uninstalling markdown-it-py-3.0.0:

      Successfully uninstalled markdown-it-py-3.0.0

  Attempting uninstall: rich

    Found existing installation: rich 13.9.4

    Uninstalling rich-13.9.4:

      Successfully uninstalled rich-13.9.4

Successfully installed markdown-it-py-3.0.0 mdurl-0.1.2 pygments-2.19.1 rich-13.9.4


[notice] A new release of pip is available: 24.0 -> 25.0.1

[notice] To update, run: pip install --upgrade pip


────────────────────────────────────────────────────────────────────────────────────────


[03:38:15] ❗️ installer returned a non-zero exit code

[03:38:15] ❗️ Error during processing dependencies! Please fix the error and push an update, or try restarting the app.

[03:39:47] ❗️ Streamlit server consistently failed status checks

[03:39:47] ❗️ Please fix the errors, push an update to the git repo, or reboot the app.

main
tztechno/streamlit_atari_breakout/main/app.py




