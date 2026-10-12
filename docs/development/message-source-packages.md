# Build a message package from local inputs

The generic generator and source-package hooks do not select a DimOS catalog.
They read explicit project inputs and installed message dependencies. A package
build never downloads schemas or invokes a native compiler.

Create `interfaces/sample_msgs/msg/Reading.msg`:

```text
int32 sequence
float64 value
```

Configure the package once in `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=70", "wheel", "dimos-message-build==0.1.0"]
build-backend = "setuptools.build_meta"

[project]
name = "sample-messages"
version = "0.1.0"
dependencies = ["dimos-message-build==0.1.0", "rosbags==0.11.0", "numpy>=1.26.4"]

[tool.dimos.messages]
source = "interfaces"
```

The module name defaults to `sample_messages`. Add this `setup.py`:

```python
from pathlib import Path
from setuptools import setup
from dimos_message_build.source_package import setup_arguments

setup(**setup_arguments(Path(__file__).resolve().parent))
```

Include the build entrypoint in `MANIFEST.in`:

```text
include pyproject.toml setup.py
prune build
```

Use matching proposal backend wheels. They are not assumed to exist on PyPI.
For an offline build, prepare those wheels and all Python dependencies first:

```sh
pip install --no-index --find-links review/wheels .
pip install --no-index --find-links review/wheels -e .
```

The same hooks generate local package sources in `build/message-source` for
wheel, source and editable builds. They emit native source projects but do not
invoke ROS, CMake or Cargo. A source distribution copies declared inputs and
resources into the archive. A wheel contains generated values and their complete
schema closure. Import does not build files or access the network.

```python
from sample_messages.sample_msgs.msg import Reading
from dimos_message_build.registry import encode, decode

value = Reading(sequence=17, value=21.5)
assert decode(encode(value), Reading).value == 21.5
```

After editing a source, repeat the editable install and restart Python processes.
Schemas and type identities are frozen within each process.

## Select roots and dependencies

Without `roots`, the project owns all messages declared under its source folder.
Use `roots = ["sample_msgs/msg/Reading"]` to select a dependency closure explicitly.
Alternatively, `catalog = "catalog.json"` reads a format-1 JSON manifest whose
`roots` maps type names to support reasons. Do not specify both.

`resources = ["pins.json"]` carries declared metadata into both source archives
and generated package resources. Input paths must remain inside the project.
The generic hooks have no repository-path or built-in package-name convention.

For another installed message package, declare an exact module/version under
`[tool.dimos.messages.dependencies]`. Add that distribution to both build-system
and project dependencies. The generator imports its owned Python/C++/Rust types,
checks schema/version/ABI identity and retains the full schema closure. It does
not generate another owner for the same type.

## Prepare upstream data explicitly

`dimos_message_build.source_inputs.prepare_sources(lock, output, cache,
offline=True)` reads a source lock. Each source specifies its URL, archive SHA256,
archive prefix and selected files with their source paths and SHA256 values.
This preparation API is separate from source-package hooks. It needs the explicit
`dimos-message-build[prepare]` dependency when used.

Online preparation downloads only declared archives. Offline preparation requires
cached archives. Both paths check archive and selected-file hashes before replacing
the prepared input directory. Unsafe paths and non-regular files fail. Product
layers own their upstream revisions, type selection, custom messages and thin
preparation/release entrypoints. No package manager or core/extras scheme is added.
