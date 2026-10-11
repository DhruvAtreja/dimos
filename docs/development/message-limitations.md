# Native CDR proposal: known limits

This proposal uses these libraries:

- Python: rosbags 0.11.0.
- C++: upstream ROSIDL/FastRTPS and Fast CDR 2.4.0.
- Rust: ros2msg 0.5.3 and re_cdr 0.1.0.

On 2026-10-09, the user agreed to continue the proposal with the limits below. This
agreement does not show that the decoder handles invalid input safely. The proposal does
not patch these libraries or add a field codec to hide their errors.

Use CDR from trusted producers and recordings. The schema must match the data. **Do not
use the Python decoder to validate hostile or invalid input.** A trusted producer can
also have errors. Round-trip tests and transport authentication do not prove parser
safety.

The small tests below did not show a denial of service or damage to valid data. They did
show that invalid data can decode to an incorrect value that looks valid.

## The original 13 strict acceptance failures

The complete CDR stack contains the tests in
`dimos/message_codegen/test_python_source.py`. This file is added by the message-package
layer; it is not part of the first generator PR. Each known failure
has its own identifier and `xfail(strict=True)` marker. If that test passes, the check
fails. Review the library version and the recorded limit.

For invalid wire data, the marker permits only an assertion that an expected exception
was not raised. For unsupported schemas, it permits only `NotImplementedError`. Other
exceptions fail the test.

From the complete stack checkout, run the tests without these markers:

```sh expected-error
python -m pytest --noconftest -o addopts='' --runxfail \
  dimos/message_codegen/test_python_source.py
```

With the pinned version, this command reports **8 passed and 13 failed**. The normal run
reports **8 passed, 13 xfailed and 0 failed**. Report both results. Do not report that
the decoder passes all tests.

### CDR-L01

**Unused bytes: 3 cases.** A valid `Bool` followed by one, two or three zero bytes is
accepted, and those bytes disappear on re-encoding. The original rule required the
decoder to consume the entire payload. Rosbags permits up to three trailing bytes;
framing and padding policies vary, so these tests do not prove every padded sample is
invalid CDR. The decoded boolean is unchanged. Four extra bytes are still rejected in
the ordinary, non-optimized test run.

### CDR-L02

**Invalid string data: 2 cases, little and big endian.** A declared string length of
zero, with no terminator, becomes an empty string. A normal CDR string length includes
the NUL terminator. Re-encoding creates length one plus NUL. Some native libraries also
tolerate a zero-length null-string form; this tolerance does not satisfy the original
assertion.

### CDR-L03

**Invalid string data: 2 cases, little and big endian.** Length two with payload `ab`
and no NUL decodes as `a`. The decoder discards the final byte without checking that it
is a terminator. This test shows that the decoder truncates invalid input. It does not
show damage to a valid encoded string.

### CDR-L04

**Invalid string data: 2 cases, little and big endian.** Declared length `0xffffffff`
with only one NUL byte becomes an empty string. The upstream decoder reads that length
as signed `-1`; its returned body position is three although five bytes were supplied.
This test shows that the decoder accepts invalid, incomplete input. It does not show a
four-gigabyte allocation or a denial of service.

### CDR-L05

**Representation mismatch: 1 case.** Header `00 03 00 00` followed by byte `01` is
accepted as `Bool(True)`. Identifier `0003` means parameter-list CDR little endian. This
body is not a valid parameter list. The proposal expects plain CDR (`0000` or `0001`).
Rosbags interprets it as plain little-endian CDR instead of rejecting the
representation.

### CDR-L06

**Invalid boolean data: 1 original case.** Boolean byte `02` becomes `True`, and
re-encoding changes it to `01`. The decoder changes the invalid value without an error.
The separate existing array regression in `test_definitions.py` exercises the same
limitation in both byte orders: **2 additional explicit xfails**, not two additional
categories. Its valid padding and NaN checks remain ordinary tests.

### CDR-L07

**Bounded-schema support: 1 case.** A valid `shape_msgs/msg/SolidPrimitive` with three
dimensions cannot encode/decode through the Python registry: it raises
`NotImplementedError`. The type can be constructed, but valid bounded messages
**cannot** be encoded or decoded through this registry.

### CDR-L08

**Bounded-schema error contract: 1 case.** The same type with four dimensions is also
rejected as unsupported, rather than with the assertion's expected bound validation
`ValueError`. The decoder rejects the value. This test does not show that it accepts a
value above the bound. It does not truncate or remove the bound.

### CDR-L09

The original bounded `demo_msgs/msg/Telemetry` generation is retained as two strict
expected failures (C++ and Rust) in `test_native_libraries.py`. Its `.msg` source is
unchanged. These tests also show that bounded strings are unsupported. They are not part
of the original 13 Python decoder tests. The built-in catalog also has two
endian-specific `CDR-L07` xfails for `SolidPrimitive`.

## Other native API and schema limits

Python encode/decode rejects a message if it or any dependency has a bounded string or
bounded sequence. C++ and Rust generation reject messages with bounded strings in their
dependency closure. Rust generation also rejects wide strings; their mapping has not
been validated. These errors are reported through the public API. The legacy
`demo_msgs/msg/Telemetry` fixture contains both a bounded string and bounded sequence.
Do not report that its full three-language/default tests or its viewer/package demos
pass. Its original schema and tests remain available for later library work.

Python/Rust construction requires explicit fields. Python native classes do not apply
`.msg` defaults; Rust's upstream floating-default generation is disabled. These are
separate from the 13 wire/bounds cases. NumPy arrays and nested values use the library's
ownership and equality rules. Helpers provide copy and read-only-view operations. Former
pybind container methods are not the native value API. Rosbags initializes Python
classes and codecs in memory from installed schemas; Python import does not compile
native code or download dependencies. Use the single frozen registry within a runtime
process. Independent rosbags reference stores replace the process-global `usertypes`
module. This can break pickle type identity. Run independent reference decoders in
separate processes. Oracle tests restore that global module after each case.

C++ distribution is source-only with explicit, on-demand native preparation. Prebuilt
SDK products are out of scope. M20 SDK access, host multicast, Foxglove sign-in and
human visual acceptance are separate limits, not covered by these expected failures. No
benchmark acceptance is implied.

## Source references and future changes

- [Rosbags type system](https://ternaris.gitlab.io/rosbags/topics/typesys.html)
- [Rosbags decoder implementation](https://gitlab.com/ternaris/rosbags/-/blob/v0.11.0/src/rosbags/typesys/store.py)
- [OMG CDR string encoding](https://www.omg.org/cgi-bin/doc?formal%2F01-09-52.pdf=)
- [OMG representation identifiers](https://www.omg.org/spec/DDSI-RTPS/2.2/PDF)

Review each expected failure before changing the upstream version, adding checks at the
API boundary or reducing the supported contract. Do not skip the suite, weaken its
assertions or add a field codec only to make the tests pass.
