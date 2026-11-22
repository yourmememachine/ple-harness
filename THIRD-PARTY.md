# Third-party notices

Third-party software used at runtime by this harness, in the form:
name, version, licence. This list is updated whenever a dependency
is added or changed. Development-only tools are not listed.

| Name | Version | Licence |
| --- | --- | --- |
| websockets | 17.4 | BSD-3-Clause |

## websockets

- **Name:** websockets
- **Version:** 17.4 (pinned in `requirements.txt`)
- **Licence:** BSD-3-Clause
- **Use:** WebSocket server transport in `serve.py`. This is the
  only third-party runtime dependency; everything else in the
  harness is Python standard library. The full licence text ships
  with the package (the `LICENSE` file in the websockets
  distribution).
