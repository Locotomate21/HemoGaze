# examples/

Clients that consume `server/`. Each is deliberately minimal and depends on
nothing from `training/` -- a consumer of the API should not need the deep
learning stack.

| file | what it is |
|---|---|
| `cli_client.py` | terminal client. Needs only `requests`. Exit codes: 0 estimate returned, 2 service unreachable or without weights, 1 bad request -- so a pipeline can tell "down" from "answered". |
| `web_client.html` | single file, no build step. Open it or serve it and point it at the API. |

    # with the server running on :8000
    python examples/cli_client.py ../../Datasets/*_cpanemic_*/images/Image_001.png --cutoff 11

A desktop client would go here too; the pattern is the same HTTP call.

## One thing both clients do on purpose

**The warning is rendered above the number, and re-read from the server response
rather than hard-coded.** An interface that shows a confident figure first and the
caveat underneath trains people to read the figure and skip the caveat. The web
client overwrites its own static warning text with whatever `/predict` returned,
so the caveat cannot drift out of sync with the model actually deployed.

They also surface `roi_fraction` and complain when it is under 5%: that means the
upload is mostly background, and the estimate is then meaningless regardless of
what the model says.
