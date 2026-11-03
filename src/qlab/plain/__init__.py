"""The plain-Russian top of an idea's page (docs/PLAIN_PAGE.md).

`plain_page` builds what the page shows; `spec_problems` is the check the researcher and the
generator run on a spec they wrote; the texts live as data in `strategies/plain/`.
"""

from qlab.plain.check import spec_problems
from qlab.plain.page import plain_page

__all__ = ["plain_page", "spec_problems"]
