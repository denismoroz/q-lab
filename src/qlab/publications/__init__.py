"""The publication feed: new papers as a source of ideas (docs/PLAN.md stage 6).

Everything here is code and spends no tokens, except the one hand-off to the
scout in `qlab.publications.stage`, which goes through the budget guard:

- `fetch`   -- arXiv and Crossref (SSRN) clients; they return `Item`s;
- `store`   -- the `publication` table: dedupe, the new tail, hand-off state;
- `filter`  -- the versioned topic rule (night/publication_filter.yaml);
- `stage`   -- the night's two steps: fetch + filter, then the scout's batch.
"""

from qlab.publications.items import Item

__all__ = ["Item"]
