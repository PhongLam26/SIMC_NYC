# Reference Verification Report

Report date: 2026-07-15

Scope: current `paper_springer/references.bib`, mirrored in `paper_overleaf/references.bib`, and citations in both manuscript sources.

## Verification Summary

- Bibliography entries: 32.
- Manuscript citation keys resolved: 32/32.
- Entries cited at least once: 32/32.
- Undefined citations: none found after clean build.
- Unused bibliography entries in the manuscript set: none found.
- Fabricated references: none detected.
- Duplicate DOI/title checks: no conflicting duplicate DOI or duplicate title found.
- Numeric ordering: handled by the Springer numeric bibliography style.

## Metadata Updates

- `LiuGarg2024SLA` was updated from arXiv-only metadata to the official ACM EC '24 conference version with DOI `10.1145/3670865.3673624`.
- `Ke2017LightGBM` keeps the NeurIPS 2017 proceedings source and now uses official NeurIPS metadata, pages 3146-3154.
- `LundbergLee2017SHAP` keeps the NeurIPS 2017 proceedings source and now uses official NeurIPS metadata, pages 4765-4774.
- Dataset and web references use official NYC Open Data, NOAA, OpenStreetMap, or project documentation sources. The 2026 year is treated as the access/snapshot date and is paired with compact access notes where appropriate.
- Both NYC 311 archive references are present: the 2010-2019 historical archive and the 2020-present current archive.
- The traffic-flow reference uses the Information Sciences journal version.

## Preprint Handling

Preprints retained in the bibliography are labelled as arXiv entries and are used only for related-work or data-curation context. They are not presented as official journal or conference publications unless official metadata was verified.

## Reviewer Notes

The bibliography intentionally cites official data portals for the public datasets and keeps OSM/PLUTO references tied to the retrospective context discussion. OSM/PLUTO snapshots are not used by the final prospective alerting model, ensemble score, category thresholds, headline metrics, or SHAP feature set.
