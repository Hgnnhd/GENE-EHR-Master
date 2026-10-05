"""Data stage · Cancer-registry coverage per participant, from the recruitment assessment centre.

UK Biobank cancer registrations come from the national registries of England, Scotland and
Wales, each with its own coverage period for a given data release. The recruitment
(Instance 0) assessment centre places a participant in one of the three countries; the
country's verified [registry_start, registry_end_exclusive) applies to that participant.

The dates are NOT inferred from the data (last diagnosis, HES cutoff, ...): they are typed into
configs/registry_coverage.json from the UK Biobank documentation for the local data release,
together with source_version and evidence, and only used once status is set to "verified".
Writes <output>/registry_coverage.csv, read by the participants stage.
"""
import json

import pandas as pd

from .common import ID, ROOT, Context, fmt, log, read
from .definitions import dates

STAGE, TITLE = "coverage", "Cancer-registry coverage by recruitment country"
COUNTRIES = ("England", "Scotland", "Wales")
# UK Biobank assessment centres (data-coding 10), matched on the first word of the centre name.
CENTRES = {
    "barts": "England", "birmingham": "England", "bristol": "England", "bury": "England", "cheadle": "England",
    "croydon": "England", "hounslow": "England", "leeds": "England", "liverpool": "England",
    "manchester": "England", "middlesborough": "England", "middlesbrough": "England", "newcastle": "England",
    "nottingham": "England", "oxford": "England", "reading": "England", "sheffield": "England",
    "stockport": "England", "stoke": "England",
    "edinburgh": "Scotland", "glasgow": "Scotland",
    "cardiff": "Wales", "swansea": "Wales", "wrexham": "Wales",
}
CODINGS = "app176660_20240512000635.dataset.codings.csv"


def manifest(ctx):
    return json.loads((ROOT / ctx.config["coverage_manifest"]).read_text(encoding="utf-8"))


def centre_column(frame):
    cols = [c for c in frame.columns if c != ID]
    for test in (lambda c: "Instance 0" in c, lambda c: "centre" in c.lower(), lambda c: True):
        hits = [c for c in cols if test(c)]
        if hits:
            return hits[0]
    raise ValueError("assessment centre file has no data column")


def countries(ctx, centre_file):
    """Recruitment country per participant (index participant_id)."""
    path = ctx.source(ctx.hosp, centre_file)
    raw = read(path).set_index(ID)
    col = centre_column(raw)
    values = raw[col].astype("string").str.strip()
    if values.dropna().str.fullmatch(r"\d+").all():  # numeric centre codes -> names via data-coding 10
        codings_path = ctx.hosp / CODINGS
        if not codings_path.exists():
            raise FileNotFoundError(f"{path.name} holds numeric centre codes; {codings_path} is needed to name them")
        coding = read(ctx.source(ctx.hosp, CODINGS))
        coding = coding.loc[coding.coding_name.eq("data_coding_10")].set_index("code").meaning
        values = values.map(coding).astype("string")
    first = values.str.lower().str.extract(r"^\s*([a-z]+)", expand=False)
    country = first.map(CENTRES)
    unknown = sorted(values[values.notna() & country.isna()].unique())
    if unknown:
        raise ValueError(f"unrecognised assessment centres in {path.name} ({col}): {unknown}; add them to CENTRES")
    log(f"{path.name}: column '{col}', {fmt(values.notna().sum())} participants with a recruitment centre")
    return pd.DataFrame({"centre": values, "region": country})


def run(args):
    ctx = Context(args, STAGE, TITLE)
    cov = manifest(ctx)
    out = ctx.out / "registry_coverage.csv"
    centre_file = cov.get("assessment_centre_file")
    places = None
    if centre_file and (ctx.hosp / centre_file).exists():
        places = countries(ctx, centre_file)
        counts = places.region.value_counts()
        ctx.qc["participants_by_country"] = {k: int(v) for k, v in counts.items()}
        ctx.qc["participants_by_centre"] = {k: int(v) for k, v in places.centre.value_counts().items()}
    if cov.get("status") != "verified":
        out.unlink(missing_ok=True)
        ctx.qc["status"] = "unverified"
        lines = ["status 'unverified' in configs/registry_coverage.json: coverage not applied, 5-year labels stay empty",
                 "to verify: fill regions.<country>.registry_start / registry_end_exclusive, source_version and evidence "
                 "from the UK Biobank documentation for this data release, then set status to 'verified'"]
        if places is not None:
            lines.insert(0, "recruitment country: " + ", ".join(f"{k} {fmt(v)}" for k, v in counts.items()))
        else:
            lines.insert(0, f"assessment centre file '{centre_file}' not found in {ctx.hosp}")
        return ctx.done(*lines)
    if not cov.get("source_version") or not cov.get("evidence"):
        raise ValueError("verified registry coverage requires source_version and evidence in configs/registry_coverage.json")
    if cov.get("coverage_file"):
        out.unlink(missing_ok=True)
        ctx.qc["status"] = "verified (participant-level file supplied)"
        return ctx.done(f"using the supplied participant-level file {cov['coverage_file']}")
    if places is None:
        raise FileNotFoundError(f"verified regional coverage needs the assessment centre file '{centre_file}' in {ctx.hosp}")
    regions = cov.get("regions") or {}
    table = {}
    for country in COUNTRIES:
        r = regions.get(country) or {}
        start, end = dates(pd.Series([r.get("registry_start")])).iloc[0], dates(pd.Series([r.get("registry_end_exclusive")])).iloc[0]
        if country in counts and (pd.isna(start) or pd.isna(end) or end <= start):
            raise ValueError(f"regions.{country} needs registry_start < registry_end_exclusive "
                             f"({fmt(counts[country])} participants recruited there)")
        table[country] = (start, end)
    frame = places.reset_index()
    frame["registry_start"] = frame.region.map(lambda c: table.get(c, (pd.NaT, pd.NaT))[0])
    frame["registry_end_exclusive"] = frame.region.map(lambda c: table.get(c, (pd.NaT, pd.NaT))[1])
    frame[[ID, "region", "centre", "registry_start", "registry_end_exclusive"]].to_csv(out, index=False)
    ctx.qc["status"] = "verified"
    ctx.qc["regions"] = {c: [str(s.date()), str(e.date())] for c, (s, e) in table.items() if pd.notna(s)}
    ctx.done(*[f"{c}: [{s.date()}, {e.date()}) for {fmt(counts.get(c, 0))} participants" for c, (s, e) in table.items() if pd.notna(s)],
             f"source: {cov['source_version']}; evidence: {cov['evidence']}",
             f"participants without a recruitment centre (no coverage): {fmt(frame.region.isna().sum())}")
