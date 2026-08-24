"""Client for the ESIOS API (Red Eléctrica de España).

Used once to build the frozen snapshot and then not modified — so the
validation lives here, before the Parquet is written, rather than in whatever
reads it later. A silently short month is the failure this file exists to
prevent: the API answers 200 with fewer rows than you asked for and nothing
downstream can tell that from a quiet market.

Three choices worth stating
---------------------------

**Granularity is not requested, it is observed.** The obvious call is to pass
``time_trunc=hour``, but that asks the server to *aggregate*, which would
silently paper over a series published at a granularity we did not expect —
exactly the thing worth knowing before freezing. So requests carry no
``time_trunc`` by default: whatever ESIOS publishes is what arrives, the grid
is then checked against :func:`bess_arb.timeline.validate_index`, and a
mismatch is an error rather than a resample. The observed granularity goes
into the manifest.

**Ranges are asked for in market-local time.** ESIOS is asked for
``2024-01-01T00:00:00+01:00`` to ``2024-02-01T00:00:00+01:00``, which is
January as Spain defines it. Asking in UTC clips an hour off each end, and
asking with a naive string lets the server choose a zone. Timestamps come
back and are stored in UTC (:func:`bess_arb.timeline.validate_index` refuses
anything else).

**Chunked by month, with a floor on the request interval.** Rate limits for
long historical pulls are documented nowhere the author could find, and the
polite failure mode is a slower pull rather than a 429 halfway through four
years. One request per second, retried with backoff on 429 and 5xx.
"""

from __future__ import annotations

import datetime as dt
import os
import time
import unicodedata
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from bess_arb.timeline import INDEX_NAME, DayLike, day_bounds

__all__ = [
    "DEFAULT_BASE_URL",
    "TOKEN_ENV_VAR",
    "EsiosClient",
    "EsiosError",
    "Indicator",
    "read_token",
]

DEFAULT_BASE_URL = "https://api.esios.ree.es"
TOKEN_ENV_VAR = "ESIOS_TOKEN"

# Retried statuses. 429 is the documented-nowhere rate limit; the 5xx family
# is ESIOS being ESIOS on a long pull.
_RETRY_STATUSES = (429, 500, 502, 503, 504)


class EsiosError(RuntimeError):
    """A request failed, or succeeded and returned something unusable."""


@dataclass(frozen=True, slots=True)
class Indicator:
    """One ESIOS series, as the catalogue describes it."""

    id: int
    name: str
    short_name: str = ""

    def __str__(self) -> str:
        return f"{self.id} — {self.name}"


def read_token(env_var: str = TOKEN_ENV_VAR) -> str:
    """The personal ESIOS token, from the environment or a file beside it.

    Two accepted forms: ``ESIOS_TOKEN`` holding the token itself, or
    ``ESIOS_TOKEN_FILE`` holding a path to a file containing it. The file
    form is the safer habit — a token in an environment variable ends up in
    shell history and in the environment of every child process.

    Never logged, never written to the manifest, and ``.gitignore`` already
    excludes ``.env``, ``*.token`` and ``esios_token*``.
    """
    path_var = f"{env_var}_FILE"
    token_path = os.environ.get(path_var)
    if token_path:
        try:
            token = Path(token_path).read_text(encoding="utf-8").strip()
        except OSError as error:
            raise EsiosError(f"cannot read {path_var}={token_path}: {error}") from error
        if not token:
            raise EsiosError(f"the file at {path_var}={token_path} is empty")
        return token

    token = os.environ.get(env_var, "").strip()
    if not token:
        raise EsiosError(
            f"no ESIOS token found. Set {env_var} to the token, or "
            f"{path_var} to a file containing it. A token is requested by "
            "email from consultasios@ree.es."
        )
    return token


def _fold(text: str) -> str:
    """Casefold and strip accents, so 'electrica' matches 'eléctrica'."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def _month_chunks(
    first_day: dt.date, last_day: dt.date
) -> Iterator[tuple[dt.date, dt.date]]:
    """Split a range into whole calendar months, both ends inclusive."""
    cursor = first_day
    while cursor <= last_day:
        if cursor.month == 12:
            next_month = dt.date(cursor.year + 1, 1, 1)
        else:
            next_month = dt.date(cursor.year, cursor.month + 1, 1)
        chunk_end = min(next_month - dt.timedelta(days=1), last_day)
        yield cursor, chunk_end
        cursor = chunk_end + dt.timedelta(days=1)


class EsiosClient:
    """A thin, polite, validating wrapper over the ESIOS REST API."""

    def __init__(
        self,
        token: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = 60.0,
        min_interval_s: float = 1.0,
        max_retries: int = 5,
        session: requests.Session | None = None,
    ) -> None:
        if not token:
            token = read_token()
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._min_interval_s = min_interval_s
        self._last_request_at = 0.0

        self._session = session if session is not None else requests.Session()
        # Both auth forms are sent. ESIOS moved from the `Authorization:
        # Token token="..."` scheme to `x-api-key`; which one a given token
        # works with is not worth a round of trial and error, and an unknown
        # header is ignored rather than rejected.
        self._session.headers.update(
            {
                "x-api-key": token,
                "Authorization": f'Token token="{token}"',
                "Accept": "application/json; application/vnd.esios-api-v1+json",
                "Content-Type": "application/json",
            }
        )
        retry = Retry(
            total=max_retries,
            backoff_factor=1.0,
            status_forcelist=_RETRY_STATUSES,
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
            raise_on_status=False,
        )
        self._session.mount("https://", HTTPAdapter(max_retries=retry))

    # -- transport -------------------------------------------------------

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """One GET, rate-limited, with the token kept out of the error text."""
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self._min_interval_s:
            time.sleep(self._min_interval_s - elapsed)

        url = f"{self._base_url}/{path.lstrip('/')}"
        try:
            response = self._session.get(url, params=params, timeout=self._timeout_s)
        except requests.RequestException as error:
            raise EsiosError(f"GET {path} failed: {error}") from error
        finally:
            self._last_request_at = time.monotonic()

        if response.status_code != 200:
            raise EsiosError(
                f"GET {path} returned {response.status_code}: {response.text[:200]!r}"
            )
        try:
            payload = response.json()
        except ValueError as error:
            raise EsiosError(f"GET {path} returned non-JSON body") from error
        if not isinstance(payload, dict):
            raise EsiosError(
                f"GET {path} returned {type(payload).__name__}, not an object"
            )
        return payload

    # -- catalogue -------------------------------------------------------

    def list_indicators(self) -> pd.DataFrame:
        """The whole indicator catalogue: id, name, short name."""
        payload = self._get("/indicators")
        rows = payload.get("indicators")
        if not isinstance(rows, list):
            raise EsiosError("/indicators did not return a list of indicators")
        frame = pd.DataFrame(
            [
                {
                    "id": int(row["id"]),
                    "name": str(row.get("name", "")).strip(),
                    "short_name": str(row.get("short_name", "")).strip(),
                }
                for row in rows
                if isinstance(row, dict) and "id" in row
            ]
        )
        return frame.sort_values("id", ignore_index=True)

    def search_indicators(self, *terms: str) -> pd.DataFrame:
        """Catalogue rows whose name contains every term, accents ignored.

        The verification step for the four indicator IDs. Searching rather
        than trusting a remembered number is the whole point: an ID that
        looks plausible and means something else produces a snapshot that is
        wrong in a way no downstream test can see.
        """
        catalogue = self.list_indicators()
        if not terms:
            return catalogue
        haystack = (catalogue["name"] + " " + catalogue["short_name"]).map(_fold)
        mask = pd.Series(True, index=catalogue.index)
        for term in terms:
            mask &= haystack.str.contains(_fold(term), regex=False)
        return catalogue.loc[mask].reset_index(drop=True)

    def indicator_metadata(self, indicator_id: int) -> dict[str, Any]:
        """The indicator's own description, without pulling any values.

        Asked for a one-day window because ESIOS has no metadata-only
        endpoint; the values are discarded.
        """
        payload = self._get(
            f"/indicators/{indicator_id}",
            {
                "start_date": "2024-06-15T00:00:00+02:00",
                "end_date": "2024-06-16T00:00:00+02:00",
            },
        )
        indicator = payload.get("indicator")
        if not isinstance(indicator, dict):
            raise EsiosError(f"indicator {indicator_id} returned no metadata")
        values = indicator.get("values") or []
        return {
            "id": int(indicator.get("id", indicator_id)),
            "name": str(indicator.get("name", "")).strip(),
            "short_name": str(indicator.get("short_name", "")).strip(),
            "step_type": indicator.get("step_type"),
            "magnitud": indicator.get("magnitud"),
            "tiempo": indicator.get("tiempo"),
            "sample_rows": len(values),
            "sample_geos": sorted(
                {
                    (int(v["geo_id"]), str(v.get("geo_name", "")))
                    for v in values
                    if isinstance(v, dict) and v.get("geo_id") is not None
                }
            ),
        }

    # -- values ----------------------------------------------------------

    def fetch_indicator(
        self,
        indicator_id: int,
        first_day: DayLike,
        last_day: DayLike,
        *,
        geo_id: int | None = None,
        geo_ids: Sequence[int] | None = None,
        time_trunc: str | None = None,
        progress: bool = False,
    ) -> pd.DataFrame:
        """All published values for an indicator over a range of delivery days.

        Returns a frame indexed by UTC instant with ``value``, ``geo_id`` and
        ``geo_name``.

        **A multi-geography indicator must be told which geography to keep.**
        Indicator 600, the day-ahead price, carries six — Portugal, France,
        Spain, Germany, Belgium, the Netherlands — and returns them
        interleaved on the same timestamps, Portugal first. Anything that
        deduplicates on the timestamp alone therefore keeps *Portugal* and
        calls it Spain. The two are equal whenever the interconnector is
        uncongested, which is most days, so the mistake survives inspection
        and shows up only as a slightly wrong number. Hence: ``geo_id`` is
        required whenever more than one is present, and the filter is applied
        client-side even when the server was asked to do it, because a
        silently ignored query parameter is the other half of the same bug.
        """
        start, end = day_bounds(first_day, last_day)
        frames: list[pd.DataFrame] = []
        chunks = list(_month_chunks(start.date(), (end - pd.Timedelta(days=1)).date()))
        if geo_id is not None and geo_ids is None:
            geo_ids = [geo_id]

        for chunk_first, chunk_last in chunks:
            chunk_start, chunk_end = day_bounds(chunk_first, chunk_last)
            params: dict[str, Any] = {
                "start_date": chunk_start.isoformat(),
                "end_date": chunk_end.isoformat(),
            }
            if time_trunc is not None:
                params["time_trunc"] = time_trunc
            if geo_ids:
                params["geo_ids[]"] = list(geo_ids)

            payload = self._get(f"/indicators/{indicator_id}", params)
            frame = _values_to_frame(payload, indicator_id)
            # The end bound is exclusive: ESIOS is inclusive at both ends for
            # some series, so the first period of the next month would
            # otherwise arrive twice and collide with the next chunk.
            frame = frame[frame.index < chunk_end.tz_convert("UTC")]
            frames.append(frame)
            if progress:
                print(
                    f"  {indicator_id}: {chunk_first}..{chunk_last} "
                    f"{len(frame):>6} rows",
                    flush=True,
                )

        if not frames:
            raise EsiosError(f"indicator {indicator_id}: no chunks requested")
        combined = pd.concat(frames).sort_index()
        if combined.empty:
            raise EsiosError(
                f"indicator {indicator_id} returned no values for "
                f"{start.date()}..{(end - pd.Timedelta(days=1)).date()}"
            )

        combined = _select_geography(combined, indicator_id, geo_id)

        # Chunks are half-open and adjacent, so after the geography is
        # resolved there is nothing legitimate left to deduplicate. A repeat
        # here means the server sent one, and dropping it quietly is how the
        # wrong series ends up frozen.
        if combined.index.has_duplicates:
            repeated = combined.index[combined.index.duplicated()].unique()
            raise EsiosError(
                f"indicator {indicator_id}: {len(repeated)} timestamps appear "
                f"more than once after selecting geography {geo_id!r}; first "
                f"is {repeated[0]}"
            )
        return combined


def _select_geography(
    frame: pd.DataFrame, indicator_id: int, geo_id: int | None
) -> pd.DataFrame:
    """Reduce a multi-geography response to the one geography asked for."""
    present = sorted(
        {
            (int(gid), str(name))
            for gid, name in zip(frame["geo_id"], frame["geo_name"], strict=True)
            if pd.notna(gid)
        }
    )
    if geo_id is None:
        if len(present) > 1:
            listed = ", ".join(f"{gid} ({name})" for gid, name in present)
            raise EsiosError(
                f"indicator {indicator_id} carries {len(present)} "
                f"geographies and none was requested: {listed}. Pass geo_id "
                "explicitly — deduplicating on the timestamp would silently "
                "keep whichever the server happened to send first."
            )
        return frame

    selected = frame[frame["geo_id"] == geo_id]
    if selected.empty:
        listed = ", ".join(f"{gid} ({name})" for gid, name in present) or "none"
        raise EsiosError(
            f"indicator {indicator_id} returned no rows for geo_id {geo_id}; "
            f"it carries: {listed}"
        )
    return selected


def _values_to_frame(payload: dict[str, Any], indicator_id: int) -> pd.DataFrame:
    """Parse one ``/indicators/{id}`` response into a UTC-indexed frame."""
    indicator = payload.get("indicator")
    if not isinstance(indicator, dict):
        raise EsiosError(f"indicator {indicator_id}: response has no 'indicator' key")
    values = indicator.get("values")
    if not isinstance(values, list):
        raise EsiosError(f"indicator {indicator_id}: response has no 'values' list")
    if not values:
        return pd.DataFrame(
            {"value": [], "geo_id": [], "geo_name": []},
            index=pd.DatetimeIndex([], tz="UTC", name=INDEX_NAME),
        )

    frame = pd.DataFrame(values)
    # `datetime_utc` is preferred: `datetime` carries a local offset and on a
    # fall-back day two rows share a wall-clock label. Parsing the offset
    # form with utc=True is correct too, but only because the offset is
    # there — which is precisely the assumption worth not making.
    stamp_column = "datetime_utc" if "datetime_utc" in frame else "datetime"
    if stamp_column not in frame:
        raise EsiosError(
            f"indicator {indicator_id}: values carry neither 'datetime_utc' "
            "nor 'datetime'"
        )
    index = pd.DatetimeIndex(
        pd.to_datetime(frame[stamp_column], utc=True, format="ISO8601"),
        name=INDEX_NAME,
    )
    # Re-index the source frame *before* selecting columns. Building the
    # result from Series that still carry the parsed RangeIndex while passing
    # `index=` would make pandas align rather than attach: the two indexes
    # share no labels, so every column would silently come back NaN.
    frame = frame.set_axis(index)

    out = pd.DataFrame(index=index)
    out["value"] = pd.to_numeric(frame["value"], errors="coerce")
    out["geo_id"] = (
        frame["geo_id"].astype("Int64")
        if "geo_id" in frame.columns
        else pd.Series(pd.NA, index=index, dtype="Int64")
    )
    out["geo_name"] = (
        frame["geo_name"].astype("string")
        if "geo_name" in frame.columns
        else pd.Series(pd.NA, index=index, dtype="string")
    )
    if out["value"].isna().any():
        missing = int(out["value"].isna().sum())
        raise EsiosError(
            f"indicator {indicator_id}: {missing} of {len(out)} values are not numeric"
        )
    return out.sort_index()
