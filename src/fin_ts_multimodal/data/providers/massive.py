"""Reserved Massive US daily-data provider boundary."""

from __future__ import annotations

from .base import Instrument, ProviderFetch, RequestRecord


class MassiveProvider:
    """Typed extension point kept out of the side-project MVP license path."""

    name = "massive"
    reference_tickers_endpoint = "https://api.massive.com/v3/reference/tickers"
    daily_aggregates_template = (
        "https://api.massive.com/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
    )

    def discover(
        self,
        *,
        include_delisted: bool,
    ) -> tuple[list[Instrument], tuple[RequestRecord, ...]]:
        del include_delisted
        raise NotImplementedError(
            "Massive provider interface is reserved; enable it only after obtaining "
            "an appropriate Developer/commercial data license"
        )

    def fetch_instrument(
        self,
        instrument: Instrument,
        *,
        start: str,
        end: str,
        dataset_profile: str,
    ) -> ProviderFetch:
        del instrument, start, end, dataset_profile
        raise NotImplementedError(
            "Massive provider interface is reserved and is not part of this MVP"
        )
