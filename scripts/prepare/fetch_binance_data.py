# HC_IA_RETAIL/scripts/prepare/fetch_binance_data.py
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import requests


# Binance limits: max 1000 klines per request
MAX_LIMIT = 1000

SPOT_BASE = "https://api.binance.com"
FUTURES_BASE = "https://fapi.binance.com"

SPOT_KLINES_ENDPOINT = "/api/v3/klines"
FUTURES_KLINES_ENDPOINT = "/fapi/v1/klines"

FUTURES_FUNDING_ENDPOINT = "/fapi/v1/fundingRate"


Market = Literal["futures", "spot"]


@dataclass(frozen=True)
class FetchResult:
    df_ohlcv: pd.DataFrame
    out_path: Path


def parse_time_to_ms(s: str) -> int:
    """Accept:
    - integer ms timestamp as string
    - ISO date like '2023-01-01' or '2023-01-01T00:00:00'

    Returns milliseconds since epoch.
    """
    s = s.strip()
    if s.isdigit():
        return int(s)
    ts = pd.to_datetime(s, utc=True)
    return int(ts.value // 10**6)


def _base_and_endpoint(market: Market) -> tuple[str, str]:
    if market == "futures":
        return FUTURES_BASE, FUTURES_KLINES_ENDPOINT
    if market == "spot":
        return SPOT_BASE, SPOT_KLINES_ENDPOINT
    raise ValueError(f"Unsupported market={market!r}")


def fetch_klines(
    *,
    symbol: str,
    interval: str,
    start_ms: int | None,
    end_ms: int | None,
    market: Market = "futures",
    limit: int = MAX_LIMIT,
    sleep_s: float = 0.2,
) -> list[list]:
    """Fetch klines from Binance public API, paginating by startTime.

    Returns raw kline rows as lists.

    Spot:    https://api.binance.com/api/v3/klines
    Futures: https://fapi.binance.com/fapi/v1/klines
    """
    base_url, endpoint = _base_and_endpoint(market)

    out: list[list] = []
    params: dict[str, object] = {"symbol": symbol, "interval": interval, "limit": int(limit)}

    if end_ms is not None:
        params["endTime"] = int(end_ms)

    cur_start = start_ms
    n_calls = 0

    while True:
        if cur_start is not None:
            params["startTime"] = int(cur_start)
        else:
            params.pop("startTime", None)

        r = requests.get(base_url + endpoint, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        if not isinstance(data, list) or len(data) == 0:
            break

        out.extend(data)
        n_calls += 1

        # next page start = last close time + 1 ms
        last_close_time = int(data[-1][6])
        next_start = last_close_time + 1

        # stop conditions
        if len(data) < limit:
            break
        if end_ms is not None and next_start > end_ms:
            break

        cur_start = next_start
        time.sleep(sleep_s)

        # safety: avoid infinite loops
        if n_calls > 20000:
            raise RuntimeError("Too many pagination calls; aborting (check parameters).")

    return out


def to_ohlcv_df(raw: list[list], *, interval: str) -> pd.DataFrame:
    """Convert Binance raw kline rows to a clean OHLCV dataframe.

    Invariants enforced:
    - timestamp is UTC tz-naive (represents UTC)
    - strictly sorted
    - unique timestamps
    - numeric float columns
    - NaN/inf removed
    - gap detection reported (diff != expected interval)
    """
    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ]
    df = pd.DataFrame(raw, columns=cols)

    # cast numeric
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    # timestamps: open_time -> UTC, then drop tz to make tz-naive (but representing UTC)
    ts_utc = pd.to_datetime(df["open_time"].astype(np.int64), unit="ms", utc=True)
    df["timestamp"] = ts_utc.dt.tz_convert("UTC").dt.tz_localize(None)

    # keep only what we need
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()

    # sort + dedupe
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    # remove NaN/inf
    before = len(df)
    num_cols = ["open", "high", "low", "close", "volume"]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["timestamp"] + num_cols).reset_index(drop=True)
    after = len(df)
    dropped = before - after
    if dropped > 0:
        print(f"[WARN] Dropped {dropped} rows due to NaN/inf.")

    # gap detection
    expected = pd.to_timedelta(interval)
    if expected is pd.NaT:
        # pd.to_timedelta doesn't parse Binance intervals like '1h' in older pandas versions sometimes;
        # fallback mapping for common cases
        mapping = {
            "1m": pd.Timedelta(minutes=1),
            "3m": pd.Timedelta(minutes=3),
            "5m": pd.Timedelta(minutes=5),
            "15m": pd.Timedelta(minutes=15),
            "30m": pd.Timedelta(minutes=30),
            "1h": pd.Timedelta(hours=1),
            "2h": pd.Timedelta(hours=2),
            "4h": pd.Timedelta(hours=4),
            "6h": pd.Timedelta(hours=6),
            "8h": pd.Timedelta(hours=8),
            "12h": pd.Timedelta(hours=12),
            "1d": pd.Timedelta(days=1),
        }
        expected = mapping.get(interval, pd.Timedelta(hours=1))

    if len(df) >= 2:
        diffs = df["timestamp"].diff()
        gaps = diffs.ne(expected) & diffs.notna()
        n_gaps = int(gaps.sum())
        if n_gaps > 0:
            # Show up to 5 examples
            idxs = df.index[gaps].tolist()[:5]
            examples = []
            for i in idxs:
                prev_ts = df.loc[i - 1, "timestamp"]
                cur_ts = df.loc[i, "timestamp"]
                examples.append(f"{prev_ts} -> {cur_ts} (Δ={cur_ts - prev_ts})")
            print(f"[WARN] Detected {n_gaps} timestamp gaps (expected step={expected}).")
            for ex in examples:
                print(f"        gap: {ex}")
        else:
            print(f"[OK] No gaps detected (expected step={expected}).")

    return df


def fetch_funding_rates(
    *,
    symbol: str,
    start_ms: int | None,
    end_ms: int | None,
    sleep_s: float = 0.2,
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch Binance Futures funding rate history.

    Endpoint:
      https://fapi.binance.com/fapi/v1/fundingRate

    Returns dataframe with columns:
      - funding_time_utc (tz-naive, represents UTC)
      - funding_rate (float)

    Note: Funding is typically every 8 hours.
    """
    params: dict[str, object] = {"symbol": symbol, "limit": int(limit)}
    if start_ms is not None:
        params["startTime"] = int(start_ms)
    if end_ms is not None:
        params["endTime"] = int(end_ms)

    out_rows: list[dict[str, object]] = []
    n_calls = 0

    # Pagination: advance by last fundingTime + 1ms
    cur_start = start_ms

    while True:
        if cur_start is not None:
            params["startTime"] = int(cur_start)
        else:
            params.pop("startTime", None)

        r = requests.get(FUTURES_BASE + FUTURES_FUNDING_ENDPOINT, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        if not isinstance(data, list) or len(data) == 0:
            break

        for row in data:
            # row: {"symbol":..., "fundingTime":..., "fundingRate":...}
            ft = int(row.get("fundingTime"))
            fr = float(row.get("fundingRate"))
            out_rows.append({"funding_time_utc": ft, "funding_rate": fr})

        n_calls += 1

        last_ft = int(data[-1].get("fundingTime"))
        next_start = last_ft + 1

        if len(data) < limit:
            break
        if end_ms is not None and next_start > end_ms:
            break

        cur_start = next_start
        time.sleep(sleep_s)

        if n_calls > 20000:
            raise RuntimeError("Too many funding pagination calls; aborting (check parameters).")

    df = pd.DataFrame(out_rows)
    if df.empty:
        return df

    # funding_time -> UTC naive
    ts_utc = pd.to_datetime(df["funding_time_utc"].astype(np.int64), unit="ms", utc=True)
    df["funding_time_utc"] = ts_utc.dt.tz_convert("UTC").dt.tz_localize(None)

    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce").astype(float)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["funding_time_utc", "funding_rate"]).copy()
    df = df.sort_values("funding_time_utc").drop_duplicates(subset=["funding_time_utc"]).reset_index(drop=True)

    return df


def _default_out_path(symbol: str, interval: str, market: Market) -> Path:
    # Explicit naming for the official dataset
    if market == "futures" and symbol.upper() == "BTCUSDT" and interval == "1h":
        return (Path(__file__).resolve().parents[2] / "data" / "BTCUSDT_PERP_1h.parquet").resolve()
    # fallback
    suffix = "PERP" if market == "futures" else "SPOT"
    return (Path(__file__).resolve().parents[2] / "data" / f"{symbol.upper()}_{suffix}_{interval}.parquet").resolve()


def _default_funding_out_path(symbol: str) -> Path:
    if symbol.upper() == "BTCUSDT":
        return (Path(__file__).resolve().parents[2] / "data" / "BTCUSDT_PERP_funding_8h.parquet").resolve()
    return (Path(__file__).resolve().parents[2] / "data" / f"{symbol.upper()}_PERP_funding_8h.parquet").resolve()


def fetch_and_save_ohlcv(
    *,
    symbol: str,
    interval: str,
    start: str,
    end: str | None,
    market: Market,
    out_path: Path,
    sleep_s: float,
) -> FetchResult:
    start_ms = parse_time_to_ms(start) if start else None
    end_ms = parse_time_to_ms(end) if end else None

    print(f"[INFO] Fetching Binance {market} klines: symbol={symbol} interval={interval}")
    print(f"[INFO] start={start if start else '(none)'} end={end if end else '(none)'}")

    raw = fetch_klines(
        symbol=symbol,
        interval=interval,
        start_ms=start_ms,
        end_ms=end_ms,
        market=market,
        sleep_s=sleep_s,
    )

    print(f"[INFO] Raw rows fetched: {len(raw)}")
    df = to_ohlcv_df(raw, interval=interval)

    if len(df) < 1000:
        print("[WARN] Very few rows; check symbol/interval/start/end.")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save parquet by default; CSV supported by extension
    if out_path.suffix.lower() == ".csv":
        df.to_csv(out_path, index=False)
    else:
        df.to_parquet(out_path, index=False)

    print(f"[OK] Saved OHLCV -> {out_path}")
    print(df.head(3))
    print(df.tail(3))

    return FetchResult(df_ohlcv=df, out_path=out_path)


def save_funding(
    *,
    symbol: str,
    start: str,
    end: str | None,
    out_path: Path,
    sleep_s: float,
) -> None:
    start_ms = parse_time_to_ms(start) if start else None
    end_ms = parse_time_to_ms(end) if end else None

    print(f"[INFO] Fetching funding rates (Futures): symbol={symbol}")
    print(f"[INFO] start={start if start else '(none)'} end={end if end else '(none)'}")

    df = fetch_funding_rates(symbol=symbol, start_ms=start_ms, end_ms=end_ms, sleep_s=sleep_s)

    if df.empty:
        print("[WARN] Funding dataframe is empty. (Maybe too short range or API returned no data.)")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() == ".csv":
        df.to_csv(out_path, index=False)
    else:
        df.to_parquet(out_path, index=False)

    print(f"[OK] Saved funding -> {out_path}")
    print(df.head(3))
    print(df.tail(3))


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch Binance OHLCV (spot or futures) + optional futures funding rates.")
    ap.add_argument("--symbol", type=str, default="BTCUSDT")
    ap.add_argument("--interval", type=str, default="1h")
    ap.add_argument("--start", type=str, default="2020-01-01", help="ISO date or ms timestamp")
    ap.add_argument("--end", type=str, default="", help="ISO date or ms timestamp (optional)")
    ap.add_argument(
        "--out_path",
        type=str,
        default="",
        help="Output path (.parquet or .csv). Default: data/BTCUSDT_PERP_1h.parquet for futures BTCUSDT 1h",
    )
    ap.add_argument(
        "--market",
        type=str,
        default="futures",
        choices=["futures", "spot"],
        help="Data source market. Default futures. Spot requires --allow_spot.",
    )
    ap.add_argument(
        "--allow_spot",
        action="store_true",
        help="Safety: required to allow market=spot. Without this flag, spot is refused.",
    )
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--fetch_funding", action="store_true", help="Also fetch futures funding rates and save parquet.")
    ap.add_argument(
        "--funding_out_path",
        type=str,
        default="",
        help="Funding output path (.parquet or .csv). Default: data/BTCUSDT_PERP_funding_8h.parquet",
    )

    args = ap.parse_args()

    symbol = args.symbol.upper()
    interval = args.interval
    market: Market = "futures" if args.market == "futures" else "spot"

    # Safety: prevent accidentally running spot data unless explicitly allowed.
    if market == "spot" and not args.allow_spot:
        raise SystemExit(
            "Refusing to run with market=spot without explicit --allow_spot. "
            "Use --market spot --allow_spot if you really want spot."
        )

    end = args.end.strip() or None

    if args.out_path:
        out_path = Path(args.out_path).expanduser().resolve()
    else:
        out_path = _default_out_path(symbol, interval, market)

    fetch_and_save_ohlcv(
        symbol=symbol,
        interval=interval,
        start=args.start,
        end=end,
        market=market,
        out_path=out_path,
        sleep_s=float(args.sleep),
    )

    if args.fetch_funding:
        if market != "futures":
            print("[WARN] --fetch_funding requested but market=spot. Funding is futures-only; skipping.")
        else:
            if args.funding_out_path:
                f_out = Path(args.funding_out_path).expanduser().resolve()
            else:
                f_out = _default_funding_out_path(symbol)

            save_funding(symbol=symbol, start=args.start, end=end, out_path=f_out, sleep_s=float(args.sleep))


if __name__ == "__main__":
    main()