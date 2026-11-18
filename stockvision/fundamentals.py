
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from stockvision.valuation import calculate_long_term_scenarios


# ==================================================
# FINANCIAL DATA HELPERS
# ==================================================

def _normalize_label(value):
    """Normalize financial statement row names."""

    return "".join(
        character.lower()
        for character in str(value)
        if character.isalnum()
    )


def _find_row(statement, aliases):
    """Find a financial statement row using alternative names."""

    if statement is None or statement.empty:
        return None

    normalized_rows = {
        _normalize_label(index): index
        for index in statement.index
    }

    # Prefer exact matches.
    for alias in aliases:
        key = _normalize_label(alias)

        if key in normalized_rows:
            return normalized_rows[key]

    return None


def _get_series(statement, aliases):
    """Extract a financial metric across available financial years."""

    row = _find_row(statement, aliases)

    if row is None:
        return pd.Series(dtype=float)

    values = pd.to_numeric(
        statement.loc[row],
        errors="coerce",
    )

    records = []

    for column, value in values.items():

        try:
            year_date = pd.Timestamp(column)

            if pd.isna(year_date):
                continue

        except (ValueError, TypeError):
            continue

        if pd.notna(value):
            records.append((year_date, float(value)))

    if not records:
        return pd.Series(dtype=float)

    result = pd.Series(
        data=[record[1] for record in records],
        index=[record[0] for record in records],
        dtype=float,
    )

    return result.sort_index()


def _latest_value(series):
    """Get the latest available financial value."""

    if series is None or series.empty:
        return np.nan

    valid = series.dropna()

    if valid.empty:
        return np.nan

    return float(valid.iloc[-1])


def _calculate_cagr(series, periods=4):
    """
    Calculate CAGR using available annual observations.

    Uses no less than two positive financial observations.
    """

    if series is None or series.empty:
        return np.nan

    valid = series.dropna().tail(periods)

    if len(valid) < 2:
        return np.nan

    first_value = float(valid.iloc[0])
    last_value = float(valid.iloc[-1])

    years = (
        valid.index[-1] - valid.index[0]
    ).days / 365.25

    if first_value <= 0 or last_value <= 0 or years <= 0:
        return np.nan

    return float(
        (last_value / first_value) ** (1 / years) - 1
    )


def _safe_ratio(numerator, denominator):
    """Safely calculate a financial ratio."""

    if pd.isna(numerator) or pd.isna(denominator):
        return np.nan

    if denominator <= 0:
        return np.nan

    return float(numerator / denominator)


def _get_statement(ticker, method_name):
    """Retrieve an annual financial statement."""

    try:
        method = getattr(ticker, method_name)

        statement = method(
            freq="yearly"
        )

        if isinstance(statement, pd.DataFrame):
            return statement

    except Exception:
        pass

    return pd.DataFrame()


# ==================================================
# AUTOMATIC COMPANY FUNDAMENTALS
# ==================================================

def fetch_auto_fundamentals(
    symbol,
    market_price,
    forecast_years=5,
):
    """
    Automatically retrieve financial data and calculate
    fundamental metrics and Bear/Base/Bull scenarios.

    NSE symbols are converted to Yahoo Finance format:
        RELIANCE -> RELIANCE.NS

    Market price is supplied by the main Kotak Neo application.

    Returns:
        company
        symbol
        latest_fiscal_year
        source
        metrics
        assumptions
        scenarios
        financial_history
        updated_at
    """

    try:
        import yfinance as yf

    except ImportError as exc:
        raise ImportError(
            "yfinance is not installed. "
            "Add yfinance to requirements.txt."
        ) from exc

    symbol = str(symbol).strip().upper()

    if not symbol:
        raise ValueError("Stock symbol cannot be empty.")

    market_price = float(market_price)

    if not np.isfinite(market_price) or market_price <= 0:
        raise ValueError("Market price must be positive.")

    if forecast_years not in (3, 5):
        raise ValueError(
            "Forecast period must be either 3 or 5 years."
        )

    ticker_symbol = (
        symbol if symbol.endswith(".NS")
        else f"{symbol}.NS"
    )

    ticker = yf.Ticker(ticker_symbol)

    # --------------------------------------------------
    # DOWNLOAD COMPANY INFORMATION
    # --------------------------------------------------

    try:
        company_info = ticker.get_info()

    except Exception as exc:
        raise RuntimeError(
            f"Unable to retrieve company information for "
            f"{symbol}: {exc}"
        ) from exc

    if not company_info:
        raise ValueError(
            f"No company information was returned for {symbol}."
        )

    # --------------------------------------------------
    # DOWNLOAD ANNUAL FINANCIAL STATEMENTS
    # --------------------------------------------------

    income = _get_statement(
        ticker,
        "get_income_stmt",
    )

    balance = _get_statement(
        ticker,
        "get_balance_sheet",
    )

    if income.empty:
        raise ValueError(
            f"No annual income statement available for {symbol}."
        )

    # --------------------------------------------------
    # EXTRACT ANNUAL FINANCIAL METRICS
    # --------------------------------------------------

    revenue = _get_series(
        income,
        [
            "Total Revenue",
            "Operating Revenue",
            "Revenue",
        ],
    )

    operating_profit = _get_series(
        income,
        [
            "Operating Income",
            "Operating Profit",
            "EBIT",
        ],
    )

    net_profit = _get_series(
        income,
        [
            "Net Income Common Stockholders",
            "Net Income",
            "Net Income From Continuing Operations",
        ],
    )

    eps_series = _get_series(
        income,
        [
            "Diluted EPS",
            "Basic EPS",
            "Normalized Diluted EPS",
        ],
    )

    equity = _get_series(
        balance,
        [
            "Stockholders Equity",
            "Total Stockholder Equity",
            "Total Equity Gross Minority Interest",
        ],
    )

    debt = _get_series(
        balance,
        [
            "Total Debt",
        ],
    )

    current_assets = _get_series(
        balance,
        [
            "Current Assets",
        ],
    )

    current_liabilities = _get_series(
        balance,
        [
            "Current Liabilities",
        ],
    )

    # --------------------------------------------------
    # LATEST FINANCIAL VALUES
    # --------------------------------------------------

    latest_revenue = _latest_value(revenue)

    latest_operating_profit = _latest_value(
        operating_profit
    )

    latest_net_profit = _latest_value(
        net_profit
    )

    latest_eps = _latest_value(
        eps_series
    )

    eps_source = "Annual financial statement"

    # Yahoo sometimes provides EPS only in its company information.
    if pd.isna(latest_eps):

        latest_eps = pd.to_numeric(
            company_info.get("trailingEps"),
            errors="coerce",
        )

        eps_source = "Yahoo trailing EPS"

    if pd.isna(latest_eps) or latest_eps <= 0:
        raise ValueError(
            f"A valid positive EPS is unavailable for {symbol}. "
            "An earnings-based P/E valuation cannot be calculated."
        )

    # --------------------------------------------------
    # FINANCIAL GROWTH
    # --------------------------------------------------

    revenue_growth = _calculate_cagr(
        revenue
    )

    profit_growth = _calculate_cagr(
        net_profit
    )

    eps_growth = _calculate_cagr(
        eps_series
    )

    growth_source = "Historical EPS CAGR"

    # Prioritize EPS growth.
    # Use net profit growth or revenue growth only as a clearly
    # identified fallback when an EPS history is unavailable.
    if pd.notna(eps_growth):

        raw_base_growth = eps_growth

    elif pd.notna(profit_growth):

        raw_base_growth = profit_growth

        growth_source = "Historical net profit CAGR proxy"

    elif pd.notna(revenue_growth):

        raw_base_growth = revenue_growth

        growth_source = "Historical revenue CAGR proxy"

    else:

        raise ValueError(
            f"Insufficient positive historical growth data for {symbol}. "
            "Automatic long-term scenarios cannot be calculated safely."
        )

    # Conservative bounds prevent extreme historical growth
    # from being extrapolated without restraint.
    base_growth = float(
        np.clip(raw_base_growth, -0.10, 0.25)
    )

    bear_growth = float(
        np.clip(base_growth - 0.05, -0.20, 0.25)
    )

    bull_growth = float(
        np.clip(base_growth + 0.05, -0.15, 0.30)
    )

    # --------------------------------------------------
    # PROFITABILITY
    # --------------------------------------------------

    operating_margin = (
        latest_operating_profit / latest_revenue * 100
        if pd.notna(latest_operating_profit)
        and pd.notna(latest_revenue)
        and latest_revenue > 0
        else np.nan
    )

    net_margin = (
        latest_net_profit / latest_revenue * 100
        if pd.notna(latest_net_profit)
        and pd.notna(latest_revenue)
        and latest_revenue > 0
        else np.nan
    )

    # --------------------------------------------------
    # BALANCE SHEET
    # --------------------------------------------------

    latest_equity = _latest_value(equity)
    latest_debt = _latest_value(debt)

    latest_current_assets = _latest_value(
        current_assets
    )

    latest_current_liabilities = _latest_value(
        current_liabilities
    )

    debt_to_equity = _safe_ratio(
        latest_debt,
        latest_equity,
    )

    current_ratio = _safe_ratio(
        latest_current_assets,
        latest_current_liabilities,
    )

    # --------------------------------------------------
    # ROE
    # --------------------------------------------------

    roe = np.nan

    if (
        not equity.empty
        and pd.notna(latest_net_profit)
    ):

        equity_values = equity.dropna()

        if len(equity_values) >= 2:

            average_equity = (
                float(equity_values.iloc[-1])
                + float(equity_values.iloc[-2])
            ) / 2

            roe = (
                _safe_ratio(
                    latest_net_profit,
                    average_equity,
                ) * 100
                if average_equity > 0
                else np.nan
            )

    # --------------------------------------------------
    # ROCE
    # --------------------------------------------------

    roce = np.nan

    if (
        pd.notna(latest_operating_profit)
        and pd.notna(latest_equity)
        and pd.notna(latest_debt)
    ):

        capital_employed = (
            latest_equity + latest_debt
        )

        roce = (
            _safe_ratio(
                latest_operating_profit,
                capital_employed,
            ) * 100
        )

    # --------------------------------------------------
    # AUTOMATIC P/E AND SCENARIOS
    # --------------------------------------------------

    current_pe = market_price / latest_eps

    if not np.isfinite(current_pe) or current_pe <= 0:
        raise ValueError(
            "A valid current P/E could not be calculated."
        )

    # The future P/E is stress-tested around today's
    # earnings multiple instead of being requested manually.
    bear_pe = current_pe * 0.80
    base_pe = current_pe
    bull_pe = current_pe * 1.20

    scenario_inputs = {
        "Bear": {
            "eps_growth": bear_growth,
            "future_pe": bear_pe,
        },
        "Base": {
            "eps_growth": base_growth,
            "future_pe": base_pe,
        },
        "Bull": {
            "eps_growth": bull_growth,
            "future_pe": bull_pe,
        },
    }

    scenarios = calculate_long_term_scenarios(
        current_price=market_price,
        current_eps=float(latest_eps),
        years=forecast_years,
        scenarios=scenario_inputs,
    )

    # --------------------------------------------------
    # FINANCIAL HISTORY
    # --------------------------------------------------

    history = pd.DataFrame({
        "Revenue": revenue,
        "Operating Profit": operating_profit,
        "Net Profit": net_profit,
        "EPS": eps_series,
    })

    history.index.name = "Fiscal Year"

    history = history.reset_index()

    # --------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------

    latest_period = (
        income.columns[-1]
        if len(income.columns)
        else None
    )

    return {
        "symbol": symbol,

        "company": (
            company_info.get("longName")
            or company_info.get("shortName")
            or symbol
        ),

        "latest_fiscal_year": str(
            pd.to_datetime(latest_period).year
            if latest_period is not None
            else "Latest reported"
        ),

        "source": "Yahoo Finance via yfinance",

        "updated_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "metrics": {
            "market_price": market_price,
            "eps": float(latest_eps),
            "eps_source": eps_source,
            "pe_ratio": float(current_pe),

            "revenue": latest_revenue,
            "operating_profit": latest_operating_profit,
            "net_profit": latest_net_profit,

            "revenue_growth_pct": (
                revenue_growth * 100
                if pd.notna(revenue_growth)
                else np.nan
            ),

            "profit_growth_pct": (
                profit_growth * 100
                if pd.notna(profit_growth)
                else np.nan
            ),

            "eps_growth_pct": (
                eps_growth * 100
                if pd.notna(eps_growth)
                else np.nan
            ),

            "operating_margin_pct": operating_margin,
            "net_profit_margin_pct": net_margin,

            "roe_pct": roe,
            "roce_pct": roce,

            "debt_to_equity": debt_to_equity,
            "current_ratio": current_ratio,
        },

        "assumptions": {
            "growth_method": growth_source,
            "bear_growth": bear_growth * 100,
            "base_growth": base_growth * 100,
            "bull_growth": bull_growth * 100,

            "bear_pe": bear_pe,
            "base_pe": base_pe,
            "bull_pe": bull_pe,
        },

        "scenarios": scenarios,

        "financial_history": history,
    }


# ==================================================
# MANUAL CSV ANALYSIS — BACKWARD COMPATIBILITY
# ==================================================

def analyze_fundamentals(
    financials,
    market_price=None,
):
    """
    Analyse an uploaded financial CSV.

    Retained for compatibility with the earlier dashboard.
    The new dashboard will use fetch_auto_fundamentals().
    """

    required = [
        "fiscal_year",
        "revenue",
        "operating_profit",
        "net_profit",
        "eps",
        "total_equity",
        "total_debt",
        "current_assets",
        "current_liabilities",
    ]

    missing = set(required) - set(financials.columns)

    if missing:
        raise ValueError(
            "Missing financial columns: "
            + ", ".join(sorted(missing))
        )

    df = financials.copy()

    numeric_columns = [
        column
        for column in required
        if column != "fiscal_year"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    if df[numeric_columns].isna().any().any():
        raise ValueError(
            "Financial statements contain missing or invalid values."
        )

    df["_year"] = pd.to_numeric(
        df["fiscal_year"],
        errors="coerce"
    )

    if df["_year"].isna().any():
        raise ValueError("Fiscal years must be numeric.")

    df = (
        df.sort_values("_year")
        .reset_index(drop=True)
    )

    if df["_year"].duplicated().any():
        raise ValueError("Duplicate fiscal years found.")

    if len(df) < 2:
        raise ValueError(
            "At least two years of financial history are required."
        )

    current = df.iloc[-1]
    previous = df.iloc[-2]

    revenue = float(current["revenue"])
    operating_profit = float(current["operating_profit"])
    net_profit = float(current["net_profit"])
    eps = float(current["eps"])

    equity = float(current["total_equity"])
    debt = float(current["total_debt"])

    current_assets = float(current["current_assets"])
    current_liabilities = float(current["current_liabilities"])

    if revenue <= 0 or equity <= 0:
        raise ValueError(
            "Revenue and shareholders' equity must be positive."
        )

    if eps <= 0:
        raise ValueError(
            "Positive EPS is required for P/E valuation."
        )

    def growth(current_value, previous_value):

        if previous_value <= 0:
            return np.nan

        return (
            (current_value / previous_value) - 1
        ) * 100

    average_equity = (
        equity + float(previous["total_equity"])
    ) / 2

    metrics = {
        "latest_fiscal_year": str(current["fiscal_year"]),
        "revenue": revenue,
        "operating_profit": operating_profit,
        "net_profit": net_profit,
        "eps": eps,

        "revenue_growth_pct": growth(
            revenue,
            float(previous["revenue"]),
        ),

        "operating_profit_growth_pct": growth(
            operating_profit,
            float(previous["operating_profit"]),
        ),

        "net_profit_growth_pct": growth(
            net_profit,
            float(previous["net_profit"]),
        ),

        "eps_growth_pct": growth(
            eps,
            float(previous["eps"]),
        ),

        "operating_margin_pct": (
            operating_profit / revenue * 100
        ),

        "net_profit_margin_pct": (
            net_profit / revenue * 100
        ),

        "roe_pct": (
            net_profit / average_equity * 100
            if average_equity > 0
            else np.nan
        ),

        "roce_pct": np.nan,

        "debt_to_equity": (
            debt / equity
            if equity > 0
            else np.nan
        ),

        "current_ratio": (
            current_assets / current_liabilities
            if current_liabilities > 0
            else np.nan
        ),

        "pe_ratio": (
            market_price / eps
            if market_price is not None
            and market_price > 0
            else np.nan
        ),
    }

    return metrics

