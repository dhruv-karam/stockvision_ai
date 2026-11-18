import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
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


def _safe_growth(current, previous):
    """Calculate growth only when the previous value is positive."""

    if pd.isna(current) or pd.isna(previous):
        return np.nan

    if previous <= 0:
        return np.nan

    return ((current / previous) - 1) * 100


def _safe_ratio(numerator, denominator):
    """Calculate a ratio safely."""

    if pd.isna(numerator) or pd.isna(denominator):
        return np.nan

    if denominator <= 0:
        return np.nan

    return numerator / denominator


def analyze_fundamentals(
    financials,
    market_price=None,
):
    """
    Analyse two or more years of company financial statements.

    Required financial columns:
        fiscal_year
        revenue
        operating_profit
        net_profit
        eps
        total_equity
        total_debt
        current_assets
        current_liabilities

    Optional columns:
        ebit
        capital_employed

    Financial statement amounts must use the same currency unit,
    for example INR crore.

    EPS and market_price must both be in INR per share.

    Returns:
        Dictionary of calculated financial metrics.
    """

    missing = set(REQUIRED_COLUMNS) - set(financials.columns)

    if missing:
        raise ValueError(
            "Missing financial columns: "
            + ", ".join(sorted(missing))
        )

    if len(financials) < 2:
        raise ValueError(
            "At least two financial years are required."
        )

    df = financials.copy()

    # Convert financial fields to numeric values.
    numeric_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column != "fiscal_year"
    ]

    optional_columns = [
        "ebit",
        "capital_employed",
    ]

    for column in numeric_columns + optional_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    # Reject missing required financial values.
    if df[numeric_columns].isna().any().any():
        raise ValueError(
            "Required financial fields contain missing or "
            "invalid numbers. Check your financial data."
        )

    # Sort financial years in chronological order.
    df["_year_number"] = pd.to_numeric(
        df["fiscal_year"],
        errors="coerce",
    )

    if df["_year_number"].isna().any():
        raise ValueError(
            "Fiscal years must be numeric, such as 2024 and 2025."
        )

    df = df.sort_values("_year_number").reset_index(drop=True)

    if df["_year_number"].duplicated().any():
        raise ValueError(
            "Duplicate fiscal years were found."
        )

    current = df.iloc[-1]
    previous = df.iloc[-2]

    revenue = float(current["revenue"])
    operating_profit = float(current["operating_profit"])
    net_profit = float(current["net_profit"])
    eps = float(current["eps"])

    total_equity = float(current["total_equity"])
    total_debt = float(current["total_debt"])

    current_assets = float(current["current_assets"])
    current_liabilities = float(current["current_liabilities"])

    if revenue <= 0:
        raise ValueError("Revenue must be positive.")

    if total_debt < 0:
        raise ValueError("Total debt cannot be negative.")

    if current_assets < 0 or current_liabilities < 0:
        raise ValueError(
            "Current assets and liabilities cannot be negative."
        )

    # Growth rates
    revenue_growth = _safe_growth(
        revenue,
        float(previous["revenue"]),
    )

    operating_profit_growth = _safe_growth(
        operating_profit,
        float(previous["operating_profit"]),
    )

    net_profit_growth = _safe_growth(
        net_profit,
        float(previous["net_profit"]),
    )

    eps_growth = _safe_growth(
        eps,
        float(previous["eps"]),
    )

    # Profitability
    operating_margin = (
        operating_profit / revenue
    ) * 100

    net_profit_margin = (
        net_profit / revenue
    ) * 100

    # Return on Equity using average shareholders' equity.
    average_equity = (
        total_equity + float(previous["total_equity"])
    ) / 2

    roe = _safe_ratio(
        net_profit,
        average_equity,
    )

    roe_pct = roe * 100 if pd.notna(roe) else np.nan

    # Return on Capital Employed.
    roce_pct = np.nan

    if (
        "ebit" in df.columns
        and "capital_employed" in df.columns
    ):
        if (
            pd.notna(current["ebit"])
            and pd.notna(current["capital_employed"])
            and pd.notna(previous["capital_employed"])
        ):
            average_capital = (
                float(current["capital_employed"])
                + float(previous["capital_employed"])
            ) / 2

            roce = _safe_ratio(
                float(current["ebit"]),
                average_capital,
            )

            if pd.notna(roce):
                roce_pct = roce * 100

    # Balance sheet ratios
    debt_to_equity = _safe_ratio(
        total_debt,
        total_equity,
    )

    current_ratio = _safe_ratio(
        current_assets,
        current_liabilities,
    )

    # Price-to-Earnings ratio.
    pe_ratio = np.nan

    if market_price is not None:
        market_price = float(market_price)

        if not np.isfinite(market_price) or market_price <= 0:
            raise ValueError(
                "Market price must be a positive number."
            )

        if eps > 0:
            pe_ratio = market_price / eps

    return {
        "latest_fiscal_year": str(current["fiscal_year"]),
        "revenue": revenue,
        "operating_profit": operating_profit,
        "net_profit": net_profit,
        "eps": eps,
        "revenue_growth_pct": revenue_growth,
        "operating_profit_growth_pct": operating_profit_growth,
        "net_profit_growth_pct": net_profit_growth,
        "eps_growth_pct": eps_growth,
        "operating_margin_pct": operating_margin,
        "net_profit_margin_pct": net_profit_margin,
        "roe_pct": roe_pct,
        "roce_pct": roce_pct,
        "debt_to_equity": debt_to_equity,
        "current_ratio": current_ratio,
        "pe_ratio": pe_ratio,
    }


def load_fundamentals_csv(file_path):
    """
    Load company financial statements from a CSV file.
    """

    try:
        financials = pd.read_csv(file_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Financial data file not found: {file_path}"
        ) from exc

    missing = set(REQUIRED_COLUMNS) - set(financials.columns)

    if missing:
        raise ValueError(
            "Financial CSV is missing columns: "
            + ", ".join(sorted(missing))
        )

    return financials
