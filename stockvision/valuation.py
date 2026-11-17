import pandas as pd


DEFAULT_SCENARIOS = {
    "Bear": {
        "eps_growth": 0.05,
        "future_pe": 15,
    },
    "Base": {
        "eps_growth": 0.12,
        "future_pe": 20,
    },
    "Bull": {
        "eps_growth": 0.18,
        "future_pe": 25,
    },
}


def calculate_long_term_scenarios(
    current_price,
    current_eps,
    years=3,
    scenarios=None,
):
    """
    Estimate possible future share prices using EPS growth
    and future P/E assumptions.

    This is a scenario analysis, not an intrinsic-value model
    or a prediction produced by machine learning.

    Args:
        current_price: Current share price.
        current_eps: Current annual earnings per share.
        years: Forecast period, such as 3 or 5.
        scenarios: Optional dictionary of scenario assumptions.

    Returns:
        DataFrame containing scenario results.
    """

    if current_price <= 0:
        raise ValueError("Current share price must be positive.")

    if current_eps <= 0:
        raise ValueError(
            "Current EPS must be positive for this growth-based model. "
            "Companies with negative earnings need a different valuation method."
        )

    if years <= 0:
        raise ValueError("Forecast period must be positive.")

    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS

    if not scenarios:
        raise ValueError("At least one valuation scenario is required.")

    results = []

    for scenario_name, assumptions in scenarios.items():

        if "eps_growth" not in assumptions:
            raise ValueError(
                f"EPS growth assumption missing for {scenario_name}."
            )

        if "future_pe" not in assumptions:
            raise ValueError(
                f"Future P/E assumption missing for {scenario_name}."
            )

        eps_growth = float(assumptions["eps_growth"])
        future_pe = float(assumptions["future_pe"])

        if eps_growth <= -1:
            raise ValueError(
                f"EPS growth for {scenario_name} must be greater than -100%."
            )

        if future_pe <= 0:
            raise ValueError(
                f"Future P/E for {scenario_name} must be positive."
            )

        # Estimate future earnings per share.
        future_eps = current_eps * (
            (1 + eps_growth) ** years
        )

        # Estimate future share price.
        future_price = future_eps * future_pe

        # Price return excludes dividends and taxes.
        total_price_return = (
            future_price / current_price
        ) - 1

        # Annualized price return.
        annualized_return = (
            (future_price / current_price) ** (1 / years)
        ) - 1

        results.append({
            "scenario": scenario_name,
            "years": years,
            "eps_growth_pct": eps_growth * 100,
            "future_pe": future_pe,
            "estimated_future_eps": future_eps,
            "estimated_future_price": future_price,
            "total_price_return_pct": total_price_return * 100,
            "annualized_price_return_pct": annualized_return * 100,
        })

    return (
        pd.DataFrame(results)
        .sort_values("estimated_future_price")
        .reset_index(drop=True)
    )


def build_pe_sensitivity(
    current_price,
    current_eps,
    eps_growth_rates=None,
    pe_multiples=None,
    years=5,
):
    """
    Build a sensitivity table showing estimated future prices
    for different EPS growth rates and future P/E multiples.

    Growth rates should be decimals:
        0.10 = 10%
        0.15 = 15%

    P/E multiples should be positive numbers.
    """

    if current_price <= 0 or current_eps <= 0:
        raise ValueError(
            "Current price and EPS must both be positive."
        )

    if years <= 0:
        raise ValueError("Forecast period must be positive.")

    if eps_growth_rates is None:
        eps_growth_rates = [
            0.05,
            0.10,
            0.15,
            0.20,
        ]

    if pe_multiples is None:
        pe_multiples = [
            12,
            15,
            20,
            25,
            30,
        ]

    if not eps_growth_rates or not pe_multiples:
        raise ValueError(
            "Growth rates and P/E multiples cannot be empty."
        )

    results = []

    for growth in eps_growth_rates:

        growth = float(growth)

        if growth <= -1:
            raise ValueError(
                "EPS growth rates must be greater than -100%."
            )

        future_eps = current_eps * (
            (1 + growth) ** years
        )

        row = {
            "eps_growth_pct": growth * 100
        }

        for pe in pe_multiples:

            pe = float(pe)

            if pe <= 0:
                raise ValueError(
                    "All P/E multiples must be positive."
                )

            future_price = future_eps * pe

            row[f"PE_{pe:g}"] = future_price

        results.append(row)

    return pd.DataFrame(results)
