import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

API_URL = "https://bank.gov.ua/NBUStatService/v1/statdirectory/banksfinrep"


def try_dates_latest_first():
    today = datetime.today()
    candidates = []
    y, m = today.year, today.month
    for _ in range(18):
        candidates.append(f"{y}{m:02d}01")
        m -= 1
        if m == 0:
            m = 12
            y -= 1
    return candidates


def fetch_banks_finrep(date_yyyymmdd: str, period: str = "m") -> pd.DataFrame:
    url = f"{API_URL}?date={date_yyyymmdd}&json=&period={period}"
    df = pd.read_json(url)
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def pick_metric(df: pd.DataFrame, pattern_en: str = None, pattern_ua: str = None, leveli: int = None) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)

    if leveli is not None and "leveli" in df.columns:
        mask &= (df["leveli"] == leveli)

    if pattern_en and "txten" in df.columns:
        mask &= df["txten"].fillna("").str.contains(pattern_en, flags=re.I, regex=True)

    if pattern_ua and "txt" in df.columns:
        mask &= df["txt"].fillna("").str.contains(pattern_ua, flags=re.I, regex=True)

    out = df.loc[mask, ["fullname", "value"]].copy()
    out = out.dropna(subset=["fullname"])
    return out


def agg_by_bank(df_metric: pd.DataFrame, value_col: str, how: str = "max") -> pd.DataFrame:
    df_metric = df_metric.dropna(subset=["fullname", value_col]).copy()
    if df_metric.empty:
        return df_metric

    if how == "max":
        g = df_metric.groupby("fullname", as_index=False)[value_col].max()
    elif how == "sum":
        g = df_metric.groupby("fullname", as_index=False)[value_col].sum()
    elif how == "mean":
        g = df_metric.groupby("fullname", as_index=False)[value_col].mean()
    else:
        raise ValueError("how must be: max / sum / mean")

    return g


def minmax(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mn, mx = np.nanmin(s.values), np.nanmax(s.values)
    if np.isclose(mx, mn):
        return pd.Series(np.ones(len(s)), index=s.index)
    return (s - mn) / (mx - mn)


def plot_bar(df_plot: pd.DataFrame, x_col: str, y_col: str, title: str, ylabel: str):
    plt.figure(figsize=(12, 6))
    plt.bar(df_plot[x_col], df_plot[y_col])
    plt.xticks(rotation=75, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


df = None
used_date = None
for d in try_dates_latest_first():
    try:
        tmp = fetch_banks_finrep(d, period="m")
        if {"fullname", "value"}.issubset(tmp.columns) and len(tmp) > 300:
            df = tmp
            used_date = d
            break
    except Exception:
        pass

print("Дата даних (yyyymmdd):", used_date, "| рядків:", len(df))

assets = pick_metric(df, pattern_en=r"^Assets.*Total", leveli=1)
if assets.empty:
    assets = pick_metric(df, pattern_ua=r"^Активи.*Усього", leveli=1)
assets = assets.rename(columns={"value": "assets_total"})
assets = agg_by_bank(assets, "assets_total", how="max")

equity = pick_metric(df, pattern_en=r"^(Equity|Capital).*Total", leveli=1)
if equity.empty:
    equity = pick_metric(df, pattern_ua=r"(Власн.*капітал|Капітал).*Усього", leveli=1)
equity = equity.rename(columns={"value": "equity_total"})
equity = agg_by_bank(equity, "equity_total", how="max")

cash = pick_metric(df, pattern_en=r"Cash and cash equivalents", leveli=2)
if cash.empty:
    cash = pick_metric(df, pattern_ua=r"Грошові кошти та їх еквіваленти", leveli=2)
cash = cash.rename(columns={"value": "cash_total"})
cash = agg_by_bank(cash, "cash_total", how="max")

profit = pick_metric(df, pattern_en=r"(Net )?profit|Profit \(loss\)|profit", leveli=None)
if profit.empty:
    profit = pick_metric(df, pattern_ua=r"прибут|збит", leveli=None)
profit = profit.rename(columns={"value": "profit_value"})
profit = agg_by_bank(profit, "profit_value", how="max")

banks = assets.merge(equity, on="fullname", how="inner").merge(cash, on="fullname", how="left")
if not profit.empty:
    banks = banks.merge(profit, on="fullname", how="left")
else:
    banks["profit_value"] = np.nan

banks["capital"] = banks["equity_total"]
banks["liquidity"] = banks["cash_total"] / banks["assets_total"]
banks["financial_stability"] = banks["equity_total"] / banks["assets_total"]
banks["profitability"] = banks["profit_value"] / banks["assets_total"]

banks = banks.sort_values("assets_total", ascending=False).head(15).reset_index(drop=True)

metrics = {
    "assets_total": "Активи",
    "capital": "Капітал",
    "liquidity": "Ліквідність",
    "financial_stability": "Фінансова стійкість",
    "profitability": "Прибутковість",
}

if banks["profitability"].isna().all():
    metrics.pop("profitability")

for col in metrics.keys():
    banks[f"n_{col}"] = minmax(banks[col].fillna(banks[col].median()))

weights = {col: 1.0 for col in metrics.keys()}
w_sum = sum(weights.values())

banks["integral_score"] = 0.0
for col, w in weights.items():
    banks["integral_score"] += (w / w_sum) * banks[f"n_{col}"]

banks["rank"] = banks["integral_score"].rank(ascending=False, method="dense").astype(int)
banks = banks.sort_values(["rank", "integral_score"], ascending=[True, False]).reset_index(drop=True)

cols_show = [
    "rank",
    "fullname",
    "integral_score",
    "capital",
    "liquidity",
    "financial_stability",
]
if "profitability" in metrics:
    cols_show.append("profitability")

print("\nРейтинг банків (ТОП-15 за активами):")
print(banks[cols_show].round(6).to_string(index=False))

plot_bar(
    banks,
    x_col="fullname",
    y_col="integral_score",
    title=f"Рейтинг банків за інтегральним показником (дата {used_date})",
    ylabel="Integral score (0..1)"
)

plot_bar(
    banks.sort_values("capital", ascending=False),
    x_col="fullname",
    y_col="capital",
    title=f"Капітал банків (дата {used_date})",
    ylabel="Capital (абсолют)"
)

plot_bar(
    banks.sort_values("liquidity", ascending=False),
    x_col="fullname",
    y_col="liquidity",
    title=f"Ліквідність (cash/assets) (дата {used_date})",
    ylabel="Liquidity ratio"
)

plot_bar(
    banks.sort_values("financial_stability", ascending=False),
    x_col="fullname",
    y_col="financial_stability",
    title=f"Фінансова стійкість (equity/assets) (дата {used_date})",
    ylabel="Financial stability ratio"
)

if "profitability" in metrics:
    plot_bar(
        banks.sort_values("profitability", ascending=False),
        x_col="fullname",
        y_col="profitability",
        title=f"Прибутковість (profit/assets, ROA proxy) (дата {used_date})",
        ylabel="Profitability ratio"
    )

out_cols_excel = ["rank", "fullname", "integral_score", "assets_total", "capital", "liquidity", "financial_stability", "profitability"]
for c in out_cols_excel:
    if c not in banks.columns:
        banks[c] = np.nan

banks[out_cols_excel].to_excel("bank_rating.xlsx", index=False)

print("\nЗбережено файл: bank_rating.xlsx")
