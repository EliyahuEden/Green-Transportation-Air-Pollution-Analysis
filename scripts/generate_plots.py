from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = PROJECT_ROOT / "plots"

FEATURES = [
    "EV_per_density",
    "Public_EV_Chargers_per_density",
    "Cycling_Routes_km_per_area",
    "Public_Bike_Companies_per_density",
    "Electric_Buses_per_density",
    "Airports_per_density",
    "Bus_Fare",
]

FEATURE_LABELS = {
    "EV_per_density": "EVs per population density",
    "Public_EV_Chargers_per_density": "EV chargers per population density",
    "Cycling_Routes_km_per_area": "Cycling route km per km2",
    "Public_Bike_Companies_per_density": "Bike companies per population density",
    "Electric_Buses_per_density": "Electric buses per population density",
    "Airports_per_density": "Airports per population density",
    "Bus_Fare": "Bus fare",
}


def build_dataset() -> pd.DataFrame:
    data = {
        "City": [
            "London",
            "Amsterdam",
            "Vienna",
            "Berlin",
            "Helsinki",
            "Paris",
            "Oslo",
            "Andorra",
            "Brussels",
            "Luxembourg",
            "Dublin",
            "Madrid",
            "Rome",
            "Sofia",
            "Budapest",
            "Copenhagen",
            "Tirana",
            "Zagreb",
            "Athens",
            "Vilnius",
            "Nicosia",
            "Vaduz",
            "Skopje",
            "Sarajevo",
        ],
        "EV": [
            80000,
            15000,
            18000,
            30000,
            25100,
            20000,
            95466,
            100,
            14000,
            10000,
            20000,
            30000,
            10000,
            3000,
            10000,
            10000,
            100,
            2000,
            2500,
            2000,
            500,
            300,
            500,
            100,
        ],
        "Public_EV_Chargers": [
            11557,
            13549,
            1374,
            3838,
            146,
            1043,
            535,
            28,
            620,
            527,
            141,
            1776,
            842,
            64,
            669,
            838,
            5,
            203,
            63,
            31,
            23,
            17,
            14,
            31,
        ],
        "Cycling_Routes_km": [
            97,
            858,
            1300,
            1000,
            1300,
            1000,
            327,
            350,
            650,
            78,
            190,
            300,
            320,
            70,
            200,
            400,
            50,
            220,
            50,
            120,
            20,
            90,
            20,
            15,
        ],
        "Public_Bike_Companies": [
            3,
            5,
            6,
            6,
            3,
            6,
            4,
            5,
            7,
            3,
            4,
            1,
            3,
            3,
            2,
            4,
            4,
            2,
            3,
            1,
            2,
            1,
            1,
            1,
        ],
        "Bus_Fare": [
            2.08,
            3.40,
            2.40,
            3.20,
            2.95,
            2.50,
            3.24,
            1.30,
            2.40,
            0.00,
            1.70,
            1.50,
            1.50,
            0.50,
            0.90,
            3.22,
            0.40,
            0.93,
            1.40,
            0.90,
            1.50,
            3.19,
            0.60,
            0.80,
        ],
        "Airports": [
            6,
            1,
            1,
            1,
            1,
            3,
            1,
            0,
            2,
            1,
            1,
            1,
            2,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
        ],
        "Electric_Buses": [
            1397,
            75,
            150,
            230,
            428,
            500,
            183,
            10,
            75,
            30,
            100,
            150,
            400,
            50,
            40,
            40,
            10,
            30,
            250,
            30,
            10,
            5,
            10,
            10,
        ],
        "Overall_Score": [
            5.87,
            5.71,
            5.70,
            5.52,
            5.36,
            5.25,
            5.20,
            4.97,
            4.93,
            4.61,
            4.60,
            4.36,
            4.26,
            4.18,
            4.12,
            4.09,
            4.09,
            3.83,
            3.80,
            3.76,
            3.64,
            3.28,
            3.07,
            2.80,
        ],
        "Air_Pollution": [
            8.40,
            9.10,
            9.10,
            10.50,
            4.90,
            10.30,
            6.20,
            5.50,
            9.80,
            8.80,
            6.30,
            9.00,
            13.10,
            12.00,
            11.70,
            7.90,
            16.70,
            14.90,
            16.70,
            10.60,
            14.10,
            7.20,
            24.60,
            28.60,
        ],
        "Density": [
            5690,
            5621,
            4556,
            4090,
            3113,
            20909,
            1699,
            176,
            7465,
            2600,
            4708,
            5390,
            2137,
            1100,
            3337,
            4417,
            538,
            1200,
            19135,
            1515,
            1700,
            330,
            950,
            2470,
        ],
        "Area_km2": [
            1572,
            219.3,
            414.6,
            891.8,
            213.8,
            105.4,
            454,
            468,
            32.61,
            51.73,
            115,
            604.3,
            1285,
            500,
            525.2,
            179.8,
            41.8,
            641,
            38.96,
            401,
            111,
            17.3,
            1854,
            142,
        ],
    }

    df = pd.DataFrame(data)
    df["Cycling_Routes_km_per_area"] = df["Cycling_Routes_km"] / df["Area_km2"]
    for col in [
        "EV",
        "Public_EV_Chargers",
        "Public_Bike_Companies",
        "Electric_Buses",
        "Airports",
    ]:
        df[f"{col}_per_density"] = df[col] / df["Density"]
    return df


def save_figure(fig: plt.Figure, filename: str) -> Path:
    PLOTS_DIR.mkdir(exist_ok=True)
    output_path = PLOTS_DIR / filename
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_air_pollution_by_city(df: pd.DataFrame) -> Path:
    sorted_df = df.sort_values("Air_Pollution", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    colors = sns.color_palette("rocket_r", n_colors=len(sorted_df))
    ax.barh(sorted_df["City"], sorted_df["Air_Pollution"], color=colors)
    ax.set_title("Air pollution by city")
    ax.set_xlabel("Air pollution index")
    ax.set_ylabel("")
    ax.set_xlim(0, sorted_df["Air_Pollution"].max() + 4)

    for i, value in enumerate(sorted_df["Air_Pollution"]):
        ax.text(value + 0.25, i, f"{value:.1f}", va="center", fontsize=8)

    sns.despine(ax=ax, left=True)
    return save_figure(fig, "air_pollution_by_city.png")


def plot_score_vs_pollution(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.regplot(
        data=df,
        x="Overall_Score",
        y="Air_Pollution",
        ax=ax,
        ci=None,
        scatter_kws={"s": 55, "alpha": 0.85, "color": "#2f6f73"},
        line_kws={"color": "#c44e52", "linewidth": 2},
    )
    for _, row in df.iterrows():
        ax.annotate(
            row["City"],
            (row["Overall_Score"], row["Air_Pollution"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set_title("Overall green transport score vs air pollution")
    ax.set_xlabel("Overall green transportation score")
    ax.set_ylabel("Air pollution index")
    ax.set_ylim(0, df["Air_Pollution"].max() + 3)
    sns.despine(ax=ax)
    return save_figure(fig, "overall_score_vs_air_pollution.png")


def plot_spearman_correlations(df: pd.DataFrame) -> Path:
    rows = []
    for feature in FEATURES:
        rho, p_value = spearmanr(df[feature], df["Air_Pollution"])
        rows.append(
            {
                "Feature": feature,
                "Label": FEATURE_LABELS[feature],
                "Spearman rho": rho,
                "p-value": p_value,
            }
        )

    corr_df = pd.DataFrame(rows).sort_values("Spearman rho")
    colors = ["#2f6f73" if value < 0 else "#c44e52" for value in corr_df["Spearman rho"]]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(corr_df["Label"], corr_df["Spearman rho"], color=colors)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_title("Spearman correlations with air pollution")
    ax.set_xlabel("Spearman rho")
    ax.set_ylabel("")
    ax.set_xlim(-0.65, 0.55)

    for i, row in enumerate(corr_df.itertuples(index=False)):
        p_value = getattr(row, "_3")
        ax.text(0.39, i, f"p={p_value:.3f}", va="center", ha="left", fontsize=8)

    sns.despine(ax=ax, left=True)
    return save_figure(fig, "spearman_correlations.png")


def plot_normalized_metric_relationships(df: pd.DataFrame) -> Path:
    selected_features = [
        "EV_per_density",
        "Public_EV_Chargers_per_density",
        "Cycling_Routes_km_per_area",
        "Electric_Buses_per_density",
        "Public_Bike_Companies_per_density",
        "Bus_Fare",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, feature in zip(axes.flat, selected_features):
        rho, _ = spearmanr(df[feature], df["Air_Pollution"])
        sns.regplot(
            data=df,
            x=feature,
            y="Air_Pollution",
            ax=ax,
            ci=None,
            scatter_kws={"s": 38, "alpha": 0.8, "color": "#4c72b0"},
            line_kws={"color": "#dd8452", "linewidth": 1.8},
        )
        ax.set_title(f"{FEATURE_LABELS[feature]}\nrho={rho:.2f}", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("Air pollution index")
        ax.set_ylim(0, df["Air_Pollution"].max() + 3)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 3))

    return save_figure(fig, "normalized_metric_relationships.png")


def plot_cycling_tertiles(df: pd.DataFrame) -> Path:
    plot_df = df.copy()
    order = ["Low", "Medium", "High"]
    plot_df["Cycling route density"] = pd.qcut(
        plot_df["Cycling_Routes_km_per_area"],
        3,
        labels=order,
    )

    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.boxplot(
        data=plot_df,
        x="Cycling route density",
        y="Air_Pollution",
        order=order,
        color="#8ab17d",
        width=0.5,
        ax=ax,
    )
    sns.stripplot(
        data=plot_df,
        x="Cycling route density",
        y="Air_Pollution",
        order=order,
        color="#264653",
        size=5,
        alpha=0.85,
        ax=ax,
    )
    ax.set_title("Air pollution by cycling-route density tertile")
    ax.set_xlabel("Cycling route km per km2 tertile")
    ax.set_ylabel("Air pollution index")
    sns.despine(ax=ax)
    return save_figure(fig, "cycling_tertile_air_pollution.png")


def plot_feature_importance(df: pd.DataFrame) -> Path:
    model = GradientBoostingRegressor(random_state=42)
    model.fit(df[FEATURES], df["Air_Pollution"])

    importances = (
        pd.Series(model.feature_importances_, index=FEATURES)
        .sort_values(ascending=True)
        .rename(index=FEATURE_LABELS)
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(importances.index, importances.values, color="#2f6f73")
    ax.set_title("Gradient boosting feature importance")
    ax.set_xlabel("Relative importance")
    ax.set_ylabel("")

    for i, value in enumerate(importances.values):
        ax.text(value + 0.005, i, f"{value:.2f}", va="center", fontsize=8)

    ax.set_xlim(0, max(importances.values) + 0.08)
    sns.despine(ax=ax, left=True)
    return save_figure(fig, "gradient_boosting_feature_importance.png")


def main() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    df = build_dataset()
    plotters = [
        plot_air_pollution_by_city,
        plot_score_vs_pollution,
        plot_spearman_correlations,
        plot_normalized_metric_relationships,
        plot_cycling_tertiles,
        plot_feature_importance,
    ]

    outputs = [plotter(df) for plotter in plotters]
    print("Generated plots:")
    for output in outputs:
        print(output.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
