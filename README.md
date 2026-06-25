# Green Transportation and Air Pollution Analysis

This project analyzes how green transportation indicators relate to air pollution across 24 European cities. The analysis compares electric vehicles, EV chargers, cycling infrastructure, public bike availability, electric buses, airports, bus fares, population density, and city area against each city's air pollution index.

The main statistical work is in `data analysis.ipynb`. The generated figures are saved in the `plots/` folder and can be recreated with `scripts/generate_plots.py`.

## Project Structure

- `data analysis.ipynb` - primary notebook with the dataset, normalized metrics, statistical tests, and model-based feature importance.
- `scripts/generate_plots.py` - reproducible script that rebuilds the notebook dataset and exports project plots.
- `plots/` - generated PNG figures for reporting and presentation use.
- Report and presentation files - project writeups and slides in PDF/DOCX/ZIP formats.

## Data

The dataset is defined directly inside the notebook and plot script. It contains one row per city and includes:

- Green transportation measures: electric vehicles, public EV chargers, cycling-route length, public bike companies, electric buses, and bus fare.
- Context variables: city population density, city area, airports, and an overall green transportation score.
- Target variable: air pollution index.

Several variables are normalized before analysis:

- `Cycling_Routes_km_per_area = Cycling_Routes_km / Area_km2`
- `EV_per_density = EV / Density`
- `Public_EV_Chargers_per_density = Public_EV_Chargers / Density`
- `Public_Bike_Companies_per_density = Public_Bike_Companies / Density`
- `Electric_Buses_per_density = Electric_Buses / Density`
- `Airports_per_density = Airports / Density`

## Methods

The notebook and plot script use:

- Shapiro-Wilk normality testing for air pollution.
- Spearman rank correlations between normalized transport metrics and air pollution.
- Mann-Whitney U tests for comparing high and low normalized EV groups.
- Kruskal-Wallis and pairwise Mann-Whitney tests for cycling-route tertiles.
- Bootstrap comparison for electric-bus density groups.
- Gradient boosting regression feature importance for normalized transport features.

## Generated Plots

The `plots/` folder contains:

- `air_pollution_by_city.png` - ranked city-level air pollution values.
- `overall_score_vs_air_pollution.png` - relationship between the overall green transportation score and air pollution.
- `spearman_correlations.png` - Spearman correlations between normalized transport indicators and air pollution.
- `normalized_metric_relationships.png` - scatter plots for key normalized metrics against air pollution.
- `cycling_tertile_air_pollution.png` - air pollution distributions by cycling-route density tertile.
- `gradient_boosting_feature_importance.png` - model-based feature importance from gradient boosting.

## Recreate The Plots

From the project root, run:

```powershell
py -3 scripts/generate_plots.py
```

Required Python packages:

```powershell
py -3 -m pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

## Notes

This is a small observational dataset, so the findings should be interpreted as associations rather than proof of causation. Results may also be sensitive to normalization choices and the limited number of cities.
