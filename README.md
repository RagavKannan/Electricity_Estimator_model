# Electricity Bill Estimator

Predicts a household's monthly electricity bill (`bill_amount_inr`) with a linear regression model trained on [electricity_data.csv](electricity_data.csv).

## How to run

```bash
pip install pandas scikit-learn
python electricity_estimator.py
```

## What the script does

1. Loads 100 household records from `electricity_data.csv`.
2. Uses three input features:
   - `units_kwh`: electricity consumed in the month (kWh)
   - `avg_temp_c`: average temperature for the month (°C)
   - `ac_hours_per_day`: average hours the AC runs per day
3. Splits the data 80/20 into training (80 rows) and test (20 rows) sets, with `random_state=42` so the split is the same on every run.
4. Fits a `LinearRegression` model and evaluates it on the test set.

## Results

```
Intercept: -590.89
Coefficients:
  units_kwh         7.709
  avg_temp_c       -0.640
  ac_hours_per_day -5.257
R²:   0.9979
RMSE: 79.07
```

The fitted model is:

```
bill ≈ -590.89 + 7.709 × units_kwh − 0.640 × avg_temp_c − 5.257 × ac_hours_per_day
```

### What each value means

| Value | Meaning |
|---|---|
| **Intercept = -590.89** | The predicted bill if every input were 0. A household can't actually use 0 kWh, so this number has no real-world meaning by itself. It comes out negative because the tariff is slab-based: each extra unit costs more as consumption rises. A straight line fitted to that curve has to start below zero so it can match the steeper cost at high usage. |
| **units_kwh = 7.709** | Each extra kWh adds about **₹7.71** to the bill, with temperature and AC hours held constant. This is the model's effective average price per unit, and it is the feature that drives the prediction. |
| **avg_temp_c = -0.640** | Each 1 °C rise changes the bill by about **−₹0.64**, which is negligible. Temperature only affects the bill through the units it makes you consume, and `units_kwh` already captures that. |
| **ac_hours_per_day = -5.257** | One more AC hour per day changes the bill by about **−₹5.26**, which is also small. The negative sign does **not** mean the AC saves money. AC use already shows up in `units_kwh` (correlation 0.79), so once units are known, AC hours add almost no new information. The small negative value is a side effect of overlapping features (multicollinearity), not a real effect. |
| **R² = 0.9979** | The model explains **99.8%** of the variation in bills on the test set. That is an excellent fit. |
| **RMSE = 79.07** | On average, predictions are off by about **₹79**. Bills in the data range from ₹443 to ₹6,248 (mean ₹2,094), so that is roughly a 4% error. |

### Key takeaways

- **The bill depends almost entirely on `units_kwh`.** In this data, units and bill have a correlation of 1.00, which is expected because the bill is calculated from units.
- **Temperature and AC hours act indirectly.** They raise consumption, and consumption raises the bill. They don't add to the bill on their own, which is why their coefficients are close to zero.
- **The near-perfect R² is partly by construction.** Using `units_kwh` as an input is close to giving the model the answer. To forecast a bill *before* you know the units, train on household features such as `num_acs`, `ac_hours_per_day`, `house_size_sqft`, `family_members`, `avg_temp_c` and `solar_panel` instead.
- **The test set is small (20 rows).** R² and RMSE will shift somewhat with a different `random_state`. Cross-validation would give a more reliable estimate.

### Example

For a household using 300 kWh at 28 °C with the AC on 2 hours a day:

```
-590.89 + 7.709×300 − 0.640×28 − 5.257×2 ≈ ₹1,693
```
