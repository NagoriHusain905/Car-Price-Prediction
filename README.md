# Car-Price-Prediction
🚗 A machine learning project using a Random Forest Regressor to predict used car prices. Includes end-to-end data processing, one-hot encoding, and performance evaluation with MSE/RMSE metrics.
<br>
# Car Price Prediction using Random Forest

## 📌 Project Overview
This project aims to predict the market price of used cars based on various features such as mileage, horsepower, and model year. By leveraging machine learning, we can provide accurate price estimates that help buyers and sellers make informed decisions.

## 📊 Dataset
The dataset contains 2,000 records of vehicle data, including:
- **Numerical Features:** Model Year, Engine Size, Mileage, Horsepower, etc.
- **Categorical Features:** Brand, Fuel Type, and Transmission.

## 🛠️ Technical Workflow

### 1. Data Cleaning
- Removed the `Car_ID` column as it is a unique identifier with no predictive power.
- Checked for missing values to ensure data integrity.

### 2. Feature Engineering
- Applied **One-Hot Encoding** to categorical variables (`Brand`, `Fuel_Type`, `Transmission`).
- Used `drop_first=True` to avoid the "Dummy Variable Trap" and ensure mathematical stability for the model.

### 3. Model Selection
- **Algorithm:** Random Forest Regressor.
- **Why Random Forest?** It handles non-linear relationships and outliers effectively by averaging the results of 100 individual decision trees.

### 4. Evaluation Metrics
I chose **Mean Squared Error (MSE)** and **Root Mean Squared Error (RMSE)** to evaluate the model because they penalize large prediction errors more heavily than simple averages.
- **MAE:** (Insert your result here)
- **RMSE:** (Insert your result here)

## 📈 Visualizations
### Correlation Heatmap
Used to identify which features have the strongest relationship with the car's price.
![Correlation Heatmap](Screenshot%20from%202026-04-21%2012-31-43.png)

### Actual vs. Predicted Prices
This plot demonstrates the model's accuracy by comparing predicted values against real market prices.
![Actual vs Predicted](Screenshot%20from%202026-04-21%2012-45-07.png)

## 💡 Key Insights
- **Horsepower** and **Model Year** were the strongest predictors of a higher price.
- **Mileage** showed a significant negative correlation, as expected.

## 🚀 How to Run
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the Jupyter Notebook `Car_Price_Prediction.ipynb`.
