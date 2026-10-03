# Freight Rate Prediction - ML Pipeline Report

This report outlines the methodology, data engineering, model selection, and validation steps taken to build a robust freight rate prediction model.

## 1. Data Engineering & Cleaning

**Validating Distance and Weights:**
First, we verified that the driving distance for all loads was greater than zero. Then we checked the weights and found instances of negative weights, which are physically impossible. We applied an absolute value transformation to correct these.

**Handling Missing Weights:**
We investigated the rows with missing weight values to see if there was any correlation or pattern to the missingness, but found none. We then analyzed the distribution of the weight data:
- **Mean:** 31,417.24
- **Median (50%):** 31,496.00

Because the mean and median are super close (indicating a very normal, symmetrical distribution), we safely imputed the missing weights using the median.

**Handling Missing Market Index:**
For the missing `market_index` values, we plotted the distribution and found it to be normal. We then grouped the data by date to see how many missing market indices occurred per day. The missingness was random and there were never more than 5 missing values on any given day. 

Because of this, we were able to calculate the average market index for that specific **Date + Pickup City** to fill the missing values. This ensured that our imputed mean was highly accurate and context-aware.

---

## 2. Advanced Feature Engineering

**Dates (Cyclical Encoding):**
We engineered the date variables into sine and cosine waves. This is crucial because standard numerical dates confuse machine learning models at the end of the year. By converting them cyclically, we make sure the model understands that December 28th is mathematically close to January 1st.

**Geography:**
We calculated directional movement (`lat_diff`, `lon_diff`) and the straight-line Euclidean distance (`geo_distance`). This helps the model understand the physical direction the truck is moving (since driving South vs North often pays differently) and provides a sense of route efficiency when compared to the actual driving distance.

**Interactions:**
We created mathematical interactions, most notably multiplying `distance` by `weight`. This creates a proxy for "Total Energy Expended." Moving 40,000 lbs for 1,000 miles is far more taxing than 10,000 lbs for 100 miles. By doing this math in advance, we give the model a massive shortcut to finding accurate pricing patterns.

**Categorical Encoding:**
Finally, categorical variables (pickup city, delivery city, equipment type) were transformed using Label Encoding so the machine learning algorithms could process them mathematically.

---

## 3. Machine Learning & Model Validation

We used a **Time-Based split** (training on the first 80% chronologically and testing on the remaining 20%) instead of a random split to prevent data leakage and accurately simulate predicting future rates.

We used `RandomizedSearchCV` to test multiple powerful algorithms (Random Forest, Extra Trees, XGBoost, LightGBM, Gradient Boosting). This allowed us to tune the hyperparameters (like `max_depth`) to prevent overfitting. The tuning process automatically selected the best model based on the lowest Mean Absolute Error (MAE), which was then used to generate the final predictions.
