import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# 1. Data load करो
data = pd.read_csv("car data.csv")

# 2. Column names clean करो
data.columns = data.columns.str.strip().str.replace(" ", "_")

# 3. Columns check (debug के लिए)
print("Columns in dataset:")
print(data.columns)

# 4. Input (X) और Output (y)
X = data[['Present_Price', 'Driven_kms', 'Owner']]
y = data['Selling_Price']

# 5. Train-Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Model बनाओ और train करो
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Prediction
y_pred = model.predict(X_test)

# 8. Error check
error = mean_absolute_error(y_test, y_pred)
print("Average Price Error:", error)

# 9. Graph दिखाओ
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Car Price Prediction")
plt.show()
