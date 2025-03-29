import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from lazypredict.Supervised import LazyRegressor

# read data
data = pd.read_csv("./dataset/StudentScore.xls")

# profile = ProfileReport(data, title="Score Report", explorative=True)
# profile.to_file("./report/score.html")

target = "writing score"

# data split
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# data processing
print(data["gender"].unique())
print(data["race/ethnicity"].unique())
print(data["parental level of education"].unique())
print(data["lunch"].unique())
print(data["test preparation course"].unique())

num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(missing_values=-1, strategy="median")),
    ("scaler", StandardScaler())
])
# x_train[["math score", "reading score"]] = num_transformer.fit_transform(x_train[["math score", "reading score"]])

education_values = ['some high school', 'high school', 'some college', "associate's degree", "bachelor's degree", "master's degree" ]
gender_values = ["male", "female"]
lunch_values = x_train["lunch"].unique()
test_values = x_train["test preparation course"].unique()
ord_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=[education_values, gender_values, lunch_values, test_values]))
])
# x_train[["parental level of education", "gender", "lunch", "test preparation course"]] = ord_transformer.fit_transform(x_train[["parental level of education", "gender", "lunch", "test preparation course"]])

nom_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder())
])
# x_train[["race/ethnicity"]] = nom_transformer.fit_transform(x_train[["race/ethnicity"]])

preprocessor = ColumnTransformer(transformers=[
    ("num_feature", num_transformer, ["reading score", "math score"]),
    ("ord_feature", ord_transformer, ["parental level of education", "gender", "lunch", "test preparation course"]),
    ("nom_feature", nom_transformer, ["race/ethnicity"])
])

reg = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor())
])

regs = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = regs.fit(x_train, x_test, y_train, y_test)

params = {
    "preprocessor__num_feature__imputer__strategy": ["median", "mean"],
    "model__n_estimators": [100, 200, 300],
    "model__criterion": ["squared_error", "absolute_error", "friedman_mse", "poisson"],
    "model__max_depth": [None, 2, 5]
}
grid_search = GridSearchCV(estimator=reg, param_grid=params, cv=5, scoring="r2", verbose=2)
grid_search.fit(x_train, y_train)

print(grid_search.best_params_)

y_predict = grid_search.predict(x_test)

print("MAE: {}".format(mean_absolute_error(y_test, y_predict)))
print("MSE: {}".format(mean_squared_error(y_test, y_predict)))
print("R2: {}".format(r2_score(y_test, y_predict)))