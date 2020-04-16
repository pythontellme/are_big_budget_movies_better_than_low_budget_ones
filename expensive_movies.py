# Import pandas
import pandas as pd

# Read in dataset from csv files
df_bank = pd.read_csv("US_inflation.csv", skiprows=4 )
df_movies = pd.read_csv("tmdb_5000_movies.csv")

# Inspect movies dataset
print(df_movies.info())

# Assign columns we need to df_movies variable
df_movies = df_movies[['budget', 'release_date', 'title', 'vote_average']]

# Inspect movies dataframe
print(df_movies.info())

# Select only USA row and transpose
df_bank = df_bank.loc[df_bank['Country Code'] == 'USA']
df_bank = df_bank.transpose()
print(df_bank)

# Rename column
df_bank.rename(columns = {249:'cpi_index'}, inplace=True)

# Delete unwanted rows
df_bank = df_bank.iloc[4:-1,]
df_bank['year'] = df_bank.index

# Set index to range
number_of_rows = len(df_bank)
s = pd.Index(list(range(number_of_rows)))
df_bank = df_bank.set_index(s)

# Inspect dataframe
print(df_bank.head())
print(df_bank.tail())

# Convert month to type datetime
df_movies['release_date'] = pd.to_datetime(df_movies['release_date'])

# Create year column
df_movies['year'] = df_movies['release_date'].dt.to_period('Y')

# Align datatypes in year columns
df_bank['year']=df_bank['year'].astype(str)
df_movies['year']=df_movies['year'].astype(str)

# Merge 
merged_df = pd.merge(df_movies, df_bank, how='left')

# Inspect dataframe 
print(merged_df.info())

# Add inflation adjusted column
merged_df['adjusted_budget'] = \
    merged_df['budget']*merged_df['cpi_index']/100

# Drop na values in dataframe
merged_df = merged_df.dropna()

# Drop zero values in budget column
merged_df = merged_df.loc[merged_df['budget']!=0]
print(merged_df.info())

'''Visualization'''

# Import pyplot from matplotlib
from matplotlib import pyplot as plt

# Plot adj budget and vote average
plt.scatter(
    merged_df['adjusted_budget'],
    merged_df['vote_average'],
    c='black'
)

plt.xlabel("adjusted budget")
plt.ylabel("vote average")
plt.show()

# Split data into two groups
avg_adj_budg = merged_df['adjusted_budget'].mean()
print(f'The average adj budget for all movies is\
    ${avg_adj_budg:,.0f}\n')
less_than_ave = merged_df.loc[merged_df['adjusted_budget'] <= avg_adj_budg]
more_than_ave = merged_df.loc[merged_df['adjusted_budget'] > avg_adj_budg]

print('low budget')
print(less_than_ave['vote_average'].describe())
print('\nhigh budget')
print(more_than_ave['vote_average'].describe())


# Regression analysis
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

X = merged_df['adjusted_budget'].values.reshape(-1,1)
y = merged_df['vote_average'].values.reshape(-1,1)

reg = LinearRegression()
reg.fit(X, y)
print("The linear model is: Y = {:.5} + {:.5}X".format(reg.intercept_[0], reg.coef_[0][0]))

predictions = reg.predict(X)
plt.scatter(
    merged_df['adjusted_budget'],
    merged_df['vote_average'],
    c='black'
)
plt.plot(
    merged_df['adjusted_budget'],
    predictions,
    c='blue',
    linewidth=2
)

plt.xlabel("adjusted budget")
plt.ylabel("vote average")
plt.show()

# Change datatype of adj budget column
merged_df['adjusted_budget']=merged_df['adjusted_budget'].astype(float)

# Create and print regression results
X = merged_df['adjusted_budget']
y = merged_df['vote_average']
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())