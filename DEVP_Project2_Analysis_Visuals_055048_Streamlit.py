import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Data.
st.title('Imports and Exports Data Analysis')
st.write('This application analyzes import/export data.')

# uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
# if uploaded_file is not None:
import_export = pd.read_csv('Imports_Exports_Dataset.csv')


# Sample 3001 records
my_data = import_export.sample(n=3001, replace=False, random_state=55048)

st.subheader('Sample Data')
st.write(my_data.head())

# Kurtosis Calculation
st.subheader('Kurtosis for Value, Weight, and Quantity by Category')
category_kurtosis = my_data.groupby('Category').agg({
    'Value': lambda x: kurtosis(x, nan_policy='omit'),
    'Weight': lambda x: kurtosis(x, nan_policy='omit'),
    'Quantity': lambda x: kurtosis(x, nan_policy='omit')
}).reset_index()

st.write(category_kurtosis)

# Plot Kurtosis
category_kurtosis_melted = category_kurtosis.melt(id_vars="Category",
                                                  value_vars=["Value", "Weight", "Quantity"],
                                                  var_name="Metric", value_name="Kurtosis")

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Category', y='Kurtosis', hue='Metric', data=category_kurtosis_melted)
plt.title('Kurtosis for Value, Weight, and Quantity by Category')
plt.xticks(rotation=45)
st.pyplot(fig)

# Skewness Calculation
st.subheader('Skewness for Value, Weight, and Quantity by Category')
category_skew = my_data.groupby('Category').agg({
    'Value': lambda x: skew(x, nan_policy='omit'),
    'Weight': lambda x: skew(x, nan_policy='omit'),
    'Quantity': lambda x: skew(x, nan_policy='omit')
}).reset_index()

st.write(category_skew)

# Plot Skewness
category_skew_melted = category_skew.melt(id_vars="Category",
                                          value_vars=["Value", "Weight", "Quantity"],
                                          var_name="Metric", value_name="Skewness")

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Category', y='Skewness', hue='Metric', data=category_skew_melted)
plt.title('Skewness for Value, Weight, and Quantity by Category')
plt.xticks(rotation=45)
st.pyplot(fig)

# Top 10 Products by Value of Imports and Exports
st.subheader('Top 10 Products by Value of Imports and Exports')
product_trade = my_data.groupby(['Product', 'Import_Export']).agg({'Quantity': np.sum, 'Value': np.sum})
sorted_product_trade = product_trade.sort_values(by='Value', ascending=False).head(10).reset_index()

st.write(sorted_product_trade)

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=sorted_product_trade, x='Product', y='Value', hue='Import_Export')
plt.title('Top 10 Products by Value of Imports and Exports')
plt.xticks(rotation=45)
st.pyplot(fig)

# Most Preferred Shipping Method by Category
st.subheader('Most Preferred Shipping Method by Category')
shipping_preference = my_data.groupby(['Category', 'Shipping_Method']).size().reset_index(name='Count')
most_preferred_shipping = shipping_preference.loc[shipping_preference.groupby('Category')['Count'].idxmax()]

st.write(most_preferred_shipping)

# Correlation Matrix
st.subheader('Correlation Matrix of Numeric Features')
corr_matrix = my_data.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
plt.title('Correlation Matrix')
st.pyplot(fig)

# Non-Categorical Variables Scatter, Histogram, and Line Plots
st.subheader('Non-Categorical Variables Analysis')
non_cat = my_data[['Quantity', 'Value', 'Date', 'Port', 'Weight']]

# Scatter plot of Quantity vs Value
st.write("### Quantity vs Value")
fig, ax = plt.subplots()
sns.scatterplot(data=non_cat, x='Quantity', y='Value')
st.pyplot(fig)

# Histogram of Weight
st.write("### Weight Distribution")
fig, ax = plt.subplots()
sns.histplot(non_cat['Weight'], bins=30, kde=True)
st.pyplot(fig)

# Value over Time
st.write("### Value over Time")
non_cat['Date'] = pd.to_datetime(non_cat['Date'])
fig, ax = plt.subplots()
sns.lineplot(data=non_cat, x='Date', y='Value')
st.pyplot(fig)

# Bar plot of Port counts
st.write("### Top 10 Ports by Count")
port_counts = non_cat['Port'].value_counts().head(10)
fig, ax = plt.subplots()
sns.barplot(x=port_counts.index, y=port_counts.values)
plt.xticks(rotation=45)
st.pyplot(fig)

# Descriptive Statistics of Non-Categorical Variables
st.subheader('Descriptive Statistics of Non-Categorical Variables')
st.write(non_cat.describe())

# Box plot for Quantity, Value, and Weight
st.write("### Box Plots for Quantity, Value, and Weight")
fig, axs = plt.subplots(1, 3, figsize=(15, 6))

sns.boxplot(data=non_cat, y='Quantity', ax=axs[0])
sns.boxplot(data=non_cat, y='Value', ax=axs[1])
sns.boxplot(data=non_cat, y='Weight', ax=axs[2])

st.pyplot(fig)

# Confidence Interval Calculations
st.subheader('Confidence Interval for Quantity, Value, and Weight')
def confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    stderr = stats.sem(data)
    margin_of_error = stderr * stats.t.ppf((1 + confidence) / 2., n-1)
    return (mean - margin_of_error, mean + margin_of_error)

ci_quantity = confidence_interval(my_data['Quantity'])
ci_value = confidence_interval(my_data['Value'])
ci_weight = confidence_interval(my_data['Weight'])

st.write({
    'Confidence Interval for Quantity': ci_quantity,
    'Confidence Interval for Value': ci_value,
    'Confidence Interval for Weight': ci_weight
})

# Plot Confidence Intervals
means = [np.mean(my_data['Quantity']), np.mean(my_data['Value']), np.mean(my_data['Weight'])]
ci_lower = [ci_quantity[0], ci_value[0], ci_weight[0]]
ci_upper = [ci_quantity[1], ci_value[1], ci_weight[1]]
errors = [mean - lower for mean, lower in zip(means, ci_lower)]

st.write("### Confidence Interval Plot")
fig, ax = plt.subplots()
plt.bar(['Quantity', 'Value', 'Weight'], means, yerr=errors, capsize=10)
st.pyplot(fig)

# F-test and T-test
st.subheader('F-test and T-test Results')
f_stat, p_val_f = stats.levene(my_data['Quantity'], my_data['Weight'])
t_stat, p_val = stats.ttest_1samp(my_data['Weight'], 50)
st.write(f"F-test: {f_stat}, p-value: {p_val_f}")
st.write(f"T-test: {t_stat}, p-value: {p_val}")

# Categorical Variable Distribution
st.subheader('Categorical Variables Analysis')
catg = my_data[['Import_Export', 'Category', 'Shipping_Method', 'Payment_Terms', 'Country']]

st.write("### Distribution of Categories")
category_counts = catg['Category'].value_counts()
fig, ax = plt.subplots()
sns.barplot(x=category_counts.index, y=category_counts.values)
plt.xticks(rotation=45)
st.pyplot(fig)

# Export vs Import Distribution
st.write("### Import vs Export Distribution")
import_export_counts = catg['Import_Export'].value_counts()
fig, ax = plt.subplots()
plt.pie(import_export_counts, labels=import_export_counts.index, autopct='%1.1f%%', startangle=140)
st.pyplot(fig)

# Export and Import Values by Country
st.subheader('Export and Import Values by Country')
country_trade_values = my_data.groupby(['Country', 'Import_Export'])['Value'].sum().reset_index()

st.write(country_trade_values.head(10))

# Export and import values country wise
st.subheader("Total Export and Import Values Country-wise")
country_trade_values = my_data.groupby(['Country', 'Import_Export'])['Value'].sum().reset_index().head(10)
st.write(country_trade_values)

# Pivot the data for the line plot
pivot_trade_values = country_trade_values.pivot(index='Country', columns='Import_Export', values='Value').fillna(0)

# Plotting the line graph
st.subheader('Import and Export Values by Country')
fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(pivot_trade_values.index, pivot_trade_values['Export'], marker='o', label='Exports')
plt.plot(pivot_trade_values.index, pivot_trade_values['Import'], marker='o', label='Imports')
plt.title('Import and Export Values by Country')
plt.xticks(rotation=45)
plt.legend(title='Trade Type')
plt.grid(True)
plt.tight_layout()
st.pyplot(fig)

# 5 Countries with max total value of imports
st.subheader('Top 5 Countries by Total Import Value')
top_import_countries = my_data[my_data['Import_Export'] == 'Import'].groupby('Country')['Value'].sum().sort_values(ascending=False).head(5)
st.write(top_import_countries)

# Bar chart for top 5 import countries
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=top_import_countries.index, y=top_import_countries.values)
plt.title('Top 5 Countries with Maximum Import Values')
plt.xticks(rotation=45)
st.pyplot(fig)

# 5 Countries with max total value of exports
st.subheader('Top 5 Countries by Total Export Value')
top_export_countries = my_data[my_data['Import_Export'] == 'Export'].groupby('Country')['Value'].sum().sort_values(ascending=False).head(5)
st.write(top_export_countries)

# Pie chart for top 5 export countries
fig, ax = plt.subplots(figsize=(8, 8))
plt.pie(top_export_countries, labels=top_export_countries.index, autopct='%1.1f%%', startangle=140)
plt.title('Top 5 Countries with Maximum Export Values')
st.pyplot(fig)

# 5 Countries with max average import value per transaction
st.subheader('Top 5 Countries by Average Import Value per Transaction')
top_average_import_countries = my_data[my_data['Import_Export'] == 'Import'].groupby('Country')['Value'].mean().sort_values(ascending=False).head(5)
st.write(top_average_import_countries)

# 5 Countries with max average export value per transaction
st.subheader('Top 5 Countries by Average Export Value per Transaction')
top_average_export_countries = my_data[my_data['Import_Export'] == 'Export'].groupby('Country')['Value'].mean().sort_values(ascending=False).head(5)
st.write(top_average_export_countries)

# Bar chart for average export values
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=top_average_export_countries.index, y=top_average_export_countries.values)
plt.title('Top 5 Countries with Maximum Average Export Values per Transaction')
plt.xticks(rotation=45)
st.pyplot(fig)

# Country wise Shipment methods count
st.subheader('Country-wise Shipment Method Count')
country_shipment_count = my_data.groupby(['Country', 'Shipping_Method']).size().unstack(fill_value=0).head(10)

# Plot the heatmap
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(country_shipment_count, annot=True, fmt='d', cmap='Blues', linewidths=.5)
plt.title('Country-wise Shipment Methods Count')
st.pyplot(fig)

# Most preferred shipping type grouped by countries
st.subheader('Most Preferred Shipping Method by Country')
country_shipping_count = my_data.groupby(['Country', 'Shipping_Method']).size().reset_index(name='Count')
most_preferred_shipping = country_shipping_count.loc[country_shipping_count.groupby('Country')['Count'].idxmax()].head(15)
st.write(most_preferred_shipping)

# Bar chart for most preferred shipping methods
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=most_preferred_shipping, x='Country', y='Count', hue='Shipping_Method')
plt.title('Most Preferred Shipping Type by Countries')
plt.xticks(rotation=45)
st.pyplot(fig)

# Group by Supplier to calculate total value and count of transactions
st.subheader('Top 10 Suppliers by Total Transaction Value')
supplier_trade = my_data.groupby('Supplier').agg({'Value': [np.sum, 'count']})
sorted_supplier_trade = supplier_trade.sort_values(by=('Value', 'sum'), ascending=False).reset_index().head(10)
st.write(sorted_supplier_trade)

# Heatmap of top suppliers' transaction values
heatmap_data = sorted_supplier_trade.set_index('Supplier')[('Value', 'sum')].to_frame()
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='Blues', linewidths=.5)
plt.title('Top 10 Suppliers by Total Transaction Value')
st.pyplot(fig)
