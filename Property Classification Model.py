import pandas as pd
import matplotlib.pyplot as plt

#%%
#Load and clean ONSPD (Postcode to LSOA)
onspd = pd.read_csv("PCD_OA_LSOA_MSOA_LAD_AUG21_UK_LU.csv", 
                    usecols=['pcds', 'lsoa11cd'], 
                    encoding='latin1')
onspd['pcds'] = onspd['pcds'].str.upper().str.replace(' ', '', regex=False)
onspd = onspd.drop_duplicates()
print("ONSPD shape after cleaning:", onspd.shape)

#%%
#Load and clean IMD 2019
imd = pd.read_csv("IMD_2019.csv", encoding='ISO-8859-1')
imd = imd.rename(columns={'ï»¿lsoa11cd': 'lsoa11cd'})  # Fix BOM issue
imd = imd[['lsoa11cd', 'IMD_Rank', 'IMD_Decile']]
print("IMD shape after cleaning:", imd.shape)

#%%
#Load and clean Land Registry Price Paid 2024
pp = pd.read_csv("pp-2024.csv", header=None)

# Correct column mapping
pp = pp.iloc[:, [1, 2, 3, 4, 5]]  
pp.columns = ['Price', 'Date', 'Postcode', 'PropertyType', 'Tenure']

# Clean postcodes
pp['Postcode'] = pp['Postcode'].str.upper().str.replace(' ', '', regex=False)
print("Price Paid shape after cleaning:", pp.shape)

#%%
#Merge Price Paid + ONSPD
merged = pp.merge(onspd, how='left', left_on='Postcode', right_on='pcds')
print("After merging ONSPD:", merged.shape)
print(f"Missing LSOA after ONSPD merge: {merged['lsoa11cd'].isna().mean():.2%}")

#%%
#Merge result + IMD
final_df = merged.merge(imd, how='left', on='lsoa11cd')
print("After merging IMD:", final_df.shape)
print(f"Missing IMD Decile after merge: {final_df['IMD_Decile'].isna().mean():.2%}")

# Fill missing IMD_Decile with neutral value for modelling
final_df['IMD_Decile'] = final_df['IMD_Decile'].fillna(5)

#%%
#Apply classification model (weights: Location=40%, PropertyType=20%, Valuation=30%, Volatility=10%)

#Location Score (40 points from IMD Decile)
final_df['LocationScore'] = final_df['IMD_Decile'] * 4

# Property Type Score mapping
property_type_map = {
    'D': 20,  # Detached
    'S': 15,  # Semi-detached
    'T': 10,  # Terraced
    'F': 5    # Flats
}
final_df['PropertyTypeScore'] = final_df['PropertyType'].map(property_type_map).fillna(10)

#Valuation Score (30 points using price quantiles)
final_df['ValuationScore'] = pd.qcut(final_df['Price'], q=5, labels=[6, 12, 18, 24, 30]).astype(int)

#Volatility Score placeholder (10 points)
final_df['VolatilityScore'] = 10

#Composite Score
final_df['CompositeScore'] = (
    final_df['LocationScore'] +
    final_df['PropertyTypeScore'] +
    final_df['ValuationScore'] +
    final_df['VolatilityScore']
)

#Tier classification (A–G with duplicates handled)
tier_labels = ['G', 'F', 'E', 'D', 'C', 'B', 'A']
final_df['Tier'] = pd.qcut(final_df['CompositeScore'], q=7, labels=tier_labels, duplicates='drop')

print(final_df['Tier'].value_counts(dropna=False))

#%%
#Visualise Tier Distribution
tier_counts = final_df['Tier'].value_counts().sort_index()
plt.figure(figsize=(8,5))
tier_counts.plot(kind='bar')
plt.title("Distribution of Properties by Tier")
plt.xlabel("Tier")
plt.ylabel("Number of Properties")
plt.show()

#%%
#Save final dataset
final_df.to_csv("classified_property_dataset.csv", index=False)
print("Classified dataset saved as classified_property_dataset.csv")
#%%


# Visualisations



#Summary statistics per Tier
summary = final_df.groupby("Tier").agg(
    Property_Count=("Price", "count"),
    Avg_Price=("Price", "mean"),
    Avg_IMD_Decile=("IMD_Decile", "mean"),
    Avg_CompositeScore=("CompositeScore", "mean")
).reset_index()

print(summary)
#%%
plt.figure(figsize=(10, 6))
final_df.boxplot(column='Price', by='Tier', grid=False)
plt.title("Property Price Distribution by Tier")
plt.suptitle('')
plt.xlabel("Tier")
plt.ylabel("Price (£)")
plt.show()
#%%
plt.figure(figsize=(10, 6))
final_df.boxplot(column='IMD_Decile', by='Tier', grid=False)
plt.title("IMD Decile Distribution by Tier")
plt.suptitle('')
plt.xlabel("Tier")
plt.ylabel("IMD Decile (1 = Most Deprived, 10 = Least)")
plt.show()
#%%
# Map of Tiers by Region
import geopandas as gpd

#Load LSOA shapefile
lsoa_shapes = gpd.read_file("LSOA_2011_EW_BGC_V3.shp")

#Map Tier letters to numeric and convert to float
tier_map = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}
final_df['Tier_Num'] = final_df['Tier'].map(tier_map).astype(float)

#Calculate average Tier per LSOA
tier_avg = final_df.groupby('lsoa11cd', as_index=False)['Tier_Num'].mean()

#Merge Tier data with LSOA shapefile
lsoa_shapes = lsoa_shapes.merge(tier_avg, left_on='LSOA11CD', right_on='lsoa11cd', how='left')

#Plot the Tier map
plt.figure(figsize=(12, 12))
lsoa_shapes.plot(
    column='Tier_Num',
    cmap='coolwarm',
    legend=True,
    missing_kwds={"color": "lightgrey", "label": "No Data"}
)
plt.title("Average Property Tier by LSOA", fontsize=15)
plt.axis('off')
plt.show()
#%%
#Filter shapefile to London LSOAs
#The shapefile has an attribute for Local Authority Name (LSOA11NM contains the name)
london_shapes = lsoa_shapes[lsoa_shapes['LSOA11NM'].str.contains("London", case=False, na=False)]

#Plot London Tier map
plt.figure(figsize=(10, 10))
london_shapes.plot(
    column='Tier_Num',
    cmap='coolwarm',
    legend=True,
    missing_kwds={"color": "lightgrey", "label": "No Data"}
)
plt.title("Average Property Tier - Greater London", fontsize=14)
plt.axis('off')
plt.show()
#%%


## Correlation Analysis: Tier, Price, IMD

import seaborn as sns

#Ensure Tier_Num is numeric
tier_map = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}
final_df['Tier_Num'] = final_df['Tier'].map(tier_map).astype(float)

#Select numeric columns for correlation
corr_df = final_df[['Tier_Num', 'Price', 'IMD_Decile', 'CompositeScore']]

#Calculate correlation matrix
corr_matrix = corr_df.corr()

print("Correlation Matrix:")
print(corr_matrix)

#Visualise correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
plt.title("Correlation between Tier, Price, IMD Decile, and Composite Score", fontsize=14)
plt.show()


