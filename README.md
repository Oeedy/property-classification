# Property Classification Framework for Tokenised UK Freehold Properties  

Python code and outputs for MSc dissertation: *Designing a Classification Framework for Tokenised UK Freehold Properties: Supporting Fractional Ownership and Enhancing Market Transparency*  

---

## Contents  
- Property Classification Model.py – full Python script for data cleaning, model construction, validation, and visualisation.  
- Figures/ – output charts used in the dissertation (tier distributions, boxplots, maps, correlation heatmaps).  
- classified_property_dataset.csv – processed dataset produced by the model.  

## Data Requirements  
The model relies on the following publicly available datasets:  
- UK Land Registry Price Paid Data (2024 sample)  
- ONS Postcode Directory (ONSPD, 2021)  
- Indices of Multiple Deprivation (IMD, 2019)  
- LSOA shapefiles for England and Wales (2011 boundaries)  

Due to licensing restrictions, raw datasets are not included in this repository. Users should obtain the data directly from the relevant public sources.  

## Running the Code  
1. Install Python 3.10 or above with the following libraries: pandas, matplotlib, geopandas, seaborn.  
2. Place the input datasets in the same working directory as the script.  
3. Run Property Classification Model.py to reproduce results.  
