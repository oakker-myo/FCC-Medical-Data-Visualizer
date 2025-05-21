import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1 - Import the data from medical_examination.csv
df = pd.read_csv('medical_examination.csv')

# 2 - Add an overweight column to the data
# Calculate BMI: weight(kg) / (height(m))^2
# If BMI > 25, person is overweight (1), otherwise not overweight (0)
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)

# 3 - Normalize data by making 0 always good and 1 always bad
# For cholesterol and gluc: if value is 1, set to 0 (good), if > 1, set to 1 (bad)
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4 - Draw the Categorical Plot
def draw_cat_plot():
    # 5 - Create DataFrame for cat plot using pd.melt
    df_cat = pd.melt(df, 
                     id_vars=['cardio'], 
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6 - Group and reformat the data to split by cardio and show counts
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    
    # 7 - Create catplot
    fig = sns.catplot(data=df_cat, 
                      kind='bar',
                      x='variable', 
                      y='total', 
                      hue='value',
                      col='cardio')
    
    # 8 - Get the figure for output
    fig = fig.fig

    # 9 - Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig

# 10 - Draw the Heat Map
def draw_heat_map():
    # 11 - Clean the data by filtering out incorrect patient segments
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &  # diastolic <= systolic
        (df['height'] >= df['height'].quantile(0.025)) &  # height >= 2.5th percentile
        (df['height'] <= df['height'].quantile(0.975)) &  # height <= 97.5th percentile
        (df['weight'] >= df['weight'].quantile(0.025)) &  # weight >= 2.5th percentile
        (df['weight'] <= df['weight'].quantile(0.975))    # weight <= 97.5th percentile
    ]

    # 12 - Calculate the correlation matrix
    corr = df_heat.corr()

    # 13 - Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14 - Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 9))

    # 15 - Plot the correlation matrix using seaborn heatmap
    sns.heatmap(corr, 
                mask=mask, 
                annot=True, 
                fmt='.1f', 
                center=0,
                square=True, 
                linewidths=0.5, 
                cbar_kws={"shrink": 0.5},
                ax=ax)

    # 16 - Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
