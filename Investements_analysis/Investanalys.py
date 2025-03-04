import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('Indian_Startup_Investment_Trends.csv')
print(df.head())

df = df.drop_duplicates()
df['Funding_Date'] = pd.to_datetime(df['Funding_Date'])

#Поиск дублирования
dupl = df.duplicated(subset=['Name', 'Pitch_Deck_Link']).value_counts()
df_cleaned = df.drop_duplicates(subset=['Name', 'Pitch_Deck_Link'], keep=False)

#Поиск пустых значений
missing_values = df.isna().sum()
df_filled = df.fillna({'Acquisition_Details ': 'Unspecified'})

#Линейное распределение созданий новых инвестпроектов
startups_per_year = df['Founded_Year'].value_counts().sort_index()
lineal = np.polyfit(startups_per_year .index, startups_per_year .values, 1) #Линейная линия тренда
nomial = np.polyfit(startups_per_year .index, startups_per_year .values, 2) #Биномиальная линия тренда
plineal = np.poly1d(lineal)
pnomial = np.poly1d(nomial)
plt.figure(figsize=(15, 10))
sns.lineplot(x=startups_per_year .index, y=startups_per_year .values, marker='o', color='green', label='Распределение образования организаций по годам')
plt.plot(startups_per_year .index, plineal(startups_per_year .index), "b-",label='Линейная линия тренда')
plt.plot(startups_per_year .index, pnomial(startups_per_year .index), "r-", label='Биномиальная линия тренда')
plt.title('Распределение образования организаций по годам', fontsize=16)
plt.xlabel('Год', fontsize=14)
plt.ylabel('Количество новых организаций', fontsize=14)
plt.grid(axis='both', linestyle='--', alpha=0.9,  which='both')
plt.legend()
plt.show()

most_popular_sector_per_year = df.groupby(['Founded_Year', 'Sector']).size().reset_index(name='Count')
print(most_popular_sector_per_year)

grouped = df.groupby(['Founded_Year', 'Sector']).size().reset_index(name='Count')
max_counts = grouped.groupby('Founded_Year')['Count'].transform(max)
most_popular_all = grouped[grouped['Count'] == max_counts]

# Тепловая диаграмма
fig, axis = plt.subplots(1, 2, figsize=(12, 5))
pivot_table = most_popular_sector_per_year.pivot(index='Founded_Year', columns='Sector', values='Count')
sns.heatmap(pivot_table, annot=True, cmap='RdYlGn', fmt='.0f', ax=axis[0])
axis[0].set_title('Сумма инвестиций по секторам и годам', fontsize=16)
axis[0].set_xlabel('Сектор', fontsize=14)
axis[0].set_ylabel('Год', fontsize=14)

# Столбчатая диаграмма
sns.barplot(x='Founded_Year', y='Count', hue='Sector', data=most_popular_all, palette='viridis', ax=axis[1])
axis[1].set_title('Самые популярные секторы по годам', fontsize=16)
axis[1].set_xlabel('Год', fontsize=14)
axis[1].set_ylabel('Количество фирм', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=90)
plt.legend(title='Сектор')
plt.tight_layout()  # Автоматическая настройка отступов
plt.show()

top_sectors = df.groupby('Sector')['Amount_Raised'].sum().nlargest(10)
amounts = df.groupby('Sector')['Amount_Raised'].sum()
sectors = df['Sector'].unique()
mean_amounts = df.groupby('Sector')['Amount_Raised'].mean().nlargest(10)

fig, axis2 = plt.subplots(1, 2, figsize=(15, 8))
sns.color_palette("dark:#5A9_r", as_cmap=True)
sns.barplot( x=top_sectors.index,y=top_sectors.values, palette="dark:#5A9_r", hue=top_sectors.index, ax=axis2[1])

plt.ticklabel_format(axis='y', style='plain')
axis2[1].set_yticklabels([f'{int(y/1e9)}' for y in axis2[1].get_yticks()])

axis2[1].set_title('Топ-10 секторов по объему инвестиций')
axis2[1].set_ylabel('Сумма инвестиций, млрд. долларов', fontsize=10)
axis2[1].set_xlabel('Сектор', fontsize=10)
axis2[1].set_xticklabels(axis2[1].get_xticklabels(), rotation=90)

sns.lineplot(x=top_sectors.index, y=mean_amounts, marker='o', color='green', ax=axis2[0])
axis2[0].set_title('Сумма инвестиций по секторам и годам', fontsize=16)
axis2[0].set_xlabel('Сектор', fontsize=10)
axis2[0].set_ylabel('Среднее количество инвестиций, тыс. долларов', fontsize=10)
axis2[0].set_yticklabels([f'{int(y/1e3)}' for y in axis2[0].get_yticks()])
axis2[0].set_xticklabels(axis2[0].get_xticklabels(), rotation=90)
plt.show()

#Анализ рентабильности по секторам
total_firms = df['Sector'].value_counts().reset_index()
total_firms.columns = ['Sector', 'Total Firms']
profitable_firms = df[df['Profitability'] == 'Yes']['Sector'].value_counts().reset_index()
profitable_firms.columns = ['Sector', 'Profitable Firms']
profitability_df = pd.merge(total_firms, profitable_firms, on='Sector')
profitability_df['Profitability Percentage'] = (profitability_df['Profitable Firms'] / profitability_df['Total Firms']) * 100
profitability_df = profitability_df.sort_values(by='Profitability Percentage', ascending=False)

sns.barplot(x='Sector', y='Profitability Percentage', data=profitability_df, palette='magma')
for index, row in profitability_df.iterrows():
    plt.text(index, row['Profitability Percentage'], f'{row["Profitability Percentage"]:.1f}%',
             ha='center', va='bottom', fontsize=12)
plt.title('Процент окупаемости по секторам', fontsize=16)
plt.xlabel('Сектор', fontsize=14)
plt.ylabel('Процент окупаемости', fontsize=14)
plt.ylim(0, 100)  # Ограничиваем ось Y до 100%
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

esg_by_sector = df.groupby(['Founded_Year'])['ESG_Score'].mean().reset_index()
esg_per_year = esg_by_sector.sort_values(by='Founded_Year')
coefficients0 = np.polyfit(esg_per_year['Founded_Year'], esg_per_year['ESG_Score'], 2)  # 2 — степень полинома
trend_line0 = np.polyval(coefficients0, esg_per_year['Founded_Year'])
fig, axis3 = plt.subplots(1, 2, figsize=(15, 8))

# Линейный график для каждого сектора
sns.lineplot(x='Founded_Year', y='ESG_Score', data=esg_per_year, marker='o', markersize=10, linewidth=2, ax=axis3[0])
for _, row in esg_per_year.iterrows():
    axis3[0].text(row['Founded_Year'], row['ESG_Score'], f'{row["ESG_Score"]:.1f}',
             ha='center', va='bottom', fontsize=10)

# Настройка графика
axis3[0].set_title('Средний индекс ESG по годам для каждого сектора', fontsize=12)
axis3[0].set_xlabel('Год', fontsize=14)
axis3[0].set_ylabel('Средний индекс ESG', fontsize=14)
axis3[0].grid(True, linestyle='--', alpha=0.7, color='gray')
axis3[0].plot(esg_per_year['Founded_Year'], trend_line0, color='red', label='Биномиальная линия тренда')
#plt.legend(title='Сектор')

growth_by_sector = df.groupby(['Founded_Year'])['Growth_Rate'].mean().reset_index()
growth_per_year = growth_by_sector.sort_values(by='Founded_Year')
coefficients = np.polyfit(growth_per_year['Founded_Year'], growth_per_year['Growth_Rate'], 2)  # 2 — степень полинома
trend_line = np.polyval(coefficients, growth_per_year['Founded_Year'])

# Линейный график для каждого сектора
sns.lineplot(x='Founded_Year', y='Growth_Rate', data=growth_per_year, marker='o', markersize=10, linewidth=2, ax=axis3[1], color='green')
for _, row in growth_per_year.iterrows():
    axis3[1].text(row['Founded_Year'], row['Growth_Rate'], f'{row["Growth_Rate"]:.1f}',
             ha='center', va='bottom', fontsize=10)
axis3[1].set_title('Средний индекс Growth Rate по годам для каждого сектора', fontsize=12)
axis3[1].set_xlabel('Год', fontsize=14)
axis3[1].set_ylabel('Средний индекс Роста', fontsize=14)
axis3[1].grid(True, linestyle='--', alpha=0.7, color='gray')
axis3[1].plot(growth_per_year['Founded_Year'], trend_line, color='red', label='Биномиальная линия тренда')
# Отображение графика
plt.tight_layout()
plt.show()



