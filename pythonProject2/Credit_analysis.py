import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
payments= pd.read_csv('payments.csv')
orders = pd.read_csv('orders.csv')
plan = pd.read_csv('plan.csv')

df = pd.merge(payments, orders, on='order_id', how='outer')
df = pd.merge(df, plan, on='order_id', how='outer')
df.to_csv('out.csv', index=False)
# Предобработка данных
# Анализ пустых значений в таблице
missing_values = df.isnull().sum()  # Количество пропущенных значений в каждом столбце
print(missing_values)

# Анализ наличия дубликатов в стобце order_id
duplicate_count = df.duplicated('order_id').sum()
print(f"Количество дубликатов в столбце 'order_id': {duplicate_count}")

# Приведение дат к формату datetime
df['created_at'] = pd.to_datetime(df['created_at'])
df['put_at'] = pd.to_datetime(df['put_at'])
df['closed_at'] = pd.to_datetime(df['closed_at'])
df['plan_at'] = pd.to_datetime(df['plan_at'])
df['paid_at'] = pd.to_datetime(df['paid_at'])


df['days_delay'] = (df['paid_at'] - df['plan_at']).dt.days
df['is_delayed'] = df['days_delay'] > 0


total_delayed = df['is_delayed'].sum()
print(f"Общее количество просроченных платежей: {total_delayed}")


df['month'] = df['plan_at'].dt.to_period('M')
delayed_by_month = df.groupby('month')['is_delayed'].sum().reset_index()
delayed_by_month['month'] = delayed_by_month['month'].dt.to_timestamp()


plt.figure(figsize=(12, 6))
sns.lineplot(x='month', y='is_delayed', data=delayed_by_month, marker='o')
plt.title('Динамика просроченных платежей по месяцам')
plt.xlabel('Месяц')
plt.ylabel('Количество просрочек')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


delayed_amount = df[df['is_delayed']]['plan_sum_total'].sum()
print(f"Общая сумма просроченных платежей: {delayed_amount}")


avg_delayed_amount = df[df['is_delayed']]['plan_sum_total'].mean()
print(f"Средняя сумма просроченного платежа: {avg_delayed_amount}")


plt.figure(figsize=(12, 6))
sns.histplot(df[df['days_delay'] > 0]['days_delay'], bins=30, kde=True)
plt.title('Распределение дней просрочки')
plt.xlabel('Дни просрочки')
plt.ylabel('Количество платежей')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()