import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# завдання 1
np.random.seed(42)
days = 30
date_range = pd.date_range(start="2026-10-01", periods=days)

users = np.random.randint(500, 2000, size=days)
sessions = users + np.random.randint(100, 500, size=days)
revenue = users * np.random.uniform(5.0, 15.0, size=days)

df_metrics = pd.DataFrame(
    {"date": date_range, "users": users, "sessions": sessions, "revenue": revenue}
)

numeric_df = df_metrics[["users", "sessions", "revenue"]]
correlation_matrix = numeric_df.corr()
print("--- Кореляційна матриця ---")
print(correlation_matrix)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].scatter(df_metrics["users"], df_metrics["sessions"], color="blue")
axes[0].set_title("Users vs Sessions")
axes[0].set_xlabel("Users")
axes[0].set_ylabel("Sessions")

axes[1].scatter(df_metrics["users"], df_metrics["revenue"], color="green")
axes[1].set_title("Users vs Revenue")
axes[1].set_xlabel("Users")
axes[1].set_ylabel("Revenue")

axes[2].scatter(df_metrics["sessions"], df_metrics["revenue"], color="purple")
axes[2].set_title("Sessions vs Revenue")
axes[2].set_xlabel("Sessions")
axes[2].set_ylabel("Revenue")

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(
    df_metrics["date"],
    df_metrics["revenue"],
    marker="o",
    linestyle="-",
    color="crimson",
)
plt.title("Динаміка виручки (Revenue) за датами")
plt.xlabel("Дата")
plt.ylabel("Виручка")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# завдання 2

np.random.seed(123)
n_obs = 150

group_a = np.random.binomial(1, 0.12, size=n_obs)  # Конверсія ~12%
group_b = np.random.binomial(1, 0.18, size=n_obs)  # Конверсія ~18%

df_ab = pd.DataFrame(
    {
        "group": ["A"] * n_obs + ["B"] * n_obs,
        "converted": np.concatenate([group_a, group_b]),
    }
)

conv_a = df_ab[df_ab["group"] == "A"]["converted"].mean()
conv_b = df_ab[df_ab["group"] == "B"]["converted"].mean()

abs_diff = conv_b - conv_a
rel_diff = (abs_diff / conv_a) * 100

print(f"Конверсія Групи A: {conv_a:.4f}")
print(f"Конверсія Групи B: {conv_b:.4f}")
print(f"Абсолютна різниця: {abs_diff:.4f}")
print(f"Відносна зміна: {rel_diff:.2f}%")


def wilson_ci(successes, n, z=1.96):
    p = successes / n
    denominator = 1 + z**2 / n
    centre_adjusted_probability = p + z**2 / (2 * n)
    adjusted_standard_deviation = z * np.sqrt(
        (p * (1 - p) + z**2 / (4 * n)) / n
    )
    lower_bound = (
        centre_adjusted_probability - adjusted_standard_deviation
    ) / denominator
    upper_bound = (
        centre_adjusted_probability + adjusted_standard_deviation
    ) / denominator
    return p, p - lower_bound, upper_bound - p


succ_a = df_ab[df_ab["group"] == "A"]["converted"].sum()
succ_b = df_ab[df_ab["group"] == "B"]["converted"].sum()

p_a, err_a_low, err_a_high = wilson_ci(succ_a, n_obs)
p_b, err_b_low, err_b_high = wilson_ci(succ_b, n_obs)

groups = ["Group A", "Group B"]
conversions = [p_a, p_b]
errors = [[err_a_low, err_b_low], [err_a_high, err_b_high]]

plt.figure(figsize=(6, 5))
plt.bar(
    groups,
    conversions,
    yerr=errors,
    capsize=5,
    color=["skyblue", "lightgreen"],
    alpha=0.8,
)
plt.ylabel("Конверсія")
plt.title("Порівняння конверсій A/B груп з 95% довірчими інтервалами")
plt.ylim(0, 0.3)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# завдання 3


np.random.seed(42)
pop_size = 60000
population = np.random.exponential(scale=2.0, size=pop_size)

n1 = 30
n2 = 100
n_samples = 1000

sample_means_n1 = [
    np.mean(np.random.choice(population, size=n1, replace=True))
    for _ in range(n_samples)
]
sample_means_n2 = [
    np.mean(np.random.choice(population, size=n2, replace=True))
    for _ in range(n_samples)
]

print(
    f"n = {n1}: Середнє вибіркових середніх = {np.mean(sample_means_n1):.4f}, Стандартне відхилення = {np.std(sample_means_n1):.4f}"
)
print(
    f"n = {n2}: Середнє вибіркових середніх = {np.mean(sample_means_n2):.4f}, Стандартне відхилення = {np.std(sample_means_n2):.4f}"
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

axes[0].hist(sample_means_n1, bins=30, color="orange", edgecolor="black", alpha=0.7)
axes[0].set_title(f"Розподіл вибіркових середніх (n = {n1})")
axes[0].set_xlabel("Середнє значення")
axes[0].set_ylabel("Частота")

axes[1].hist(sample_means_n2, bins=30, color="teal", edgecolor="black", alpha=0.7)
axes[1].set_title(f"Розподіл вибіркових середніх (n = {n2})")
axes[1].set_xlabel("Середнє значення")

plt.tight_layout()
plt.show()

# завдання 4

np.random.seed(7)
days = 90
date_range = pd.date_range(start="2026-01-01", periods=days)

sales = np.random.poisson(lam=100, size=days) + np.sin(
    np.linspace(0, 3 * np.pi, days)
) * 20
df_sales = pd.DataFrame({"date": date_range, "sales": sales})

window_size = 7
df_sales["rolling_mean"] = (
    df_sales["sales"].rolling(window=window_size).mean()
)
df_sales["rolling_std"] = (
    df_sales["sales"].rolling(window=window_size).std()
)

print("--- Перші рядки з новими стовпчиками ---")
print(df_sales.head(10))

plt.figure(figsize=(12, 5))
plt.plot(
    df_sales["date"],
    df_sales["sales"],
    label="Вихідні продажі",
    color="lightgray",
    marker=".",
)
plt.plot(
    df_sales["date"],
    df_sales["rolling_mean"],
    label=f"Ковзне середнє ({window_size} днів)",
    color="blue",
    linewidth=2,
)
plt.title("Продажі та ковзне середнє")
plt.xlabel("Дата")
plt.ylabel("Значення")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 3.5))
plt.plot(
    df_sales["date"],
    df_sales["rolling_std"],
    label=f"Ковзне ст. відхилення ({window_size} днів)",
    color="purple",
    linewidth=2,
)
plt.title("Ковзне стандартне відхилення волатильності продажів")
plt.xlabel("Дата")
plt.ylabel("Std Dev")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()