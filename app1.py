import streamlit as st
import joblib
import pandas as pd

# Load the model and scaler
model = joblib.load('too.pkl')
scaler = joblib.load('scalerr.pkl')

# Define required features
required_features = [
    'goal', 'backers', 'usd_pledged_real', 'usd_goal_real',
    'main_category_Comics', 'main_category_Crafts', 'main_category_Dance', 'main_category_Design',
    'main_category_Fashion', 'main_category_Film & Video', 'main_category_Food',
    'main_category_Games', 'main_category_Journalism', 'main_category_Music',
    'main_category_Photography', 'main_category_Publishing', 'main_category_Technology',
    'main_category_Theater', 'country_AU', 'country_BE', 'country_CA', 'country_CH',
    'country_DE', 'country_DK', 'country_ES', 'country_FR', 'country_GB', 'country_HK',
    'country_IE', 'country_IT', 'country_JP', 'country_LU', 'country_MX', 'country_NL',
    'country_NO', 'country_NZ', 'country_SE', 'country_SG', 'country_US'
]

# Page title
st.title("Прогноз успешности проекта")

# Instructions
st.write("Введите данные проекта ниже, чтобы предсказать его вероятность успеха.")

# Input fields for numeric features
goal = st.number_input("Введите сумму, которую проект хочет собрать.", min_value=0.0, step=100.0)
backers = st.number_input("Введите количество людей, которые поддержали проект", min_value=0, step=1)
usd_pledged_real = st.number_input("Введите общую сумму, которую проект уже собрал в долларах США.", min_value=0.0, step=100.0)
usd_goal_real = st.number_input("Введите реальную цель проекта в долларах США.", min_value=0.0, step=100.0)

# Dropdowns for categories and countries (one-hot encoded fields)
main_category = st.selectbox("Основная категория проекта", [
    '',
    "Комиксы", "Ремесло", "Танцы", "Дизайн", "Мода", "Кино и видео", "Еда",
    "Игры", "Журналистика", "Музыка", "Фотография", "Издательство", "Технологии", "Театр"
])

country = st.selectbox("Страна проекта", [
    '',
    "Австралия", "Бельгия", "Канада", "Швейцария", "Германия", "Дания", "Испания", "Франция", "Великобритания", "Гонконг", "Ирландия", "Италия", "Япония", "Люксембург",
    "Мексика", "Нидерланды", "Норвегия", "Новая Зеландия", "Швеция", "Сингапур", "США"
])


# Prepare input data for model prediction
input_data = {
    'goal': goal,
    'backers': backers,
    'usd_pledged_real': usd_pledged_real,
    'usd_goal_real': usd_goal_real,
}

# Add one-hot encoded features for main_category and country
for category in [f"main_category_{cat}" for cat in [
        "Comics", "Crafts", "Dance", "Design", "Fashion", "Film & Video", "Food",
        "Games", "Journalism", "Music", "Photography", "Publishing", "Technology", "Theater"
    ]]:
    input_data[category] = 1 if category == f"main_category_{main_category}" else 0

for cnt in [f"country_{ctry}" for ctry in [
        "AU", "BE", "CA", "CH", "DE", "DK", "ES", "FR", "GB", "HK", "IE", "IT", "JP",
        "LU", "MX", "NL", "NO", "NZ", "SE", "SG", "US"
    ]]:
    input_data[cnt] = 1 if cnt == f"country_{country}" else 0

# Convert input data to a DataFrame
input_df = pd.DataFrame([input_data])

# Scale data before prediction
input_data_scaled = scaler.transform(input_df)

# Predict and display results
if st.button("Предсказать успешность"):
    prediction = model.predict(input_data_scaled)
    probability = model.predict_proba(input_data_scaled)[0][1]  # Probability of success class

    if prediction[0] == 1:
        st.success(f"Модель предсказывает, что проект, скорее всего, будет успешным с уверенностью {probability:.2%}.")
    else:
        st.warning(f"Модель предсказывает, что проект, возможно, не будет успешным, с уверенностью {1 - probability:.2%}.")
