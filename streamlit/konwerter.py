import streamlit as st

# --- Funkcje konwersji --- #
def convert_mass(value, from_unit, to_unit):
    factors = {
        "kilogram": 1.0,
        "gram": 0.001,
        "funt": 0.453592,
        "uncja": 0.0283495,
    }
    return value * factors[from_unit] / factors[to_unit]


def convert_length(value, from_unit, to_unit):
    factors = {
        "metr": 1.0,
        "kilometr": 1000.0,
        "mila": 1609.34,
        "jard": 0.9144,
        "stopa": 0.3048,
        "cal": 0.0254,
    }
    return value * factors[from_unit] / factors[to_unit]


def convert_temperature(value, from_unit, to_unit):
    # Najpierw konwertujemy do Celsjusza
    if from_unit == "Fahrenheit":
        value = (value - 32) * 5/9
    elif from_unit == "Kelwin":
        value = value - 273.15

    # Następnie z Celsjusza na docelową jednostkę
    if to_unit == "Fahrenheit":
        return value * 9/5 + 32
    elif to_unit == "Kelwin":
        return value + 273.15
    else:
        return value

# --- UI aplikacji --- #
st.title("🔄 Konwerter jednostek")
st.write("Wybierz kategorię jednostek i przelicz wartość.")

# Wybór kategorii
category = st.selectbox(
    "Wybierz kategorię:",
    ["Masa", "Długość", "Temperatura"]
)

# Lista jednostek dla każdej kategorii
units = {
    "Masa": ["kilogram", "gram", "funt", "uncja"],
    "Długość": ["metr", "kilometr", "mila", "jard", "stopa", "cal"],
    "Temperatura": ["Celsjusz", "Fahrenheit", "Kelwin"]
}

from_unit = st.selectbox("Z jednostki:", units[category])
to_unit = st.selectbox("Na jednostkę:", units[category])

value = st.number_input("Wartość do przeliczenia:", min_value=-1e9, max_value=1e9, value=1.0)

# --- Obliczenia --- #
if st.button("Przelicz"):
    if category == "Masa":
        result = convert_mass(value, from_unit, to_unit)
    elif category == "Długość":
        result = convert_length(value, from_unit, to_unit)
    else:
        result = convert_temperature(value, from_unit, to_unit)

    st.success(f"{value} {from_unit} = {result:.4f} {to_unit}")
