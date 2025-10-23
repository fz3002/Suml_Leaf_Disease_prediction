import streamlit as st

st.set_page_config(page_title="Lista zakupów", layout="centered")

st.title("🛒 Lista zakupów")
st.write("Dodaj produkty, usuń je lub wyczyść całą listę.")

# --- Inicjalizacja listy w stanie sesji --- #
if "shopping_list" not in st.session_state:
    st.session_state.shopping_list = []

# --- Formularz dodawania produktu --- #
with st.form("add_item_form", clear_on_submit=True):
    new_item = st.text_input("Dodaj produkt:")
    submitted = st.form_submit_button("Dodaj")
    if submitted and new_item.strip() != "":
        st.session_state.shopping_list.append(new_item.strip())

# --- Wyświetlanie listy zakupów --- #
if st.session_state.shopping_list:
    st.subheader("Twoja lista:")
    for i, item in enumerate(st.session_state.shopping_list):
        col1, col2 = st.columns([4, 1])
        col1.write(f"- {item}")
        if col2.button("Usuń", key=f"remove_{i}"):
            st.session_state.shopping_list.pop(i)
            st.experimental_rerun()  # odświeżenie po usunięciu
else:
    st.info("Lista jest pusta. Dodaj coś powyżej")

# --- Dodatkowe opcje --- #
if st.session_state.shopping_list:
    if st.button("Wyczyść całą listę"):
        st.session_state.shopping_list = []
        st.success("Lista została wyczyszczona.")
        st.experimental_rerun()
