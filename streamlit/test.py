import streamlit as st

st.title("Moja pierwsza aplikacja Streamlit")
st.write("Hej — działa! 🎉")

x = st.slider("Wybierz liczbę", 0, 100, 25)
st.write("Kwadrat wybranej liczby:", x * x)

if st.button("Kliknij mnie"):
    st.success("Dziękuję — przycisk działa!")