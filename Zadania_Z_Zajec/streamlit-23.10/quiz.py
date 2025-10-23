import streamlit as st

st.title("Mini Quiz")

# Pytania i odpowiedzi
quiz = [
    {
        "question": "Jaki jest stolica Polski?",
        "options": ["Warszawa", "Kraków", "Gdańsk", "Wrocław"],
        "answer": "Warszawa"
    },
    {
        "question": "Ile nóg ma pająk?",
        "options": ["6", "8", "10", "12"],
        "answer": "8"
    },
    {
        "question": "Który język programowania jest używany w Streamlit?",
        "options": ["Java", "C++", "Python", "Ruby"],
        "answer": "Python"
    },
]

# Inicjalizacja odpowiedzi w stanie sesji
if "responses" not in st.session_state:
    st.session_state.responses = [None] * len(quiz)

# Formularz quizu
with st.form("quiz_form"):
    for i, q in enumerate(quiz):
        st.write(f"**{i+1}. {q['question']}**")
        st.session_state.responses[i] = st.radio(
            "Wybierz odpowiedź:",
            q["options"],
            key=f"q{i}",
            index=0 if st.session_state.responses[i] is None else q["options"].index(st.session_state.responses[i])
        )
    submit = st.form_submit_button("Sprawdź wynik")

# Po naciśnięciu "Sprawdź wynik"
if submit:
    score = 0
    for i, q in enumerate(quiz):
        if st.session_state.responses[i] == q["answer"]:
            score += 1

    st.write("---")
    st.success(f"Twój wynik: {score} / {len(quiz)}")

    # Opcjonalnie pokaż odpowiedzi poprawne
    st.write("Poprawne odpowiedzi:")
    for i, q in enumerate(quiz):
        st.write(f"{i+1}. {q['question']}: **{q['answer']}**")
