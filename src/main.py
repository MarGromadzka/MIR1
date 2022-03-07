import streamlit as st
from data_preparation import load_json_from_api, list_to_string, get_total_spendings, get_spendings_plot
from model import get_data, choose_best_users, choose_true_best_users, check_accuracy, choose_random_users



n_clients = st.slider("Pick number of best clients", min_value=1, max_value=200, value=52)
file1 = st.file_uploader("Pick sessions.json")
file2 = st.file_uploader("Pick products.json")
file3 = st.file_uploader("Pick users.json")

if file1 is not None and file2 is not None and file3 is not None:
    ses, pro, usr = load_json_from_api(file1, file2, file3)
    data = get_data(ses, pro, usr)
    preds_random = choose_random_users(n_clients, data)
    preds_reqnet = choose_best_users(n_clients, data)
    best = choose_true_best_users(n_clients, data)
    check_accuracy(preds_reqnet, best)
    st.title("ReqNet")
    st.markdown("Top clients:")
    st.markdown(list_to_string(preds_reqnet, delimiter = ", "))
    st.write(f"Accuracy {check_accuracy(preds_reqnet, best)}")
    st.write(f"Total spendings {get_total_spendings(12, preds_reqnet, ses, pro, usr)}")
    st.download_button('Download ReqNet.csv', list_to_string(preds_reqnet, delimiter=",\n"), 'ReqNet.csv')
    st.title("Random")
    st.markdown("Top clients:")
    st.markdown(list_to_string(preds_random, delimiter=", "))
    st.write(f"Accuracy {check_accuracy(preds_random, best)}")
    st.write(f"Total spendings {get_total_spendings(12, preds_random, ses, pro, usr)}")
    st.title("Comparison")
    st.write("")
    st.pyplot(get_spendings_plot(12, choose_best_users(100, data), choose_random_users(100, data), ses, pro, usr))
