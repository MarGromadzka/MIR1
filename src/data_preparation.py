import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_json_from_api(products, sessions, users):
    return pd.read_json(path_or_buf=products, lines=True), pd.read_json(path_or_buf=sessions, lines=True), pd.read_json(path_or_buf=users, lines=True)


def get_users_purchases(_sessions, _users):
    sessions = _sessions
    users = _users
    columns_to_drop = ['name', 'city', 'street', 'event_type', 'session_id', 'purchase_id']
    return pd.merge(sessions[sessions['event_type'] == 'BUY_PRODUCT'], users, on='user_id').drop(columns=columns_to_drop)


def get_users_spendings(_sessions, _products, _users):

    products = _products
    users_purchases = get_users_purchases(_sessions, _users)
    columns_to_drop = ['product_name', 'category_path']
    return pd.merge(users_purchases, products, on='product_id').drop(columns=columns_to_drop)


def get_spendings_in_months(_sessions=None, _products=None, _users=None):
    users_spendings = get_users_spendings(_sessions, _products, _users)
    spendings_in_months = []
    for i in range(1, 13):
        spendings_in_months.append(users_spendings[users_spendings['timestamp'].dt.month == i])
    return spendings_in_months


def group_by_users_price(_sessions=None, _products=None, _users=None):
    spendings_in_months = get_spendings_in_months(_sessions, _products, _users)
    udata = np.array([])
    users = _users
    unique_users = np.unique(users['user_id'])
    for user in unique_users:
        seq = np.array([month[month['user_id'] == user]['price'].sum() for month in spendings_in_months])
        if udata.size == 0:
            udata = seq
        else:
            udata = np.vstack([udata, seq])
    return udata


def group_by_users_discount(_sessions=None, _products=None, _users=None):
    spendings_in_months = get_spendings_in_months(_sessions, _products, _users)
    udata = np.array([])
    users = _users
    unique_users = np.unique(users['user_id'])
    for user in unique_users:
        seq = np.array([month[month['user_id'] == user]['offered_discount'].sum() for month in spendings_in_months])
        if udata.size == 0:
            udata = seq
        else:
            udata = np.vstack([udata, seq])
    return udata


def group_by_users_price_with_id(_sessions=None, _products=None, _users=None):
    if _products is None or _sessions is None or _users is None:
        spendings_in_months = get_spendings_in_months()
    else:
        spendings_in_months = get_spendings_in_months(_sessions, _products, _users)
    udata = np.array([])
    users = _users
    unique_users = np.unique(users['user_id'])
    for user in unique_users:
        seq = np.array([user])
        seq = np.append(seq, np.array([month[month['user_id'] == user]['price'].sum() for month in spendings_in_months]))
        if udata.size == 0:
            udata = seq
        else:
            udata = np.vstack([udata, seq])
    return udata


def get_top_users_in_month(month, no_top_users, _sessions=None, _products=None, _users=None):
    spendings_in_month = get_spendings_in_months(_sessions, _products, _users)[month - 1]
    users = _users
    unique_users = np.unique(users['user_id'])
    spendings = {}
    for user in unique_users:
        spendings[user] = spendings_in_month[spendings_in_month['user_id'] == user]['price'].sum()
    clients_ids = [client_id for client_id in sorted(spendings, key=spendings.get, reverse=True)]
    return clients_ids[:no_top_users]


def get_total_spendings(month, top_clients, _sessions=None, _products=None, _users=None):
    spendings = get_spendings_in_months(_sessions, _products, _users)[max(0, month - 1)]
    sum = 0
    for user in top_clients:
        sum += spendings[spendings['user_id'] == int(user)]['price'].sum()
    return round(sum, 2)


def get_spendings_plot(month, top_clients1, top_clients2, _sessions=None, _products=None, _users=None):
    no_clients = [x for x in range(1, 100, 5)]
    y1 = [get_total_spendings(month, top_clients1[:x], _sessions, _products, _users) for x in no_clients]
    y2 = [get_total_spendings(month, top_clients2[:x], _sessions, _products, _users) for x in no_clients]
    plt.title("Total spendings of top clients picked by algorithms")
    plt.plot(no_clients, y1, color='green', label="Reqnet")
    plt.plot(no_clients, y2, color='blue', label='Random')
    plt.legend(loc="upper left")
    plt.xlabel("Number of top clients")
    plt.ylabel("Total spendings")
    return plt


def list_to_string(s, delimiter):
    return ' '.join([str(int(elem)) for elem in s]).replace(" ", delimiter)
