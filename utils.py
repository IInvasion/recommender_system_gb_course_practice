import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



def prefilter_items(data, take_n_popular=5000, item_features=None):
    # Уберем не интересные для рекоммендаций категории (department)
    if item_features is not None:
        department_size = pd.DataFrame(item_features. \
                                       groupby('department')['item_id'].nunique(). \
                                       sort_values(ascending=False)).reset_index()

        department_size.columns = ['department', 'n_items']
        rare_departments = department_size[department_size['n_items'] < 150].department.tolist()
        items_in_rare_departments = item_features[
            item_features['department'].isin(rare_departments)].item_id.unique().tolist()

        data = data[~data['item_id'].isin(items_in_rare_departments)]

    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
    data = data[data['price'] > 2]

    # Уберем слишком дорогие товарыs
    data = data[data['price'] < 50]

    # Возбмем топ по популярности
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

    top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()

    # Заведем фиктивный item_id (если юзер покупал товары из топ-5000, то он "купил" такой товар)
    data.loc[~data['item_id'].isin(top), 'item_id'] = 999999


    return data

def postfilter_items(user_id, recommednations):
    pass

def reduce_dims(df, dims=2, method='pca'):
    """Reduce dimensions number."""

    assert method in ['pca', 'tsne'], 'Wrong method chosen.'

    if method == 'pca':
        pca = PCA(n_components=dims)
        components = pca.fit_transform(df)
    elif method == 'tsne':
        tsne = TSNE(n_components=dims,
                    learning_rate=250,
                    random_state=42,
                    n_iter=300,
                    n_iter_without_progress=20)
        components = tsne.fit_transform(df)
    else:
        print('Error in method picking')

    colnames = {'component_' + str(i) for i in range(1, dims+1)}

    return pd.DataFrame(data=components, columns=colnames)

def display_components_in_2D_space(components_df, labels='category', marker='D'):
    """Display components."""

    groups = components_df.groupby(labels)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.margins(0.05)

    for name, group in groups:
        ax.plot(group.component_1, group.component_2,
                marker='o', ms=6,
                linestyle='',
                alpha=0.7,
                label=name)

    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

    plt.xlabel('component_1')
    plt.ylabel('component_2')
    plt.show()
