import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS

 Input
 -----
 user_item_matrix: pd.DataFrame
     Матрица взаимодействий user-item
 """

    def __init__(self, data, weighting=True):
        """Constructor."""

        popularity = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        popularity.sort_values('quantity', ascending=False, inplace=True)
        popularity = popularity[popularity['item_id'] != 999999]
        popularity = popularity.groupby('user_id').head(5)
        popularity.sort_values(by=['user_id', 'quantity'], ascending=False, inplace=True)
        self.popularity = popularity

        self.user_item_matrix = MainRecommender.prepare_matrix(data)
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = MainRecommender.prepare_dicts(self.user_item_matrix)

        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)

    @staticmethod
    def prepare_matrix(data: pd.DataFrame):
        """Prepare user item matrix."""

        user_item_matrix = pd.pivot_table(data,
                                          index='user_id', columns='item_id',
                                          values='quantity',
                                          aggfunc='count',
                                          fill_value=0)

        user_item_matrix = user_item_matrix.astype(float)
        sparse_user_item = csr_matrix(user_item_matrix).tocsr()

        return user_item_matrix, sparse_user_item

    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    def get_recommendations(self, user, model, N=5):
        """Get recommendations."""

        res = [self.id_to_itemid[rec[0]] for rec in model.recommend(userid=self.userid_to_id[user],
                                                                    user_items=csr_matrix(self.user_item_matrix).tocsr(),
                                                                    N=N,
                                                                    filter_already_liked_items=False,
                                                                    filter_items=[self.itemid_to_id[999999]],
                                                                    recalculate_user=False)]

        return res

    def get_similar_item(self, model, item_id):
        """Recommend 1 most similar item."""

        recs = model.similar_items(self.itemid_to_id[item_id])
        top_rec = recs[1][0]

        return self.id_to_itemid[top_rec]

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())

        return own_recommender

    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)

        model.fit(csr_matrix(user_item_matrix).T.tocsr())

        return model

    def get_similar_items_recommendation(self):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        self.popularity['similar_recommendation'] = self.popularity['item_id'].apply(lambda x: self.get_similar_item(self.model, x))

        recommendation_similar_items = self.popularity.groupby('user_id')['similar_recommendation'].unique().reset_index()
        recommendation_similar_items.columns = ['user_id', 'similar_recommendation']

        return recommendation_similar_items

    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        result = []

        similar_users = self.model.similar_users(self.userid_to_id[user], N=N+1)
        similar_users = [rec[0] for rec in similar_users]
        similar_users = similar_users[1:]

        for user in similar_users:
            userid = self.id_to_userid[user]
            result.extend(self.get_recommendations(userid, model=self.own_recommender, N=1))

        return result
