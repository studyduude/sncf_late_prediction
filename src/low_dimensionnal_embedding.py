import math
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from preprocessing import (
    TransformedDataset,
    fill_missing_coordinates,
    load_data,
    split_months_train_test,
)

# Fixing randomness to get reproducible results
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


class LowDimDataset:
    def __init__(self):
        self.data_before_2023, self.data_2023 = map(
            self.prepare_df, load_data(True, True)
        )

    def get_train_test_split(self) -> TransformedDataset:
        months_train, months_test = split_months_train_test(self.data_before_2023)
        data_train = self.data_before_2023[
            self.data_before_2023["date"].isin(months_train)
        ].copy()
        data_test = self.data_before_2023[
            self.data_before_2023["date"].isin(months_test)
        ].copy()
        data_2023 = self.data_2023.copy()
        return self.fit_transform(data_train, data_test, data_2023)

    @staticmethod
    def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
        percentages = [
            "prct_cause_externe",
            "prct_cause_infra",
            "prct_cause_gestion_trafic",
            "prct_cause_materiel_roulant",
            "prct_cause_gestion_gare",
            "prct_cause_prise_en_charge_voyageurs",
        ]
        to_keep = [
            "duree_moyenne",
            "nb_train_prevu",
            "Longitude_gare_depart",
            "Lattitude_gare_depart",
            "Longitude_gare_arrivee",
            "Lattitude_gare_arrivee",
            "distance",
            "retard_moyen_arrivee",
            *percentages,
            "gare_depart",
            "gare_arrivee",
            "date",
            "month",
        ]
        df = df[to_keep].copy()
        fill_missing_coordinates(df)
        df["month_sin"] = df.apply(
            lambda row: math.sin(math.pi * int(row["month"]) / 6), axis=1
        )
        df["month_cos"] = df.apply(
            lambda row: math.cos(math.pi * int(row["month"]) / 6), axis=1
        )
        df["link"] = df.apply(
            lambda row: (row["gare_depart"], row["gare_arrivee"]), axis=1
        )
        df[percentages] /= 100
        df.drop(columns=["month", "gare_depart", "gare_arrivee"], inplace=True)
        return df

    @staticmethod
    def compute_link_embeddings(
        data_train: pd.DataFrame,
    ) -> dict[tuple[str, str], np.array]:
        link_embeddings = defaultdict(list)
        for _, row in data_train.iterrows():
            link_embeddings[row["link"]].append(
                [
                    row["retard_moyen_arrivee"],
                    row["prct_cause_externe"],
                    row["prct_cause_infra"],
                    row["prct_cause_gestion_trafic"],
                    row["prct_cause_materiel_roulant"],
                    row["prct_cause_gestion_gare"],
                    row["prct_cause_prise_en_charge_voyageurs"],
                ]
            )
        link_embeddings = {
            key: np.mean(values, axis=0) for key, values in link_embeddings.items()
        }
        return link_embeddings

    @staticmethod
    def apply_link_embeddings(
        df: pd.DataFrame, link_embeddings: dict[tuple[str, str], np.array]
    ) -> None:
        new_features = [
            "lm_delay",
            "lm_external",
            "lm_infra",
            "lm_traffic",
            "lm_train",
            "lm_station",
            "lm_passenger",
        ]
        for i, feature_name in enumerate(new_features):
            df[feature_name] = df.apply(
                lambda row: link_embeddings[row["link"]][i],
                axis=1,
            )

    @staticmethod
    def get_data_transformers() -> tuple[ColumnTransformer, ColumnTransformer]:
        x_standard_scaled_features = [
            "duree_moyenne",
            "nb_train_prevu",
            "Longitude_gare_depart",
            "Lattitude_gare_depart",
            "Longitude_gare_arrivee",
            "Lattitude_gare_arrivee",
            "distance",
            "lm_delay",
        ]
        x_identity_features = [
            "lm_external",
            "lm_infra",
            "lm_traffic",
            "lm_train",
            "lm_station",
            "lm_passenger",
        ]
        y_standard_scaled_features = ["retard_moyen_arrivee"]
        y_identity_features = [
            "prct_cause_externe",
            "prct_cause_infra",
            "prct_cause_gestion_trafic",
            "prct_cause_materiel_roulant",
            "prct_cause_gestion_gare",
            "prct_cause_prise_en_charge_voyageurs",
        ]
        identity_transformer = FunctionTransformer(
            lambda x: x, validate=True, accept_sparse=True
        )
        x_transformer = ColumnTransformer(
            [
                ("standard_scaled", StandardScaler(), x_standard_scaled_features),
                ("identity", identity_transformer, x_identity_features),
            ]
        )
        y_transformer = ColumnTransformer(
            [
                ("standard_scaled", StandardScaler(), y_standard_scaled_features),
                ("identity", identity_transformer, y_identity_features),
            ]
        )
        return x_transformer, y_transformer

    def fit_transform(
        self, data_train: pd.DataFrame, data_test: pd.DataFrame, data_2023: pd.DataFrame
    ) -> TransformedDataset:
        link_embeddings = self.compute_link_embeddings(data_train)
        self.apply_link_embeddings(data_train, link_embeddings)
        self.apply_link_embeddings(data_test, link_embeddings)
        self.apply_link_embeddings(data_2023, link_embeddings)
        x_transformer, y_transformer = self.get_data_transformers()
        x_train = x_transformer.fit_transform(data_train)
        y_train = y_transformer.fit_transform(data_train)
        x_test = x_transformer.transform(data_test)
        y_test = y_transformer.transform(data_test)
        x_2023 = x_transformer.transform(data_2023)
        y_2023 = y_transformer.transform(data_2023)
        return x_train, y_train, x_test, y_test, x_2023, y_2023


if __name__ == "__main__":
    LowDimDataset().get_train_test_split()
