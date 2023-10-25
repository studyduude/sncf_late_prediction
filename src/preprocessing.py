import math
import os
import random
from typing import TypeAlias

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Fixing randomness to get reproducible results
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


TransformedDataset: TypeAlias = tuple[
    csr_matrix, csr_matrix, csr_matrix, csr_matrix, csr_matrix, csr_matrix
]


def haversine(lat1, lon1, lat2, lon2):
    # Radius of the Earth in km
    R = 6371.0

    # Convert latitude and longitude from degrees to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # Compute the differences between latitudes and longitudes
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Haversine formula
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Calculate the distance
    distance = R * c
    if math.isnan(distance):
        print(lat1, lon1, lat2, lon2)
    return distance


def split_months_train_test(
    data_before_2023: pd.DataFrame,
) -> tuple[set[str], set[str]]:
    months_train = set(data_before_2023["date"].unique())
    months_test = set()
    for month in range(1, 13):
        year = random.randrange(2018, 2023)
        date = f"{year}-{month:02d}"
        assert date == number_to_date(date_to_number(date))
        months_train.remove(date)
        months_test.add(date)
    return months_train, months_test


def split_explicative_target(
    data: pd.DataFrame, add_month: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    explicative_columns = [
        "date",
        "service",
        "gare_depart",
        "gare_arrivee",
        "duree_moyenne",
        "nb_train_prevu",
        "Longitude_gare_depart",
        "Lattitude_gare_depart",
        "Longitude_gare_arrivee",
        "Lattitude_gare_arrivee",
        "distance",
    ]
    target_columns = [
        "retard_moyen_arrivee",
        "prct_cause_externe",
        "prct_cause_infra",
        "prct_cause_gestion_trafic",
        "prct_cause_materiel_roulant",
        "prct_cause_gestion_gare",
        "prct_cause_prise_en_charge_voyageurs",
    ]
    if add_month:
        explicative_columns.append("month")
    explicative_variables = data[explicative_columns]
    target_variables = data[target_columns]
    return explicative_variables, target_variables


def to_month_id(year: int, month: int) -> int:
    return (year - 1) * 12 + month - 1


def from_month_id(month_id: int) -> tuple[int, int]:
    q, r = divmod(month_id, 12)
    year = q + 1
    month = r + 1
    return year, month


@np.vectorize
def date_to_number(date: str) -> float:
    """
    Affine transform
    2018-01 -> -1.0
    2022-12 ->  1.0
    """
    year, month = map(int, date.split("-"))
    january_18 = to_month_id(2018, 1)
    december_22 = to_month_id(2022, 12)
    month_id = to_month_id(year, month)
    return (2 * month_id - january_18 - december_22) / (december_22 - january_18)


@np.vectorize
def number_to_date(x: float) -> str:
    """
    Affine transform
    -1.0 -> 2018-01
    1.0 -> 2022-12
    """
    january_18 = to_month_id(2018, 1)
    december_22 = to_month_id(2022, 12)
    month_id = round((january_18 * (1 - x) + december_22 * (1 + x)) / 2)
    year, month = from_month_id(month_id)
    return f"{year}-{month:02d}"


def get_data_transformers(
    use_month: bool = True, return_feature_categories: bool = False
) -> dict:
    x_categorical_features = ["service", "gare_depart", "gare_arrivee"]
    if use_month:
        x_categorical_features.append("month")
    x_standard_scaled_features = [
        "duree_moyenne",
        "nb_train_prevu",
        "Longitude_gare_depart",
        "Lattitude_gare_depart",
        "Longitude_gare_arrivee",
        "Lattitude_gare_arrivee",
        "distance",
    ]
    y_standard_scaled_features = ["retard_moyen_arrivee"]
    y_percentage_features = [
        "prct_cause_externe",
        "prct_cause_infra",
        "prct_cause_gestion_trafic",
        "prct_cause_materiel_roulant",
        "prct_cause_gestion_gare",
        "prct_cause_prise_en_charge_voyageurs",
    ]
    date_encoder = FunctionTransformer(
        date_to_number, number_to_date, check_inverse=False
    )
    percentage_scaler = FunctionTransformer(
        lambda x: x / 100, lambda x: 100 * x, validate=True, accept_sparse=True
    )
    x_transformer = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(), x_categorical_features),
            ("standard_scaled", StandardScaler(), x_standard_scaled_features),
            ("date", date_encoder, ["date"]),
        ]
    )
    y_transformer = ColumnTransformer(
        [
            ("standard_scaled", StandardScaler(), y_standard_scaled_features),
            ("percentages", percentage_scaler, y_percentage_features),
        ]
    )

    if return_feature_categories:
        return {
            "x_transformer": x_transformer,
            "y_transformer": y_transformer,
            "x_categorical_features": x_categorical_features,
            "x_standard_scaled_features": x_standard_scaled_features,
            "y_standard_scaled_features": y_standard_scaled_features,
            "y_percentage_features": y_percentage_features,
        }

    return {"x_transformer": x_transformer, "y_transformer": y_transformer}


def embedding(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
    x_2023: pd.DataFrame,
    y_2023: pd.DataFrame,
    use_month: bool = True,
    *,
    return_transformers_and_feature_names: bool = False,
) -> (
    TransformedDataset
    | tuple[TransformedDataset, tuple[ColumnTransformer, ColumnTransformer], np.array]
):
    data = get_data_transformers(use_month, return_transformers_and_feature_names)
    x_transformer, y_transformer = data["x_transformer"], data["y_transformer"]
    x_train = x_transformer.fit_transform(x_train)
    y_train = y_transformer.fit_transform(y_train)
    x_test = x_transformer.transform(x_test)
    y_test = y_transformer.transform(y_test)
    x_2023 = x_transformer.transform(x_2023)
    y_2023 = y_transformer.transform(y_2023)
    transformed_dataset = x_train, y_train, x_test, y_test, x_2023, y_2023

    if return_transformers_and_feature_names:
        x_categorical_features = data["x_categorical_features"]
        x_standard_scaled_features = data["x_standard_scaled_features"]
        one_hot_feature_names = x_transformer.named_transformers_[
            "categorical"
        ].get_feature_names_out(x_categorical_features)
        numeric_feature_names = x_standard_scaled_features
        date_feature_name = ["date"]
        all_feature_names = np.concatenate(
            [numeric_feature_names, date_feature_name, one_hot_feature_names]
        )
        return transformed_dataset, (x_transformer, y_transformer), all_feature_names
    return transformed_dataset


def fill_missing_coordinates(data: pd.DataFrame) -> None:
    city_coordinates = {
        "BARCELONA": (41.3851, 2.1734),
        "FRANCFORT": (50.1109, 8.6821),
        "GENEVE": (46.2044, 6.1432),
        "ITALIE": (45.0703, 7.6869),
        "LAUSANNE": (46.5197, 6.6323),
        "MADRID": (40.4168, -3.7038),
        "STUTTGART": (48.7758, 9.1829),
        "VALENCE TGV RHÔNES-ALPES SUD": (44.9334, 4.8922),
        "ZURICH": (47.3769, 8.5417),
    }
    for index, row in data.iterrows():
        if row["gare_depart"] in city_coordinates:
            (
                data.at[index, "Lattitude_gare_depart"],
                data.at[index, "Longitude_gare_depart"],
            ) = city_coordinates[row["gare_depart"]]
        if row["gare_arrivee"] in city_coordinates:
            (
                data.at[index, "Lattitude_gare_arrivee"],
                data.at[index, "Longitude_gare_arrivee"],
            ) = city_coordinates[row["gare_arrivee"]]


def load_data(
    add_month: bool = False, split_2023: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    filename = "regularite-mensuelle-tgv-ext-data.csv"
    filepath = f"data/{filename}"
    if not os.path.exists(filepath):
        filepath = "../" + filepath
        if not os.path.exists(filepath):
            raise FileNotFoundError(f'Could not find "{filename}".')
    data = pd.read_csv(filepath, delimiter=",")
    fill_missing_coordinates(data)
    data["distance"] = data.apply(
        lambda row: haversine(
            row["Lattitude_gare_depart"],
            row["Longitude_gare_depart"],
            row["Lattitude_gare_arrivee"],
            row["Longitude_gare_arrivee"],
        ),
        axis=1,
    )
    if add_month:
        data["month"] = data.apply(lambda row: row["date"][-2:], axis=1)
    if split_2023:
        data_before_2023 = data[data["date"] < "2023-01"]
        data_2023 = data[data["date"] >= "2023-01"]
        return data_before_2023, data_2023
    return data


def load_and_split_train_test(
    one_hot_month: bool = True,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    data_before_2023, data_2023 = load_data(one_hot_month, True)
    months_train, months_test = split_months_train_test(data_before_2023)
    data_train = data_before_2023[data_before_2023["date"].isin(months_train)]
    data_test = data_before_2023[data_before_2023["date"].isin(months_test)]
    x_train, y_train = split_explicative_target(data_train, one_hot_month)
    x_test, y_test = split_explicative_target(data_test, one_hot_month)
    x_2023, y_2023 = split_explicative_target(data_2023, one_hot_month)
    return x_train, y_train, x_test, y_test, x_2023, y_2023


def load_and_process(
    one_hot_month: bool = True, *, return_transformers_and_feature_names: bool = False
) -> (
    TransformedDataset
    | tuple[TransformedDataset, tuple[ColumnTransformer, ColumnTransformer]]
):
    x_train, y_train, x_test, y_test, x_2023, y_2023 = load_and_split_train_test(
        one_hot_month
    )
    return embedding(
        x_train,
        y_train,
        x_test,
        y_test,
        x_2023,
        y_2023,
        use_month=one_hot_month,
        return_transformers_and_feature_names=return_transformers_and_feature_names,
    )
