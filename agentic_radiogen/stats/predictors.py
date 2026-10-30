"""Train/eval backends for radiomics → gene-mutation AUPRC / F1.

Tabular radiomics features are nonlinearly related to mutation labels; the
default path uses tree/kernel models. Deep options (LSTM / VGG19 / ResNeXt /
SE-CNN) adapt those architectures to 1-D feature vectors and require PyTorch.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC

MODEL_CHOICES: tuple[str, ...] = (
    "lstm",
    "gat",
    "svm",
    "random_forest",
    "xgboost",
    "vgg19",
    "resnext",
    "se_cnn",
    "ridge_reg",
    "elastic_net",
    "poly_reg",
)

_ALIASES: dict[str, str] = {
    "lstm": "lstm",
    "gat": "gat",
    "svm": "svm",
    "random_forest": "random_forest",
    "randomforest": "random_forest",
    "rf": "random_forest",
    "xgboost": "xgboost",
    "xgb": "xgboost",
    "vgg19": "vgg19",
    "vgg": "vgg19",
    "resnext": "resnext",
    "se_cnn": "se_cnn",
    "secnn": "se_cnn",
    "ridge_reg": "ridge_reg",
    "ridge": "ridge_reg",
    "ridgeregression": "ridge_reg",
    "elastic_net": "elastic_net",
    "elasticnet": "elastic_net",
    "poly_reg": "poly_reg",
    "poly": "poly_reg",
    "polynomial": "poly_reg",
    "polynomialregression": "poly_reg",
}

_DL_MODELS = frozenset({"lstm", "gat", "vgg19", "resnext", "se_cnn"})
_REG_MODELS = frozenset({"ridge_reg", "elastic_net", "poly_reg"})


def normalize_model_name(name: str | None) -> str:
    raw = (name or "random_forest").strip().lower()
    raw = raw.replace(" ", "_").replace("-", "_")
    compact = raw.replace("_", "")
    resolved = _ALIASES.get(raw) or _ALIASES.get(compact)
    if resolved is None:
        raise ValueError(
            f"Unknown --model {name!r}. Choose one of: {', '.join(MODEL_CHOICES)}"
        )
    return resolved


def predict_proba_mutation(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    """Fit on train and return class=1 scores on test."""
    _, test_scores = fit_score_mutation(model_name, X_train, y_train, X_test)
    return test_scores


def fit_score_mutation(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit on train; return (train_scores, test_scores) for class=1."""
    name = normalize_model_name(model_name)
    if name in _DL_MODELS:
        return _predict_torch(name, X_train, y_train, X_test)
    est = _sklearn_estimator(name)
    if name in _REG_MODELS:
        est.fit(X_train, y_train.astype(float))
        train_scores = np.asarray(est.predict(X_train), dtype=float)
        test_scores = np.asarray(est.predict(X_test), dtype=float)
        # Map regression outputs to (0, 1) for a stable ranking/proba score scale.
        return (
            1.0 / (1.0 + np.exp(-train_scores)),
            1.0 / (1.0 + np.exp(-test_scores)),
        )
    est.fit(X_train, y_train.astype(int))
    return (
        np.asarray(est.predict_proba(X_train)[:, 1], dtype=float),
        np.asarray(est.predict_proba(X_test)[:, 1], dtype=float),
    )


def _sklearn_estimator(name: str):
    if name == "svm":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        class_weight="balanced",
                        random_state=0,
                    ),
                ),
            ]
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            random_state=0,
            n_jobs=-1,
        )
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier  # type: ignore
        except ImportError:
            return Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        HistGradientBoostingClassifier(
                            max_depth=6,
                            learning_rate=0.08,
                            max_iter=200,
                            random_state=0,
                        ),
                    ),
                ]
            )
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=200,
                        max_depth=4,
                        learning_rate=0.08,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=0,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if name == "ridge_reg":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("reg", Ridge(alpha=1.0, random_state=0)),
            ]
        )
    if name == "elastic_net":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "reg",
                    ElasticNet(
                        alpha=0.05,
                        l1_ratio=0.5,
                        max_iter=5000,
                        random_state=0,
                    ),
                ),
            ]
        )
    if name == "poly_reg":
        # Degree-2 polynomial features + Ridge (regularized polynomial regression).
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "poly",
                    PolynomialFeatures(degree=2, include_bias=False),
                ),
                ("reg", Ridge(alpha=1.0, random_state=0)),
            ]
        )
    raise ValueError(f"No sklearn backend for model {name!r}")


def _predict_torch(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError(
            f"--model {name} requires PyTorch. Install torch or choose "
            "svm / random_forest / xgboost."
        ) from exc

    device = torch.device("cpu")
    n_features = int(X_train.shape[1])
    feature_neighbors = _feature_neighbors(X_train) if name == "gat" else None
    model = _build_torch_model(
        name, n_features, nn, feature_neighbors=feature_neighbors
    ).to(device)

    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    Xtr = (X_train - mu) / sigma
    Xte = (X_test - mu) / sigma

    xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32, device=device)
    xv = torch.tensor(Xte, dtype=torch.float32, device=device)

    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    epochs = 40 if len(y_train) >= 12 else 25
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        train_probs = torch.sigmoid(model(xt)).cpu().numpy().reshape(-1)
        test_probs = torch.sigmoid(model(xv)).cpu().numpy().reshape(-1)
    return (
        np.asarray(train_probs, dtype=float),
        np.asarray(test_probs, dtype=float),
    )


def _feature_neighbors(X_train: np.ndarray, max_neighbors: int = 8) -> np.ndarray:
    """Build a sparse feature graph using training-set correlations only."""
    n_features = int(X_train.shape[1])
    if n_features <= 1:
        return np.zeros((n_features, 1), dtype=np.int64)

    centered = np.asarray(X_train, dtype=float) - np.mean(X_train, axis=0)
    norms = np.linalg.norm(centered, axis=0)
    normalized = centered / np.where(norms > 1e-8, norms, 1.0)
    similarity = np.abs(normalized.T @ normalized)
    np.fill_diagonal(similarity, -np.inf)

    n_neighbors = min(max_neighbors, n_features - 1)
    neighbors = np.argpartition(
        similarity, kth=n_features - n_neighbors, axis=1
    )[:, -n_neighbors:]
    self_indices = np.arange(n_features, dtype=np.int64)[:, None]
    return np.concatenate((self_indices, neighbors), axis=1)


def _build_torch_model(
    name: str,
    n_features: int,
    nn,
    *,
    feature_neighbors: np.ndarray | None = None,
):
    import torch

    if name == "lstm":

        class LSTMHead(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # 4 stacked LSTM levels (was single-layer / num_layers=1).
                self.lstm = nn.LSTM(
                    input_size=1,
                    hidden_size=32,
                    num_layers=4,
                    batch_first=True,
                    dropout=0.1,
                )
                self.attention = nn.Linear(32, 1)
                self.fc = nn.Linear(32, 1)

            def forward(self, x):
                sequence, _ = self.lstm(x.unsqueeze(-1))
                weights = torch.softmax(self.attention(sequence).squeeze(-1), dim=1)
                context = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
                return self.fc(context)

        return LSTMHead()

    if name == "gat":
        if feature_neighbors is None:
            raise ValueError("GAT requires a feature-neighbor graph")

        class FeatureGATLayer(nn.Module):
            def __init__(
                self,
                in_features: int,
                out_features: int,
                heads: int,
                neighbors: np.ndarray,
                *,
                concatenate: bool,
            ) -> None:
                super().__init__()
                self.out_features = out_features
                self.heads = heads
                self.concatenate = concatenate
                self.projection = nn.Linear(
                    in_features, heads * out_features, bias=False
                )
                self.attention_source = nn.Parameter(
                    torch.empty(heads, out_features)
                )
                self.attention_target = nn.Parameter(
                    torch.empty(heads, out_features)
                )
                nn.init.xavier_uniform_(self.attention_source)
                nn.init.xavier_uniform_(self.attention_target)
                self.register_buffer(
                    "neighbors", torch.as_tensor(neighbors, dtype=torch.long)
                )

            def forward(self, nodes):
                batch_size = nodes.shape[0]
                projected = self.projection(nodes).view(
                    batch_size, n_features, self.heads, self.out_features
                )
                source_scores = (
                    projected * self.attention_source[None, None, :, :]
                ).sum(dim=-1)
                target_scores = (
                    projected * self.attention_target[None, None, :, :]
                ).sum(dim=-1)
                neighbor_scores = target_scores[:, self.neighbors, :]
                logits = torch.nn.functional.leaky_relu(
                    source_scores.unsqueeze(2) + neighbor_scores,
                    negative_slope=0.2,
                )
                weights = torch.softmax(logits, dim=2)
                weights = torch.nn.functional.dropout(
                    weights, p=0.1, training=self.training
                )
                neighbor_nodes = projected[:, self.neighbors, :, :]
                attended = (weights.unsqueeze(-1) * neighbor_nodes).sum(dim=2)
                if self.concatenate:
                    return torch.nn.functional.elu(attended.flatten(start_dim=2))
                return torch.nn.functional.elu(attended.mean(dim=2))

        class FeatureGAT(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.feature_embedding = nn.Parameter(torch.empty(n_features, 16))
                nn.init.xavier_uniform_(self.feature_embedding)
                self.value_projection = nn.Linear(1, 16)
                self.gat1 = FeatureGATLayer(
                    16, 16, 4, feature_neighbors, concatenate=True
                )
                self.gat2 = FeatureGATLayer(
                    64, 32, 1, feature_neighbors, concatenate=False
                )
                self.classifier = nn.Sequential(
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(32, 1),
                )

            def forward(self, values):
                nodes = self.value_projection(values.unsqueeze(-1))
                nodes = torch.relu(nodes + self.feature_embedding.unsqueeze(0))
                nodes = self.gat1(nodes)
                nodes = self.gat2(nodes)
                pooled = torch.cat(
                    (nodes.mean(dim=1), nodes.amax(dim=1)), dim=1
                )
                return self.classifier(pooled)

        return FeatureGAT()

    if name == "vgg19":

        class VGG19Head(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                channels = [1, 16, 32, 64, 64]
                layers: list = []
                for cin, cout in zip(channels, channels[1:]):
                    layers.extend(
                        [
                            nn.Conv1d(cin, cout, kernel_size=3, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv1d(cout, cout, kernel_size=3, padding=1),
                            nn.ReLU(inplace=True),
                            nn.MaxPool1d(2, ceil_mode=True),
                        ]
                    )
                self.features = nn.Sequential(*layers)
                self.head = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(channels[-1], 32),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, 1),
                )

            def forward(self, x):
                return self.head(self.features(x.unsqueeze(1)))

        return VGG19Head()

    if name == "resnext":

        class ResNeXtBlock(nn.Module):
            def __init__(self, channels: int, cardinality: int = 4) -> None:
                super().__init__()
                self.conv = nn.Conv1d(
                    channels, channels, kernel_size=3, padding=1, groups=cardinality
                )
                self.bn = nn.BatchNorm1d(channels)
                self.act = nn.ReLU(inplace=True)

            def forward(self, x):
                return self.act(self.bn(self.conv(x)) + x)

        class ResNeXtHead(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Conv1d(1, 32, kernel_size=3, padding=1),
                    nn.BatchNorm1d(32),
                    nn.ReLU(inplace=True),
                )
                self.block1 = ResNeXtBlock(32)
                self.block2 = ResNeXtBlock(32)
                self.head = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(32, 1),
                )

            def forward(self, x):
                x = self.stem(x.unsqueeze(1))
                x = self.block1(x)
                x = self.block2(x)
                return self.head(x)

        return ResNeXtHead()

    if name == "se_cnn":

        class SEBlock(nn.Module):
            def __init__(self, channels: int, reduction: int = 4) -> None:
                super().__init__()
                hidden = max(1, channels // reduction)
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.fc = nn.Sequential(
                    nn.Linear(channels, hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden, channels),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                w = self.fc(self.pool(x).squeeze(-1)).unsqueeze(-1)
                return x * w

        class SECNNHead(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv1d(1, 32, kernel_size=5, padding=2),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(32, 32, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )
                self.se = SEBlock(32)
                self.head = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(32, 1),
                )

            def forward(self, x):
                x = self.se(self.conv(x.unsqueeze(1)))
                return self.head(x)

        return SECNNHead()

    raise ValueError(f"No torch backend for model {name!r} (n_features={n_features})")
