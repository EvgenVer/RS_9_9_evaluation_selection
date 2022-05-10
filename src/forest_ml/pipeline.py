from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def create_pipeline(
    use_scalar: bool, model_type: str, random_state: int, params: dict
) -> Pipeline:
    pipeline_steps = []
    if use_scalar:
        pipeline_steps.append(("scaler", StandardScaler()))
    if model_type == 'logreg':
        pipeline_steps.append(
            ('classifire', LogisticRegression(**params, random_state=random_state))
        )
    elif model_type == 'forest':
        pipeline_steps.append(
            ('classifire', RandomForestClassifier(**params, random_state=random_state))
        )
    return Pipeline(steps=pipeline_steps)
