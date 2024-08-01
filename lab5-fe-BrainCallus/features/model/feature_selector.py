from typing import Dict, Tuple


class ModelResult:
    def __init__(self, classifier_name: str, metric_name: str, result: float, selected: bool):
        self.classifier_name = classifier_name
        self.classifier_name = metric_name
        self.result = result
        self.selected = selected


class FeatureSelectorDescr:
    def __init__(self, name: str, selected_indices: [int], selected_features: [str]):
        self.name = name
        self.selected_indices = selected_indices
        self.selected_features = selected_features
        self.classifier_results: Dict[str, Tuple[ModelResult]] = {}
