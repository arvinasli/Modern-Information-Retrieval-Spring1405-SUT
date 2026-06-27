import math
from typing import List


class Evaluation:
    def __init__(self, name: str):
        self.name = name

    def _validate(self, actual: List[List[str]], predicted: List[List[str]]):
        if len(actual) != len(predicted):
            raise ValueError("actual and predicted must have the same length")

    def calculate_precision(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates macro precision.
        """
        self._validate(actual, predicted)
        total_precision = 0.0
        for act, pred in zip(actual, predicted):
            act_set = set(act)
            if len(pred) == 0:
                total_precision += 0.0
            else:
                relevant_retrieved = 0
                for doc in pred:
                    if doc in act_set:
                        relevant_retrieved += 1
                total_precision += relevant_retrieved / len(pred)
        return total_precision / len(actual) if actual else 0.0

    def calculate_recall(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates macro recall.
        """
        self._validate(actual, predicted)
        total_recall = 0.0
        for act, pred in zip(actual, predicted):
            act_set = set(act)
            if len(act) == 0:
                total_recall += 1.0
            elif len(pred) == 0:
                total_recall += 0.0
            else:
                relevant_retrieved = 0
                for doc in pred:
                    if doc in act_set:
                        relevant_retrieved += 1
                total_recall += relevant_retrieved / len(act)
        return total_recall / len(actual) if actual else 0.0

    def calculate_F1(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates F1 score.
        """
        p = self.calculate_precision(actual, predicted)
        r = self.calculate_recall(actual, predicted)

        if p + r == 0.0:
            return 0.0
        
        return (2 * p * r) / (p + r)

    def _average_precision_single(self, actual: List[str], predicted: List[str]) -> float:
        act_set = set(actual)
        if not act_set:
            return 0.0
        hits = 0
        sum_prec = 0.0
        for i, doc in enumerate(predicted, 1):
            if doc in act_set:
                hits += 1
                sum_prec += hits / i
        return sum_prec / len(act_set)

    def calculate_MAP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates MAP.
        """
        self._validate(actual, predicted)
        aps = []
        for act, pred in zip(actual, predicted):
            aps.append(self._average_precision_single(act, pred))
        return sum(aps) / len(aps) if aps else 0.0

    def _dcg_single(self, actual: List[str], predicted: List[str]) -> float:
        # r = 1 if in actual, 0 otherwise
        act_set = set(actual)
        dcg = 0.0
        for i, doc in enumerate(predicted, 1):
            rel = 1 if doc in act_set else 0
            dcg += rel / math.log2(i + 1)
        return dcg
    
    def _idcg_single(self, actual: List[str], predicted: List[str]) -> float:
        n_rel = len(actual)
        n_pred = len(predicted)
        idcg = 0.0
        for i in range(1, min(n_rel, n_pred) + 1):
            idcg += 1.0 / math.log2(i + 1)
        return idcg
    
    def calculate_DCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates mean DCG.
        """
        self._validate(actual, predicted)
        dcgs = []
        for act, pred in zip(actual, predicted):
            dcgs.append(self._dcg_single(act, pred))
        return sum(dcgs) / len(dcgs) if dcgs else 0.0

    def calculate_NDCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates mean NDCG.
        """
        self._validate(actual, predicted)
        ndcgs = []
        for act, pred in zip(actual, predicted):
            dcg = self._dcg_single(act, pred)
            idcg = self._idcg_single(act, pred)
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcgs.append(ndcg)
        return sum(ndcgs) / len(ndcgs) if ndcgs else 0.0

    def calculate_RR(self, actual: List[str], predicted: List[str]) -> float:
        """
        Calculate reciprocal rank.
        """
        act_set = set(actual)
        rr = 0.0
        for i, doc in enumerate(predicted, 1):
            if doc in act_set:
                rr = 1.0 / i
                break
        return rr
        
    def calculate_MRR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates MRR.
        """
        self._validate(actual, predicted)
        rrs = []
        for act, pred in zip(actual, predicted):
            rr = self.calculate_RR(act, pred)
            rrs.append(rr)
        return sum(rrs) / len(rrs) if rrs else 0.0
        
    def print_evaluation(self, precision, recall, f1, map, dcg, ndcg, mrr):
        """
        Prints the evaluation metrics.
        """
        print(f"name = {self.name}")
        print(f"Precision = {precision:.6f}")
        print(f"Recall = {recall:.6f}")
        print(f"F1 = {f1:.6f}")
        print(f"MAP = {map:.6f}")
        print(f"DCG = {dcg:.6f}")
        print(f"NDCG = {ndcg:.6f}")
        print(f"MRR = {mrr:.6f}")

    def log_evaluation(self, precision, recall, f1, map, dcg, ndcg, mrr):
        """
        Use Wandb to log the evaluation metrics.
        """
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'map': map,
                    'dcg': dcg,
                    'ndcg': ndcg,
                    'mrr': mrr,
                })
        except Exception:
            pass

    def calculate_evaluation(self, actual: List[List[str]], predicted: List[List[str]]):
        """
        Call all functions to calculate evaluation metrics.
        """
        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)
        f1 = self.calculate_F1(actual, predicted)
        map = self.calculate_MAP(actual, predicted)
        dcg = self.calculate_DCG(actual, predicted)
        ndcg = self.calculate_NDCG(actual, predicted)
        mrr = self.calculate_MRR(actual, predicted)
        self.print_evaluation(precision, recall, f1, map, dcg, ndcg, mrr)
        self.log_evaluation(precision, recall, f1, map, dcg, ndcg, mrr)
        return precision, recall, f1, map, dcg, ndcg, mrr