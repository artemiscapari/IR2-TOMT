from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator
from . import SentenceEvaluator
from typing import Dict
from ..util import cos_sim, dot_score
import csv
import torch
from torch import topk, where, cuda


class BM25IREvaluator(SentenceEvaluator):
    def __init__(
            self,
            query_texts: Dict[str, str],
            bm25_doc_texts : list,
            top_bm25_doc_idxs : list,
            use_bi_encoder : bool,
            use_cross_encoder : bool,
            rerank_with_ce: bool = False,
            output_path: str = None,
            batch_size: int = 100,
            top_ks : list = [1, 10, 50],
            cross_encoder= None):

        self.batch_size = batch_size
        self.bm25_doc_texts = bm25_doc_texts
        self.top_ks = top_ks
        self.score_functions = [cos_sim, dot_score]
        self.query_texts = query_texts
        self.output_path = output_path
        self.num_queries = len(self.query_texts)
        self.score_names = ['dot', 'cos']
        self.use_bi_encoder = use_bi_encoder
        self.use_cross_encoder = use_cross_encoder
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        self.top_bm25_doc_idxs = top_bm25_doc_idxs.to(self.device)
        self.cross_encoder= cross_encoder
        self.rerank_with_ce = rerank_with_ce


    def _compute_metrics(self, sims):
        metrics = {}

        for k in self.top_ks:
            _, top_k_indeces = topk(sims, k)
            top_bm25_doc_idxs = self.top_bm25_doc_idxs.to(self.device)
            check_indeces = top_bm25_doc_idxs == top_k_indeces

            if k == 10:
                mrr = ((1. / (where(check_indeces)[1] + 1)).sum() / self.num_queries).item()
                metrics["MRR@10"] = mrr

            recall = ( (check_indeces).sum() / self.top_bm25_doc_idxs.shape[0] ).item()
            metrics["Recall@" + str(k)] = recall

        return metrics


    def compute_metrics(self, model):

        if self.use_bi_encoder:
            metrics = {}
            query_embeddings = model.encode(self.query_texts, show_progress_bar=False, batch_size=self.batch_size, convert_to_tensor=True).detach()
            doc_embeddings = model.encode(self.bm25_doc_texts, show_progress_bar=False, batch_size=self.batch_size, convert_to_tensor=True).detach()
            for i, score_func in enumerate(self.score_functions):
                sims = score_func(query_embeddings, doc_embeddings).detach()

                if self.rerank_with_ce:
                    if not self.cross_encoder:
                        print('No cross encoder was found, please use one of the cross encoders found on https://huggingface.co/cross-encoder')
                        break

                    _, top_k_indeces = topk(sims, 100)

                    combined = []

                    for qidx in range(top_k_indeces.shape[0]):
                        for docidx in top_k_indeces[qidx]:
                            combined.append([self.query_texts[qidx], self.bm25_doc_texts[docidx]])
                    new_sims = self.cross_encoder.predict(combined, convert_to_tensor=True).reshape(self.num_queries, top_k_indeces.shape[1]).detach()

                    top_k_frame = torch.ones((self.num_queries, len(self.bm25_doc_texts))).to(self.device)*-99.
                    _, top_k_indeces = topk(new_sims, 100)
                    for q in range(self.num_queries):
    
                        top_k_frame[q][top_k_indeces[q]] = new_sims[q, top_k_indeces[q]].detach()

                    metrics[ self.score_names[i] ] = self._compute_metrics(top_k_frame)
                else:
                    metrics[ self.score_names[i] ] = self._compute_metrics(sims)

        elif self.use_cross_encoder:
            # Change format of the input texts so that every query is compared to every document.
            combined = [[q, d] for q in self.query_texts for d in self.bm25_doc_texts ]
            # Compute the similarities with the cross-encoder for every query-doc pair.
            sims = model.predict(combined, convert_to_tensor=True, batch_size=1000).reshape(self.num_queries, len(self.bm25_doc_texts)).detach()
            metrics = self._compute_metrics(sims)

        return metrics


    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        metrics = self.compute_metrics(model)
        line = [epoch, steps]

        if self.use_bi_encoder:
            for score in self.score_names:
                recall_scores = [ metrics[score]['Recall@' + str(k)] for k in self.top_ks ]
                line += [ metrics[score]['MRR@10'] ] + recall_scores

        elif self.use_cross_encoder:
            recall_scores = [ metrics['Recall@' + str(k)] for k in self.top_ks ]
            line += [ metrics['MRR@10'] ] + recall_scores

        with open(output_path + "/metrics.csv", 'a') as f:
            writer = csv.writer(f)
            writer.writerow(line)

        if self.use_bi_encoder:
            return metrics['cos']['MRR@10']

        elif self.use_cross_encoder:
            return metrics['MRR@10']


