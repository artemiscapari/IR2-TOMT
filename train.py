"""
:param model_name: name of a pre-trained bi-encoder from https://huggingface.co/models?library=sentence-transformers of a pre-trained cross-encoder from https://huggingface.co/cross-encoder.
:param loss_fn:  loss function.
:param epochs: number of  epochs
:param warmup_steps: warmup step
:param learn_rate: learning rate passed
:param bm25_topk: number of retrieved bm25 results
:param train_size:number of positive samples of the training set
:param eval_size: number of positive samples of the evaluation set
:param test_size: number of positive samples of the test set
:param train_eval_size: number of positive samples of the training set to evaluate on
:param evals_per_epoch: number of times the evaluation is performed per epoch
"""

from torch.utils.data import DataLoader
from torch.optim import AdamW
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.losses import CosineSimilarityLoss, MultipleNegativesRankingLoss
from data import *
from sentence_transformers.evaluation import BM25IREvaluator
from argparse import ArgumentParser
from bm25_utils import *
from sentence_transformers.datasets import NoDuplicatesDataLoader
parser = ArgumentParser(description='Process some integers.')
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--loss_fn", type=str, default='multi-neg')
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--warmup_steps", type=int, default=0)
parser.add_argument("--learn_rate", type=float, default=2e-5)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--eval_batch_size", type=int, default=200)
parser.add_argument("--bm25_topk", type=int, default=100)
parser.add_argument("--train_size", type=int, required=True)
parser.add_argument("--eval_size", type=int, required=True)
parser.add_argument("--test_size", type=int, required=True)
parser.add_argument("--train_eval_size", type=int, required=True)
parser.add_argument("--eval_steps", type=int, default=None)
parser.add_argument("--evals_per_epoch", type=int, default=1)
parser.add_argument("--cross_encoder", type=str, default=None)
parser.add_argument("--freeze", type=bool, default=False)
args = parser.parse_args()

print("Training setup: ", vars(args))
RERANK = False
cross_encoder = None

if 'cross-encoder' in args.model_name:
    USE_CE = True
    USE_BE = False
    model = CrossEncoder(args.model_name, max_length=512)

elif 'sentence-transformer' in args.model_name:
    USE_BE = True
    USE_CE = False
    model = SentenceTransformer(args.model_name)
    if args.freeze:
        auto_model = model._first_module().auto_model

        for name, param in auto_model.named_parameters():
            if 'weight' in name:
                param.requires_grad = False

    # model = SentenceTransformer(args.model_name)

train_data, train_qrels, eval_qrels, test_qrels, corpus, queries = make_data_splits(
    path = 'data/Movies/',
    train_size = args.train_size,
    eval_size = args.eval_size,
    test_size = args.test_size,
    train_eval_size  = args.train_eval_size,
    splits = [0.8, 0.1, 0.1],
    min_words = 8,
    max_words = 256,
    seed = 32.)

if USE_BE and args.loss_fn == 'multi-neg':
    print('test')
    train_loader = NoDuplicatesDataLoader(train_data, batch_size=args.batch_size)
else:
    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)

print('STEPS PER EPOCH ', len(train_loader) )

# If no eval steps are given evaluate N times per epoch.
if not args.eval_steps:
    args.eval_steps = int(len(train_loader) / args.evals_per_epoch)
    print('EVALS STEPS ', args.eval_steps)

print("\nTotal training data: ", len(train_data),
      "\nNum train queries: ", len(train_qrels),
      "\nNum eval queries: ", len(eval_qrels),
      "\nNum test queries: ", len(test_qrels))

# Training data evaluator
doc_indeces, doc_texts =  disk_load_bm25_docs(corpus, queries, train_qrels, 'train', args.bm25_topk)
query_texts = [queries[qid] for qid in train_qrels]
train_evaluator = BM25IREvaluator(query_texts, doc_texts, doc_indeces, use_cross_encoder=USE_CE, use_bi_encoder=USE_BE , rerank_with_ce=RERANK, cross_encoder=cross_encoder)

# Eval data evaluator
doc_indeces, doc_texts =  disk_load_bm25_docs(corpus, queries, eval_qrels, 'eval', args.bm25_topk)
query_texts = [queries[qid] for qid in eval_qrels]
evaluator = BM25IREvaluator(query_texts, doc_texts, doc_indeces, use_cross_encoder=USE_CE, use_bi_encoder=USE_BE, rerank_with_ce=RERANK, cross_encoder=cross_encoder)

# Test data evaluator
doc_indeces, doc_texts =  disk_load_bm25_docs(corpus, queries, test_qrels, 'test', args.bm25_topk)
query_texts = [queries[qid] for qid in test_qrels]
test_evaluator = BM25IREvaluator(query_texts, doc_texts, doc_indeces, use_cross_encoder=USE_CE, use_bi_encoder=USE_BE, rerank_with_ce=RERANK, cross_encoder=cross_encoder)


general_path = "evaluation/" + args.model_name + "-bm25topk=" + str(args.bm25_topk) + "-datasizes=" + str( [len(x) for x in [train_data, train_qrels, eval_qrels, test_qrels]] )
# if args.rerank_with_ce:
if args.cross_encoder:
    cross_encoder_name =  args.cross_encoder.replace('/', '-')
    general_path = "evaluation/" + args.model_name + cross_encoder_name + "-bm25topk=" + str(args.bm25_topk) + "-datasizes=" + str( [len(x) for x in [train_data, train_qrels, eval_qrels, test_qrels]] )


if USE_BE:
    if not os.path.exists('evaluation/'):
        os.makedirs('evaluation/')

    if args.loss_fn == 'multi-neg':
        loss_fn = MultipleNegativesRankingLoss(model)

    elif args.loss_fn == 'cos-sim':
        loss_fn = CosineSimilarityLoss(model)

    print(loss_fn)

    model.fit(
        train_objectives = [(train_loader, loss_fn)],
        epochs = args.epochs,
        warmup_steps = args.warmup_steps,
        optimizer_class = AdamW,
        optimizer_params = {'lr' : args.learn_rate, 'weight_decay' : args.weight_decay},
        weight_decay = args.weight_decay,
        max_grad_norm = 1.,
        show_progress_bar = False,
        evaluation_steps = args.eval_steps,
        save_best_model = True,
        output_path = general_path + "-" + args.loss_fn,
        evaluator = evaluator,
        train_evaluator = train_evaluator,
        test_evaluator = test_evaluator
        )

elif USE_CE:

    for p in ["", "train_", "test_"]:
        os.makedirs(p + general_path, exist_ok=True)

    model.fit(
        train_dataloader=train_loader,
        evaluator=evaluator,
        train_evaluator=train_evaluator,
        test_evaluator=test_evaluator,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        optimizer_class=AdamW,
        show_progress_bar=False,
        optimizer_params={'lr' : args.learn_rate, 'weight_decay' : args.weight_decay},
        weight_decay=args.weight_decay,
        evaluation_steps=args.eval_steps,
        output_path = general_path,
        train_output_path = "train_" + general_path,
        test_output_path = "test_" + general_path,
        save_best_model=True,
        max_grad_norm=1.)