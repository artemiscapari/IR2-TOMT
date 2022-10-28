# import os

# directory = 'evaluation/multi-qa-distilbert-cos-v1'

from operator import mod
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
# losses = ['online-contrastive', 'cosine-sim', 'contrastive', 'multi-neg']

splits = ['train_', '', 'test_']
# model_name = 'all-MiniLM-L6-v2'
# model_name = 'multi-qa-distilbert-cos-v1'
metrics = {'mrr':'MRR', 'recall@1':'Recall at 1', 'recall@10': 'Recall at 10', 'recall@50': 'Recall at 50'}
split_names = ['train', 'eval', 'test']
sims = ['dot', 'cos']

if not os.path.exists('plots/'):
    os.makedirs('plots/')

def plot_graph(val_dict, sim, metric, metrics, model_name, loss, encoder, fract, top_k):
    fig = go.Figure()
    # fig=make_subplots(
            # specs=[[{"secondary_y": True}]])
    # fig.update_layout(xaxis2= {'anchor': 'y', 'overlaying': 'x', 'side': 'top'},
                    # yaxis_domain=[0, 1]);
    # print(val_dict['train'])
    fig.add_trace(
        go.Scatter(x=val_dict['eval']['steps'],
                y=val_dict['eval'][sim][metric],
                name="Evaluation",
                line_color="blue",
                showlegend=True))

    if 'train' in val_dict:
        print("hello?")
        train = val_dict['train'][sim][metric]
        x_ax = val_dict['train']['steps']

        fig.add_trace(
            go.Scatter(x=x_ax,
                    y=train,
                    name="Train",
                    line_color="red",
                    showlegend=True))

        if 'test' in val_dict:
            y_test_ax = [val_dict['test'][sim][metric][0] for _ in range(len(x_ax))]
            fig.add_trace(
                go.Scatter(x=x_ax,
                        y=y_test_ax,
                        mode='lines',
                        name="Test",
                        line_color="green"))
    print(fract)
    fig.update_layout(
        title = metrics[metric], #+' of '+ model_name+' with '+str(int(float(fract)*100))+'% of the TOMT dataset.',
        width= 700,
        height=475,
        margin=dict(l=20, r=0, t=0, b=20),
        legend=dict(
        title = "Data Set",
        yanchor="bottom",
        y=0.05,
        xanchor="right",
        x=0.95),
        xaxis=dict(range=[0,4]),
        xaxis_title='epochs',
        # xaxis=dict(tickmode='linear', tick0=0, dtick=1),
        yaxis_title= metrics[metric],
        # yaxis= ()
        font=dict(
            family="Courier New, monospace",
            size=12,
            color="Black"
        ),
    )
    # fig.show()
    # exit()
    fig.write_image(f"plots/{encoder}_{model_name}_frac={fract}_bm25={top_k}_{loss}_{sim}_{metric}.png")

def make_plots(split_names, splits, metrics, sims):
    encoders = []
    # for set in splits:
    d = 'evaluation/'
    subdirs = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
    # print(subdirs)
    for e in subdirs:
        encoder = e.split("evaluation/",1)[1]
        encoders.append(d+encoder)
        # print(e.split("evaluation/",1)[0])
    # print("encoders", encoders)

    for e in encoders:

        encoder = e.split("evaluation/",1)[1]
        # print(encoder)

        # d = set+'evaluation/'
        subdirs = [os.path.join(e, o) for o in os.listdir(e) if os.path.isdir(os.path.join(e,o))]

        # directories = []
        for sub in subdirs:
            # print(s)
            print(sub)
            dir = sub.split(encoder+'/')[1]
            model_name = dir.split('-bm25')[0]
            top_k = dir.split('topk=')[1].split('-')[0]
            fract = dir.split('datasizes=')[1].split('-')[0]
            loss = dir.split('datasizes='+fract+'-')[1]




            # print(dir)
            # print("model name", model_name)
            # print("loss", loss)
            # print("fract", fract)
            # print("top k", top_k)

            vals = { }
            vals[fract] = {}

            # for encoders in

            for i,set in enumerate(splits):


                # print(os.listdir(set+'evaluation/')+os.path.isdir(set+'evaluation/'))


                # print(subdirs)
                # print([x[0] for x in os.walk(set+'evaluation/')])
                # print(next(os.walk(set+'evaluation/')))
                # exit()
                eval_dir = set+sub+'/eval/metrics.csv'
                # print("evalllll", eval_dir)

                if os.path.isfile(eval_dir):
                    # print("yes?")

                    # directories.append(eval_dir)
                # else:
                #     break

            # if len(directories) == len(split_names):

                # for i,s in enumerate(split_names):
                    s = split_names[i]


                    # vals[]
                    vals[fract][s] = {'dot':{m:[] for m in metrics }, 'cos':{m:[] for m in metrics}, 'steps':[], 'epochs':[] }
                    increment = 0
                    with open(eval_dir) as f:
                        # print(s)

                        results = f.readlines()

                        increment = 0
                        for step_results in results[1:]:
                            epoch,step,dot_mrr,dot_recall_1,dot_recall_10,dot_recall_50,cos_mrr,cos_recall_1,cos_recall_10,cos_recall_50 = step_results.strip().split(',')
                            epoch = int(epoch)
                            vals[fract][s]['dot']['mrr'] += [ float(dot_mrr)]
                            vals[fract][s]['dot']['recall@1'] += [float(dot_recall_1)]
                            vals[fract][s]['dot']['recall@10'] += [float(dot_recall_10)]
                            vals[fract][s]['dot']['recall@50'] += [float(dot_recall_50)]
                            vals[fract][s]['cos']['mrr'] += [float(cos_mrr)]
                            vals[fract][s]['cos']['recall@1'] += [float(cos_recall_1)]
                            vals[fract][s]['cos']['recall@10'] += [float(cos_recall_10)]
                            vals[fract][s]['cos']['recall@50'] += [float(cos_recall_50)]


                            # print(s, vals[fract][s]['epochs'])
                            # if len(vals[fract][s]['epochs']) != 0:
                            # # if int(epoch) != 0:
                            #     if vals[fract][s]['epochs'][-1] != epoch:
                            # if

                            if epoch not in vals[fract][s]['epochs']:
                                increment = 0
                                vals[fract][s]['epochs'] += [epoch]
                            # else:
                            vals[fract][s]['steps'] += [epoch+increment]
                            if s == 'eval':
                                increment += 0.2
                            #         increment = vals[fract][s]['steps'][-1]




                # else:
                #     del(vals[''])
            # print("HEY", vals['0.1']['eval'])
            for fract in vals:
                # print(fract)
                # print("yooooo", vals[fract])
                for sim in sims:
                    for m in metrics.keys():
                        if 'eval' in vals[fract].keys():
                            # print("hey")
                            plot_graph(vals[fract], sim, m, metrics, model_name, loss, encoder, fract, top_k)

                # else:
            #     vals = {i:{} for i in split_names }
            #     directories = []

make_plots(split_names, splits, metrics, sims)
